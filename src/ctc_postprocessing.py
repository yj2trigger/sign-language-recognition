"""Log-space CTC prefix beam search and explicit train-only lexical reranking."""
from collections import Counter
import math

import numpy as np


def log_probabilities(logits):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("invalid_logits")
    with np.errstate(over="ignore", under="ignore"):
        shifted = values - values.max(axis=1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def checked_probabilities(log_probs, blank_id, length):
    values = np.asarray(log_probs, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] < 2 or np.isnan(values).any()
            or np.isposinf(values).any() or (values > 1e-8).any()
            or not np.isfinite(values).any(axis=1).all()):
        raise ValueError("invalid_log_probabilities")
    if type(blank_id) is not int or not 0 <= blank_id < values.shape[1]:
        raise ValueError("invalid_blank")
    length = len(values) if length is None else length
    if type(length) is not int or not 0 <= length <= len(values):
        raise ValueError("invalid_length")
    return values[:length]


@np.errstate(over="ignore", under="ignore")
def ctc_log_probability(log_probs, target, *, blank_id=0, length=None):
    """Exact sum of every CTC path for one target, with no length bonus."""
    values = checked_probabilities(log_probs, blank_id, length)
    target = tuple(target)
    if any(type(token) is not int or not 0 <= token < values.shape[1] or token == blank_id for token in target):
        raise ValueError("invalid_target")
    states = np.full(len(target) * 2 + 1, blank_id, dtype=np.int64)
    states[1::2] = target
    alpha = np.full(len(states), -np.inf)
    alpha[0] = 0
    skip_allowed = np.zeros(len(states), bool)
    skip_allowed[2:] = (states[2:] != blank_id) & (states[2:] != states[:-2])
    for row in values:
        previous = np.r_[-np.inf, alpha[:-1]]
        skipped = np.r_[-np.inf, -np.inf, alpha[:-2]][:len(alpha)]
        alpha = np.logaddexp(np.logaddexp(alpha, previous),
                             np.where(skip_allowed, skipped, -np.inf)) + row[states]
    return float(np.logaddexp(alpha[-1], alpha[-2])) if target else float(alpha[0])


@np.errstate(over="ignore", under="ignore")
def prefix_beam_search(log_probs, *, beam_width=10, blank_id=0, length=None):
    """Full-alphabet CTC search retaining separate blank/nonblank masses.

    Vectorized expansions avoid constructing discarded prefix tuples. Output
    mass is the retained-path lower bound after pruning, not a normalized
    posterior over strings. Exact final scores are available separately.
    """
    values = checked_probabilities(log_probs, blank_id, length)
    if type(beam_width) is not int or beam_width < 1:
        raise ValueError("invalid_beam_width")
    tokens = [index for index in range(values.shape[1]) if index != blank_id]
    token_columns = {token: column for column, token in enumerate(tokens)}
    prefixes = [()]
    blank = np.array([0.0])
    nonblank = np.array([-np.inf])
    for row in values:
        total = np.logaddexp(blank, nonblank)
        expanded_nonblank = total[:, None] + row[tokens][None]
        expanded_blank = np.full_like(expanded_nonblank, -np.inf)
        old_blank = total + row[blank_id]
        old_nonblank = np.full(len(prefixes), -np.inf)
        lookup = {prefix: index for index, prefix in enumerate(prefixes)}
        for index, prefix in enumerate(prefixes):
            if prefix:
                column = token_columns[prefix[-1]]
                # Repeating without a blank keeps the original prefix.
                old_nonblank[index] = nonblank[index] + row[prefix[-1]]
                # Extending by the same token requires an earlier blank.
                expanded_nonblank[index, column] = blank[index] + row[prefix[-1]]
        for index, prefix in enumerate(prefixes):
            parent = lookup.get(prefix[:-1]) if prefix else None
            if parent is not None:
                column = token_columns[prefix[-1]]
                expanded_nonblank[parent, column] = np.logaddexp(
                    expanded_nonblank[parent, column], old_nonblank[index])
                expanded_blank[parent, column] = old_blank[index]
                old_blank[index] = old_nonblank[index] = -np.inf
        candidate_blank = np.r_[expanded_blank.ravel(), old_blank]
        candidate_nonblank = np.r_[expanded_nonblank.ravel(), old_nonblank]
        score = np.logaddexp(candidate_blank, candidate_nonblank)
        # Stable tie order is deterministic and independent of target labels.
        selected = np.argsort(-score, kind="stable")[:beam_width]
        selected = selected[np.isfinite(score[selected])]
        expanded_count = expanded_nonblank.size
        next_prefixes = []
        for index in selected:
            if index < expanded_count:
                parent, column = divmod(int(index), len(tokens))
                next_prefixes.append(prefixes[parent] + (tokens[column],))
            else:
                next_prefixes.append(prefixes[int(index) - expanded_count])
        prefixes = next_prefixes
        blank, nonblank = candidate_blank[selected], candidate_nonblank[selected]
    return [dict(ids=prefix, beam_log_mass=float(score))
            for prefix, score in zip(prefixes, np.logaddexp(blank, nonblank))]


def exact_ranked_beam(log_probs, *, beam_width=10, blank_id=0):
    candidates = prefix_beam_search(log_probs, beam_width=beam_width, blank_id=blank_id)
    for candidate in candidates:
        candidate["ctc_log_probability"] = ctc_log_probability(log_probs, candidate["ids"], blank_id=blank_id)
    return sorted(candidates, key=lambda row: (-row["ctc_log_probability"], row["ids"]))


def training_lexicon(training_records, encoder):
    """Only the caller's training records contribute vocabulary or counts."""
    counts = Counter(row["text"] for row in training_records)
    return [dict(text=text, count=count, ids=encoder.encode(text).ids)
            for text, count in sorted(counts.items())]


def lexical_candidates(log_probs, candidates, lexicon, *, edit_distance, max_edits=3,
                       neighbors=3, source_candidates=3):
    """Add nearby training words, score each with the same exact CTC forward."""
    combined = {tuple(row["ids"]): dict(row, lexical_count=0, lexical_word=None) for row in candidates}
    nearby = {}
    for source in candidates[:source_candidates]:
        ranked = sorted((edit_distance(source["ids"], row["ids"]), row["text"], row)
                        for row in lexicon)
        for distance, _, word in ranked[:neighbors]:
            if distance <= max_edits:
                nearby[word["text"]] = word
    for word in nearby.values():
        ids = tuple(word["ids"])
        score = ctc_log_probability(log_probs, ids)
        if not np.isfinite(score):
            continue
        if ids not in combined:
            combined[ids] = dict(ids=ids, ctc_log_probability=score, beam_log_mass=None)
        combined[ids].update(lexical_count=word["count"], lexical_word=word["text"])
    # Existing beam words also receive their training-count support.
    known = {tuple(row["ids"]): row for row in lexicon}
    for ids, candidate in combined.items():
        if ids in known:
            candidate.update(lexical_count=known[ids]["count"], lexical_word=known[ids]["text"])
    return list(combined.values())


def lexical_rerank(candidates, weight):
    """Exact acoustic log probability plus weight * log(1 + training count)."""
    if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError("invalid_lexical_weight")
    ranked = [dict(row, combined_score=row["ctc_log_probability"] +
                   weight * math.log1p(row["lexical_count"])) for row in candidates]
    return sorted(ranked, key=lambda row: (-row["combined_score"], row["ids"]))


def token_edit_diagnostics(reference, prediction):
    """Deterministic minimum-edit alignment for diagnostic token confusions."""
    costs = [[0] * (len(prediction) + 1) for _ in range(len(reference) + 1)]
    for i in range(len(reference) + 1):
        costs[i][0] = i
    for j in range(len(prediction) + 1):
        costs[0][j] = j
    for i, expected in enumerate(reference, 1):
        for j, actual in enumerate(prediction, 1):
            costs[i][j] = min(costs[i - 1][j - 1] + (expected != actual),
                              costs[i - 1][j] + 1, costs[i][j - 1] + 1)
    substitutions, deletions, insertions = [], [], []
    i, j = len(reference), len(prediction)
    while i or j:
        if i and j and costs[i][j] == costs[i - 1][j - 1] + (reference[i - 1] != prediction[j - 1]):
            if reference[i - 1] != prediction[j - 1]:
                substitutions.append((reference[i - 1], prediction[j - 1]))
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            deletions.append(reference[i - 1])
            i -= 1
        else:
            insertions.append(prediction[j - 1])
            j -= 1
    return dict(substitutions=substitutions, deletions=deletions, insertions=insertions)
