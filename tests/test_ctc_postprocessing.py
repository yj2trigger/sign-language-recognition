"""Exhaustive-path oracles and leakage/fallback checks for CTC decoding."""
from collections import defaultdict
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ctc_postprocessing as decoder


def collapse(path, blank=0):
    return tuple(token for index, token in enumerate(path)
                 if token != blank and (index == 0 or token != path[index - 1]))


@pytest.mark.parametrize("blank", [0, 1, 2])
def test_prefix_and_exact_forward_equal_exhaustive_path_sum(blank):
    probabilities = np.array([[.4, .35, .25], [.2, .5, .3], [.6, .1, .3], [.2, .4, .4]])
    expected = defaultdict(float)
    for path in product(range(3), repeat=4):
        expected[collapse(path, blank)] += np.prod([probabilities[t, token] for t, token in enumerate(path)])
    logp = np.log(probabilities)
    candidates = decoder.prefix_beam_search(logp, beam_width=1000, blank_id=blank)
    actual = {row["ids"]: np.exp(row["beam_log_mass"]) for row in candidates}
    assert set(actual) == set(expected)
    for target, probability in expected.items():
        assert actual[target] == pytest.approx(probability)
        assert np.exp(decoder.ctc_log_probability(logp, target, blank_id=blank)) == pytest.approx(probability)
    assert sum(actual.values()) == pytest.approx(1)


def test_repeat_with_and_without_intervening_blank():
    logp = np.array([[-np.inf, 0], [0, -np.inf], [-np.inf, 0]])
    assert decoder.prefix_beam_search(logp)[0]["ids"] == (1, 1)
    assert decoder.ctc_log_probability(logp, (1, 1)) == 0
    assert decoder.ctc_log_probability(logp[:2], (1, 1)) == -np.inf
    assert decoder.prefix_beam_search(logp[[0, 2]])[0]["ids"] == (1,)


def test_empty_length_padding_and_all_blank():
    values = np.array([[0, -np.inf], [0, -np.inf], [-np.inf, 0]])
    assert decoder.prefix_beam_search(values, length=2)[0]["ids"] == ()
    assert decoder.ctc_log_probability(values, (), length=2) == 0
    assert decoder.prefix_beam_search(values, length=0) == [dict(ids=(), beam_log_mass=0.0)]
    assert decoder.ctc_log_probability(values, (), length=0) == 0
    assert decoder.ctc_log_probability(values, (1,), length=0) == -np.inf


def test_extreme_logits_are_stable_and_scores_sorted():
    logits = np.array([[1e308, -1e308, 0], [-1e308, 1e308, 0]])
    logp = decoder.log_probabilities(logits)
    assert not np.isnan(logp).any()
    assert decoder.exact_ranked_beam(logp)[0]["ids"] == (1,)
    regular = decoder.exact_ranked_beam(decoder.log_probabilities(np.zeros((4, 3))), beam_width=5)
    scores = [row["ctc_log_probability"] for row in regular]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_logits_rejected(value):
    with pytest.raises(ValueError, match="invalid_logits"):
        decoder.log_probabilities([[0, value]])


@pytest.mark.parametrize("width", [0, -1, True, 1.5])
def test_invalid_beam_width(width):
    with pytest.raises(ValueError, match="invalid_beam_width"):
        decoder.prefix_beam_search(np.log([[.5, .5]]), beam_width=width)


def test_lexicon_uses_only_training_text_and_oov_fallback():
    from fingerspelling_runtime import vocabulary
    from fingerspelling.ctc_targets import TargetEncoder
    encoder = TargetEncoder(vocabulary())
    lexicon = decoder.training_lexicon([dict(text="가"), dict(text="가"), dict(text="나")], encoder)
    assert {row["text"]: row["count"] for row in lexicon} == {"가": 2, "나": 1}
    assert "학교" not in {row["text"] for row in lexicon}
    logp = decoder.log_probabilities(np.zeros((3, 52)))
    candidate = dict(ids=(), beam_log_mass=-1, ctc_log_probability=-1)
    combined = decoder.lexical_candidates(logp, [candidate], lexicon,
                                          edit_distance=lambda a, b: 100)
    assert decoder.lexical_rerank(combined, 8)[0]["ids"] == ()


def test_lexical_score_cannot_use_a_reference_argument():
    candidates = [dict(ids=(1,), ctc_log_probability=-2, lexical_count=0),
                  dict(ids=(2,), ctc_log_probability=-3, lexical_count=3)]
    assert decoder.lexical_rerank(candidates, 0)[0]["ids"] == (1,)
    assert decoder.lexical_rerank(candidates, 1)[0]["ids"] == (2,)
    with pytest.raises(ValueError, match="invalid_lexical_weight"):
        decoder.lexical_rerank(candidates, float("nan"))


@pytest.mark.parametrize("reference,prediction", [("ㄱㅏㄴㅏ", "ㄱㅓㄴ"), ("", "ㄱ"), ("ㄱ", ""), ("ㄱㄱ", "ㄱ")])
def test_token_confusions_equal_minimum_edit_distance(reference, prediction):
    from fingerspelling.cer_metrics import edit_counts
    errors = decoder.token_edit_diagnostics(reference, prediction)
    assert sum(len(value) for value in errors.values()) == edit_counts(reference, prediction).errors


@pytest.mark.parametrize("length", [-1, 5, True])
def test_invalid_input_length(length):
    with pytest.raises(ValueError, match="invalid_length"):
        decoder.prefix_beam_search(np.log(np.full((4, 3), 1 / 3)), length=length)


def test_candidate_json_roundtrip_reproduces_scores():
    import json
    values = decoder.log_probabilities(np.array([[1, 3, 2], [3, 1, 2], [1, 3, 2]]))
    first = decoder.exact_ranked_beam(values, beam_width=10)
    second = decoder.exact_ranked_beam(values, beam_width=10)
    assert json.dumps(first, allow_nan=False) == json.dumps(second, allow_nan=False)
    restored = json.loads(json.dumps(first))
    assert [row["ctc_log_probability"] for row in restored] == [row["ctc_log_probability"] for row in first]
