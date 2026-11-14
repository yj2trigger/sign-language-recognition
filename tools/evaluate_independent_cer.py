"""Execute existing CTC/CER components after independent-evidence checks."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from build_independent_review_handoff import canonical, eligibility, index_unique
from process_crowd_blocks import digest
from fingerspelling.cer_metrics import ErrorCounts, edit_counts
from fingerspelling.ctc_decode import greedy_decode
from fingerspelling.ctc_targets import TargetEncoder
from fingerspelling.sequence_validation import validate_sequence
from fingerspelling_runtime import FingerspellingReader, features, render, tokens_for, vocabulary


def load_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def evidence_file(root, row, source_field):
    source = row.get(source_field, '')
    path = (root / source).resolve()
    if not source or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('local evidence file missing or outside evidence root')
    observed_hash = digest(path)
    if observed_hash != row.get('evidence_sha256'):
        raise ValueError('evidence hash mismatch')
    return dict(path=str(path), sha256=observed_hash)


def candidate_features(row, feature_root):
    vid = canonical(row['video_id'])
    path = feature_root / (vid + '.npz')
    if digest(path) != row['feature_sha256']:
        raise ValueError('evaluation feature hash mismatch')
    with np.load(path, allow_pickle=False) as saved:
        xy, mask, times = saved['xy'].copy(), saved['mask'].copy(), saved['t'].copy()
    if digest(path) != row['feature_sha256']:
        raise ValueError('evaluation feature changed while reading')
    validate_sequence(xy, mask, allow_all_missing=True)
    if len(xy) < 2 or not mask[:, 11:].any():
        raise ValueError('no usable observed hands')
    if times.shape != (len(xy),) or not np.isfinite(times).all():
        raise ValueError('invalid feature PTS')
    if not np.allclose(np.diff(times), 1 / 15, atol=.002, rtol=0):
        raise ValueError('feature requires explicit 15fps policy')
    start, end = float(row['start']), float(row['end'])
    if abs(times[0] - start) > 1 / 15 + .002 or abs(times[-1] - end) > 1 / 15 + .002:
        raise ValueError('features do not cover verified transcript interval')
    text = row['transcript_text']
    if not text or not all('\uac00' <= char <= '\ud7a3' for char in text):
        raise ValueError('independent transcript must contain complete Hangul syllables')
    features(xy, mask)
    target = TargetEncoder(vocabulary()).encode(text, input_length=len(xy))
    return xy, mask, ''.join(target.tokens)


def checkpoint_predictor(checkpoint, device):
    reader = FingerspellingReader(checkpoint, device=device)
    def predict(xy, mask, reference):
        values = features(xy, mask)
        if reader.feature_transform is not None:
            values = reader.feature_transform(values)
        scores, lengths = reader.recognizer.predict(values[None], [len(values)])
        output_length = int(lengths[0])
        TargetEncoder(vocabulary()).encode(reference, input_length=output_length)
        ids = greedy_decode(scores[0].tolist(), blank_id=0,
                            num_classes=reader.recognizer.model.config.num_classes, length=output_length)
        return ''.join(tokens_for(ids)), render(ids), output_length
    return predict


def evaluate(config, mappings, reviews, feature_root, evidence_root, predictor_factory):
    """Evidence authenticity requires provider/reviewer approval outside this tool.

    Injected predictors are for synthetic tests. CLI always uses the pinned model.
    Every candidate is checked before any model initialization or inference.
    """
    train = [row['video_id'] for row in config['splits']['train']]
    hashes = [config['feature_hashes'][vid] for vid in train]
    gate = eligibility(train, mappings, reviews, hashes)
    mapping_index = index_unique(mappings); review_index = index_unique(reviews)
    prepared = []; evidence = []
    for decision in gate['decisions']:
        if not decision['declared_ready']:
            continue
        vid = decision['video_id']; row = review_index[vid]
        try:
            proofs = [evidence_file(evidence_root, mapping_index[identity], 'mapping_source')
                      for identity in [*train, vid]]
            proofs.append(evidence_file(evidence_root, row, 'evidence_source'))
            xy, mask, jamo = candidate_features(row, feature_root)
            prepared.append((vid, row, xy, mask, jamo))
            evidence.extend(proofs)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            decision['declared_ready'] = False
            decision['reasons'].append('ACTUAL_EVIDENCE_OR_FEATURE_REJECTED: ' + str(exc))
    hash_counts = {}
    for _, row, _, _, _ in prepared:
        key = row['feature_sha256']
        hash_counts[key] = hash_counts.get(key, 0) + 1
    duplicates = {row[0] for row in prepared if hash_counts[row[1]['feature_sha256']] > 1}
    for decision in gate['decisions']:
        if decision['video_id'] in duplicates:
            decision['declared_ready'] = False
            decision['reasons'].append('EVALUATION_FEATURE_DUPLICATE')
    prepared = [row for row in prepared if row[0] not in duplicates]
    gate['declared_ready_count'] = len(prepared)
    result = dict(status='BLOCKED', gate=gate, independent_evaluation_videos=0,
                  independent_performer_count=0, jamo_CER=None, hangul_CER=None,
                  exact_match=None, videos=[], evidence=evidence,
                  interpretation='Accepted evidence declarations and hashes require external authenticity approval.')
    if not prepared:
        return result
    predict = predictor_factory()
    total_jamo = ErrorCounts(); total_hangul = ErrorCounts(); exact = 0
    people = set()
    for vid, row, xy, mask, reference_jamo in prepared:
        predicted_jamo, predicted_text, output_steps = predict(xy, mask, row['transcript_text'])
        if not isinstance(predicted_jamo, str) or not isinstance(predicted_text, str):
            raise ValueError('predictor must return text')
        jamo_counts = edit_counts(reference_jamo, predicted_jamo)
        hangul_counts = edit_counts(row['transcript_text'], predicted_text)
        total_jamo += jamo_counts; total_hangul += hangul_counts
        exact += row['transcript_text'] == predicted_text
        person = mapping_index[vid]['performer_id'].strip(); people.add(person)
        result['videos'].append(dict(video_id=vid, performer_id=person, reference=row['transcript_text'],
            predicted_jamo=predicted_jamo, predicted_text=predicted_text, input_frames=len(xy),
            output_steps=output_steps, jamo=asdict(jamo_counts), hangul=asdict(hangul_counts)))
    result.update(status='MEASURED', independent_evaluation_videos=len(prepared),
                  independent_performer_count=len(people), jamo_CER=total_jamo.cer,
                  hangul_CER=total_hangul.cer, exact_match=exact / len(prepared),
                  jamo_counts=asdict(total_jamo), hangul_counts=asdict(total_hangul))
    return result


def run(config_path, model_result, checkpoint, mapping_csv, review_csv, feature_root, evidence_root, device):
    provenance = json.loads(model_result.read_text(encoding='utf-8'))
    if provenance['config']['source_config_sha256'] != digest(config_path):
        raise ValueError('training configuration is not bound to model result')
    if provenance['inference_sha256'] != digest(checkpoint):
        raise ValueError('checkpoint is not bound to model result')
    input_paths = (config_path, model_result, checkpoint, mapping_csv, review_csv)
    initial_hashes = {str(path): digest(path) for path in input_paths}
    config = json.loads(config_path.read_text(encoding='utf-8'))
    result = evaluate(config, load_csv(mapping_csv), load_csv(review_csv), feature_root, evidence_root,
                      lambda: checkpoint_predictor(checkpoint, device))
    if initial_hashes != {str(path): digest(path) for path in input_paths}:
        raise ValueError('evaluation inputs changed while running')
    result['inputs'] = initial_hashes
    result['device'] = device
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('training-config', 'model-result', 'checkpoint', 'mapping-csv', 'review-csv',
                'feature-root', 'evidence-root', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    if any(value.resolve().drive.upper() != 'D:' for value in vars(args).values() if isinstance(value, Path)):
        parser.error('D paths required')
    if args.output.exists():
        parser.error('output already exists')
    result = run(args.training_config, args.model_result, args.checkpoint, args.mapping_csv,
                 args.review_csv, args.feature_root, args.evidence_root, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in ('status', 'independent_evaluation_videos', 'jamo_CER', 'hangul_CER')}))
