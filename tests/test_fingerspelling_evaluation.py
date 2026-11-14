from math import isinf
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fingerspelling.ctc_decode import DecodeError, TokenRenderer
from fingerspelling.cer_metrics import ErrorCounts, MetricError, TextPolicy
from fingerspelling.evaluation import (EvaluationError, Reference, evaluate_scores,
                                       evaluate_texts, validate_evaluation_data)


def test_corpus_vs_mean_and_performer():
    rows = [Reference('v1', 'a', 'p1'), Reference('v2', 'abcdefghi', 'p2')]
    report = evaluate_texts(rows, {'v2': 'abcdefghi', 'v1': ''},
        training_performer_ids=['train'], identities_verified=True, require_verified=True)
    assert report.corpus == ErrorCounts(0, 1, 0, 10)
    assert report.corpus.cer == .1
    assert report.mean_video_cer == .5
    assert report.by_performer['p1'].cer == 1
    assert report.by_performer['p2'].cer == 0
    assert report.identity_status == 'verified'
    with pytest.raises(TypeError):
        report.by_performer['new'] = ErrorCounts()


def test_full_pipeline_synthetic_keyed_lengths():
    # Frames: a,a,blank,a,b,b -> aab; padding a ignored in second video.
    rows = [Reference('v1', 'aab', 'p'), Reference('v2', 'a', 'q')]
    scores = {'v2': [[0, 0, 3], [0, 3, 0]],
              'v1': [[0, 3, 0], [0, 2, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3], [0, 0, 2]]}
    report = evaluate_scores(rows, scores, renderer=TokenRenderer({'a': 1, 'b': 2}, blank_id=0),
        lengths={'v1': 6, 'v2': 1}, training_performer_ids=['train'], identities_verified=True)
    assert report.corpus == ErrorCounts(1, 0, 0, 4)
    assert [v.counts.cer for v in report.videos] == [0, 1]


def test_pipeline_composition_and_nfc():
    # Table fixture verifies callback wiring, not a second Hangul algorithm.
    def compose(tokens):
        assert tokens == ['ㄱ', 'ㅏ']
        return '가'
    renderer = TokenRenderer({'ㄱ': 1, 'ㅏ': 2}, blank_id=0, compose=compose)
    report = evaluate_scores([Reference('v', '가')], {'v': [[0, 3, 0], [0, 0, 3]]}, renderer=renderer)
    assert report.corpus == ErrorCounts(reference_characters=1)
    assert report.identity_status == 'unverified'


@pytest.mark.parametrize("rows,pred,code", [
    ([Reference('v', 'a'), Reference('v', 'b')], {'v': ''}, 'duplicate_video_id'),
    ([Reference('v', None)], {'v': ''}, 'missing_reference'),
    ([Reference('v', 'a')], {}, 'missing_prediction'),
    ([Reference('v', 'a')], {'v': None}, 'missing_prediction'),
    ([Reference('v', 'a')], {'v': '', 'x': ''}, 'unexpected_prediction'),
    ([Reference('v', 'a')], [('v', ''), ('v', '')], 'duplicate_prediction_id'),
    ([Reference('', 'a')], {}, 'invalid_identity'),
    ([Reference('v', 'a', '')], {'v': ''}, 'invalid_identity'),
    ([{}], {}, 'invalid_reference'),
    (None, {}, 'invalid_references'),
    ([Reference('v', 'a')], ['vx'], 'invalid_prediction_pair'),
])
def test_validation(rows, pred, code):
    with pytest.raises(EvaluationError, match='^' + code + '$'):
        evaluate_texts(rows, pred)


def test_leakage_always_rejected_even_unverified():
    rows = [Reference('v', 'a', 'person')]
    for kwargs, code in [({'training_performer_ids': ['person']}, 'performer_leakage'),
                          ({'training_video_ids': ['v']}, 'video_leakage')]:
        with pytest.raises(EvaluationError, match=code):
            evaluate_texts(rows, {'v': 'a'}, **kwargs)


def test_unverified_identity_is_explicit():
    rows = [Reference('v', 'a', 'CROWD01')]
    report = evaluate_texts(rows, {'v': 'a'}, training_performer_ids=['CROWD02'])
    assert report.identity_status == 'unverified'
    assert report.unverified_reasons == ('identity_provenance_unverified',)
    for kwargs in ({}, {'identities_verified': True},
                   {'training_performer_ids': [], 'identities_verified': False}):
        with pytest.raises(EvaluationError, match='held_out_unverified'):
            evaluate_texts(rows, {'v': ''}, require_verified=True, **kwargs)
    report = evaluate_texts([Reference('v', '')], {'v': ''},
                            training_performer_ids=[], identities_verified=True)
    assert 'missing_performer_identity' in report.unverified_reasons
    assert None in report.by_performer


def test_empty_corpus_and_zero_denominators():
    report = evaluate_texts([], {}, training_performer_ids=[], identities_verified=True)
    assert report.mean_video_cer is None
    assert report.corpus.cer == 0
    assert report.identity_status == 'unverified'
    report = evaluate_texts([Reference('v', '')], {'v': 'abc'})
    assert isinf(report.corpus.cer) and isinf(report.mean_video_cer)
    report = evaluate_texts([Reference('v', ''), Reference('w', 'a')], {'v': 'xx', 'w': 'a'})
    assert report.corpus.cer == 2 and isinf(report.mean_video_cer)


def test_thousands_of_records_generator():
    rows = (Reference(str(i), 'ab', 'p' + str(i % 10)) for i in range(5000))
    pred = ((str(i), 'ac' if i % 2 else 'ab') for i in reversed(range(5000)))
    report = evaluate_texts(rows, pred)
    assert report.corpus == ErrorCounts(2500, 0, 0, 10000)
    assert len(report.videos) == 5000
    assert len(report.by_performer) == 10
    assert report.by_performer['p1'].substitutions == 500


def test_bad_options_and_types():
    for kwargs in ({'training_performer_ids': 'p'}, {'training_video_ids': [1]},
                   {'identities_verified': 1}, {'require_verified': 1}, {'policy': None}):
        with pytest.raises(EvaluationError):
            evaluate_texts([Reference('v', '')], {'v': ''}, **kwargs)
    for invalid in (1, [], '\ud800'):
        with pytest.raises(MetricError):
            evaluate_texts([Reference('v', '')], {'v': invalid})
    for lengths in ({}, {'v': None}, {'v': True}, {'v': 1}, []):
        with pytest.raises((DecodeError, EvaluationError)):
            evaluate_scores([Reference('v', '')], {'v': []},
                            renderer=TokenRenderer({'a': 1}, blank_id=0), lengths=lengths)


def test_reports_hide_text_and_ids_from_repr():
    row = Reference('PRIVATE_VIDEO', 'PRIVATE_LABEL', 'PRIVATE_PERSON')
    report = evaluate_texts([row], {'PRIVATE_VIDEO': ''})
    assert 'PRIVATE' not in repr(row) + repr(report) + repr(report.videos)
    assert evaluate_texts([Reference('v', '가')], {'v': '가'}, policy=TextPolicy(False)).corpus.cer == 2


def test_reviewed_pr11_pr13_interfaces_in_memory():
    """Execute pinned, reviewed local Git objects only; never copy PR sources.

    Optional on clean clones without these objects. Diagnostic Unicode tokens
    below are NOT an official 37-token inventory.
    """
    import subprocess
    import types
    root = Path(__file__).resolve().parents[1]
    modules = []
    for ref, path in [
        ('8f23314c7f924081f38d2b34b65996931398bb7c', 'src/fingerspelling/jamo.py'),
        ('b37954ac6541d0d80a47fd0ad6a9a2b2eb38b6d8', 'src/fingerspelling/tokens.py'),
    ]:
        source = subprocess.run(['git', 'show', ref + ':' + path], cwd=root,
                                capture_output=True)
        if source.returncode:
            pytest.skip('Pinned unmerged PR objects are unavailable in this clone')
        module = types.ModuleType('reviewed_contract')
        exec(compile(source.stdout.decode('utf-8'), '<reviewed-contract>', 'exec'), module.__dict__)
        modules.append(module)
    jamo, tokens = modules
    vocab = tokens.TokenVocabulary({t: i + 1 for i, t in enumerate(sorted(jamo.JAMO))}, blank_id=0)
    renderer = TokenRenderer.from_vocabulary(vocab, compose=jamo.compose_hangul)
    for code in range(0xAC00, 0xD7A4):
        text = chr(code)
        assert renderer.render(vocab.encode(jamo.decompose_hangul(text))) == text
    path = vocab.encode(jamo.decompose_hangul('가각'))
    scores = []
    for index in path:
        scores.append([int(c == index) for c in range(vocab.num_classes)])
        scores.append([int(c == vocab.blank_id) for c in range(vocab.num_classes)])
    report = evaluate_scores([Reference('synthetic', '가각')], {'synthetic': scores}, renderer=renderer)
    assert report.corpus == ErrorCounts(reference_characters=2)


@pytest.mark.parametrize('options,code', [
    ({'training_performer_ids': ['person']}, 'performer_leakage'),
    ({'training_video_ids': ['v']}, 'video_leakage'),
    ({'require_verified': True}, 'held_out_unverified'),
    ({'policy': None}, 'invalid_evaluation_policy'),
])
def test_review_regression_validation_precedes_decoding(options, code):
    calls = []
    def compose(tokens):
        calls.append(tokens)
        return 'a'
    def forbidden_scores():
        pytest.fail('Scores consumed before invalid metadata was rejected')
        yield []
    renderer = TokenRenderer({'a': 1}, blank_id=0, compose=compose)
    with pytest.raises(EvaluationError, match='^' + code + '$'):
        evaluate_scores([Reference('v', 'a', 'person')], {'v': forbidden_scores()},
                        renderer=renderer, **options)
    assert calls == []


def test_review_regression_training_inventory_consumed_once():
    class OneShot:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def __iter__(self):
            self.calls += 1
            assert self.calls == 1, 'Training inventory was consumed more than once'
            return iter([self.value])

    performers, videos = OneShot('train'), OneShot('train_video')
    report = evaluate_scores([Reference('v', 'a', 'person')], {'v': [[0, 1]]},
        renderer=TokenRenderer({'a': 1}, blank_id=0),
        training_performer_ids=performers, training_video_ids=videos,
        identities_verified=True, require_verified=True)
    assert report.identity_status == 'verified'
    assert report.corpus.cer == 0
    assert performers.calls == videos.calls == 1
