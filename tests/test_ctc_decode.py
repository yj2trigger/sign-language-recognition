"""Synthetic scores and externally supplied diagnostic tokens only."""
import copy
import itertools
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fingerspelling.ctc_decode import DecodeError, TokenRenderer, greedy_decode, greedy_decode_batch


def scores(path, width=3):
    return [[int(i == token) for i in range(width)] for token in path]


@pytest.mark.parametrize("path,expected", [([], ()), ([0, 0], ()),
    ([1, 1, 0, 1, 2, 2, 0], (1, 1, 2)), ([2, 0, 2], (2, 2)),
    ([1, 2, 1], (1, 2, 1))])
def test_collapse(path, expected):
    assert greedy_decode(scores(path), blank_id=0, num_classes=3) == expected


def test_exhaustive_short_paths():
    # Independent oracle: group path first, then filter blank.
    for size in range(7):
        for path in itertools.product(range(3), repeat=size):
            expected = tuple(k for k, _ in itertools.groupby(path) if k != 0)
            assert greedy_decode(scores(path), blank_id=0, num_classes=3) == expected


def test_batch_lengths_and_immutability():
    batch = [scores([1, 0, 1]), scores([2, 2]), []]
    before = copy.deepcopy(batch)
    assert greedy_decode_batch(batch, blank_id=0, num_classes=3, lengths=[2, 1, 0]) == ((1,), (2,), ())
    assert batch == before
    assert greedy_decode_batch([], blank_id=0, num_classes=3, lengths=[]) == ()
    assert greedy_decode_batch(iter(batch), blank_id=0, num_classes=3) == ((1, 1), (2,), ())


def test_ties_nonzero_blank_and_long_sequence():
    assert greedy_decode([[1, 1, 1], [2, 2, 0]], blank_id=2, num_classes=3) == (0,)
    assert greedy_decode(scores([1, 0] * 10000), blank_id=0, num_classes=3) == (1,) * 10000


@pytest.mark.parametrize("value", [float('nan'), float('inf'), -float('inf'), True, '1', None, 10**1000])
def test_bad_scores_even_in_padding(value):
    with pytest.raises(DecodeError):
        greedy_decode([[0, value]], blank_id=0, num_classes=2, length=0)


@pytest.mark.parametrize("kwargs", [dict(blank_id=-1, num_classes=2), dict(blank_id=True, num_classes=2),
    dict(blank_id=2, num_classes=2), dict(blank_id=0, num_classes=0), dict(blank_id=0, num_classes=2.0),
    dict(blank_id=0, num_classes=2, length=-1), dict(blank_id=0, num_classes=2, length=True),
    dict(blank_id=0, num_classes=2, length=1)])
def test_bad_contract(kwargs):
    with pytest.raises(DecodeError):
        greedy_decode([], **kwargs)


@pytest.mark.parametrize("rows", [None, 'text', {}, [None], [[0]], [[0, 1, 2]], [[0, 1], []]])
def test_bad_shape(rows):
    with pytest.raises(DecodeError):
        greedy_decode(rows, blank_id=0, num_classes=2)


@pytest.mark.parametrize("lengths", [[], [0, 0], [None], [True], [-1], [2], '1', 1])
def test_bad_lengths(lengths):
    with pytest.raises(DecodeError):
        greedy_decode_batch([[[0, 1]]], blank_id=0, num_classes=2, lengths=lengths)


def test_renderer_sparse_contract():
    source = {'가': 2, '1 ': 4}
    vocab = SimpleNamespace(token_ids=source, blank_id=1, num_classes=5)
    renderer = TokenRenderer.from_vocabulary(vocab)
    source['가'] = 99
    assert renderer.render([1, 2, 2, 4, 1]) == '가가1 '
    assert renderer.tokens([2, 4]) == ('가', '1 ')
    for invalid in ([0], [3], [5], [-1], [True], [1.0], None, '2'):
        with pytest.raises(DecodeError):
            renderer.render(invalid)
    vocab.num_classes = 6
    with pytest.raises(DecodeError):
        TokenRenderer.from_vocabulary(vocab)


@pytest.mark.parametrize("mapping,blank", [({}, 0), ({'': 1}, 0), ({'a': 0}, 0),
    ({'a': 1, 'b': 1}, 0), ({'a': -1}, 0), ({'a': True}, 0), ({'a': 1}, True)])
def test_invalid_renderer(mapping, blank):
    with pytest.raises(DecodeError):
        TokenRenderer(mapping, blank_id=blank)


def test_composer_contract_and_sanitized_errors():
    seen = []
    def compose(tokens):
        assert isinstance(tokens, list)
        seen.append(tokens[:])
        tokens.clear()
        return '가'
    renderer = TokenRenderer({'ㄱ': 1, 'ㅏ': 2}, blank_id=0, compose=compose)
    assert renderer.render([1, 2]) == '가'
    assert renderer.render([1, 2]) == '가'
    assert seen == [['ㄱ', 'ㅏ'], ['ㄱ', 'ㅏ']]
    def reject(tokens):
        raise ValueError('PRIVATE')
    with pytest.raises(DecodeError, match='^composition_rejected$') as error:
        TokenRenderer({'a': 1}, blank_id=0, compose=reject).render([1])
    assert error.value.__suppress_context__
    with pytest.raises(DecodeError, match='invalid_composer_result'):
        TokenRenderer({'a': 1}, blank_id=0, compose=lambda x: None).render([])


def test_numpy_without_runtime_dependency():
    np = pytest.importorskip('numpy')
    array = np.array([[[0., 1.], [1., 0.], [0., 1.]]], dtype=np.float32)
    before = array.copy()
    assert greedy_decode_batch(array, blank_id=np.int64(0), num_classes=2,
                               lengths=np.array([3])) == ((1, 1),)
    assert np.array_equal(array, before)
