import itertools
from math import isinf
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fingerspelling.cer_metrics import ErrorCounts, MetricError, TextPolicy, character_error_rate, edit_counts


@pytest.mark.parametrize("ref,pred,expected", [('abc', 'axc', (1, 0, 0, 3)),
    ('abc', 'ac', (0, 1, 0, 3)), ('ac', 'abc', (0, 0, 1, 2)),
    ('kitten', 'sitting', (2, 0, 1, 6)), ('', '', (0, 0, 0, 0)),
    ('', 'abc', (0, 0, 3, 0)), ('abc', '', (0, 3, 0, 3)),
    ('ab', 'ba', (2, 0, 0, 2))])
def test_known_counts(ref, pred, expected):
    assert edit_counts(ref, pred) == ErrorCounts(*expected)


def test_empty_and_unbounded_cer():
    assert character_error_rate('', '') == 0
    assert isinf(character_error_rate('', 'x'))
    assert character_error_rate('a', 'aaaa') == 3
    assert character_error_rate('abc', '') == 1


def test_unicode_and_literal_policy():
    assert character_error_rate('가', '가') == 0
    assert character_error_rate('가', '가', policy=TextPolicy(False)) == 2
    assert character_error_rate('가', 'ㄱㅏ') == 2  # NFC is not NFKC or a jamo composer.
    assert edit_counts(' 1Ａa\t', '1Aa').errors == 3
    assert character_error_rate('1', '１') == 1
    assert character_error_rate('😀', '') == 1


def distance_oracle(a, b):
    # Full scalar matrix, independent of the count-carrying rolling DP.
    grid = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        grid[i][0] = i
    for j in range(len(b) + 1):
        grid[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            grid[i][j] = min(grid[i-1][j]+1, grid[i][j-1]+1,
                             grid[i-1][j-1]+(a[i-1] != b[j-1]))
    return grid[-1][-1]


def test_exhaustive_against_independent_distance():
    strings = [''.join(p) for n in range(5) for p in itertools.product('ab', repeat=n)]
    for a in strings:
        for b in strings:
            counts = edit_counts(a, b)
            assert counts.errors == distance_oracle(a, b)
            matches = len(a) - counts.substitutions - counts.deletions
            assert len(b) == matches + counts.substitutions + counts.insertions


def test_long_sequence_and_counts_addition():
    assert edit_counts('a' * 1500, 'a' * 1499 + 'b') == ErrorCounts(1, 0, 0, 1500)
    assert ErrorCounts(1, 0, 0, 2) + ErrorCounts(0, 1, 2, 3) == ErrorCounts(1, 1, 2, 5)


@pytest.mark.parametrize("value", [None, 1, [], '\ud800'])
def test_invalid_text(value):
    with pytest.raises(MetricError):
        edit_counts(value, '')
    with pytest.raises(MetricError):
        edit_counts('', value)


def test_invalid_policy_and_counts():
    for counts in [(-1, 0, 0, 1), (True, 0, 0, 1), (2, 0, 0, 1), (0, 1, 0, 0)]:
        with pytest.raises(MetricError):
            ErrorCounts(*counts)
    with pytest.raises(MetricError):
        TextPolicy('NFC')
    with pytest.raises(MetricError):
        edit_counts('', '', policy=None)
