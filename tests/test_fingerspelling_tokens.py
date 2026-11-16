from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fingerspelling.tokens import TokenVocabulary, TokenError


@pytest.mark.parametrize("mapping,blank", [({}, 0), ({"": 1}, 0),
    ({"a": -1}, 0), ({"a": True}, 0), ({"a": 1, "b": 1}, 0),
    ({"a": 0}, 0), ({"a": 1}, True), ({"a": 1}, -1)])
def test_invalid_vocabulary(mapping, blank):
    with pytest.raises(TokenError):
        TokenVocabulary(mapping, blank_id=blank)


def test_explicit_stable_ids():
    source = {"a": 4, "b": 2}
    vocab = TokenVocabulary(source, blank_id=0)
    source["a"] = 99
    assert vocab.encode(["a", "b", "a"]) == (4, 2, 4)
    assert vocab.num_classes == 5
    with pytest.raises(TypeError):
        vocab.token_ids["a"] = 8
    with pytest.raises(TokenError, match="unregistered_token"):
        vocab.encode(["private-token"])
