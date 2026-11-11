"""Synthetic contract fixtures, NOT a replacement Hangul decomposition engine."""
from pathlib import Path
import sys
import unicodedata
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fingerspelling.tokens import TokenVocabulary, TokenError
from fingerspelling.ctc_targets import TargetEncoder, CTCError, audit_crowd

FIXTURES = {"가": ["ㄱ", "ㅏ"], "각": ["ㄱ", "ㅏ", "ㄱ"],
            "값": ["ㄱ", "ㅏ", "ㅄ"], "꽤": ["ㄲ", "ㅙ"], "ㄱ": ["ㄱ"]}


def fixture_split(char):
    return FIXTURES[char]


def encoder(**kwargs):
    symbols = ("ㄱ", "ㅏ", "ㅄ", "ㄲ", "ㅙ", "space", "digit")
    vocab = TokenVocabulary({t: i + 1 for i, t in enumerate(symbols)}, blank_id=0)
    return TargetEncoder(vocab, decompose=fixture_split, **kwargs)


@pytest.mark.parametrize("text,expected", [("가", (1, 2)), ("각", (1, 2, 1)),
    ("값꽤", (1, 2, 3, 4, 5)), ("ㄱㄱ", (1, 1))])
def test_hangul_contract(text, expected):
    enc = encoder()
    target = enc.encode(text)
    assert target.ids == expected
    assert target == enc.encode(text)
    assert target.original_text == text
    assert 0 not in target.ids


def test_normalization_preserves_original():
    nfd = unicodedata.normalize("NFD", "각")
    assert encoder().encode(nfd).ids == encoder().encode("각").ids
    assert encoder().encode(nfd).original_text == nfd


@pytest.mark.parametrize("text,code", [("", "empty_text"), ("가1", "digit_policy"),
    ("가１", "digit_policy"), ("가 ", "whitespace_policy"), ("가A", "unsupported"),
    ("\u1100", "unsupported"), ("\ud800", "unsupported"), ("\u200b", "unsupported")])
def test_rejections(text, code):
    with pytest.raises(CTCError, match=code):
        encoder().encode(text)


def test_explicit_literals():
    result = encoder(literal_mapping={" ": ("space",), "1": ("digit",)}).encode("가 1")
    assert result.ids == (1, 2, 6, 7)


@pytest.mark.parametrize("mapping", [{" ": ()}, {"가": ("ㄱ",)}, {"\n": ("space",)},
                                     {"\u212b": ("space",)}])
def test_no_deletion_or_hangul_override(mapping):
    with pytest.raises(CTCError):
        encoder(literal_mapping=mapping)


def test_ctc_repeat_length():
    with pytest.raises(CTCError, match="insufficient"):
        encoder().encode("ㄱㄱ", input_length=2)
    assert encoder().encode("ㄱㄱ", input_length=3).minimum_input_length == 3
    assert encoder().encode("가", input_length=2).minimum_input_length == 2
    for length in (True, -1, 2.0):
        with pytest.raises(CTCError, match="invalid_input_length"):
            encoder().encode("가", input_length=length)


def test_unregistered_and_sanitized():
    enc = TargetEncoder(TokenVocabulary({"ㄱ": 1}, blank_id=0), decompose=fixture_split)
    with pytest.raises(TokenError, match="^unregistered_token$"):
        enc.encode("가")
    def reject(char):
        raise ValueError("PRIVATE SOURCE")
    enc = TargetEncoder(enc.vocabulary, decompose=reject)
    with pytest.raises(CTCError, match="^decomposition_rejected$") as error:
        enc.encode("가")
    assert error.value.__suppress_context__


def test_missing_dependency(monkeypatch):
    import fingerspelling.ctc_targets as module
    def absent(*args):
        raise ModuleNotFoundError(name="fingerspelling.jamo")
    monkeypatch.setattr(module, "import_module", absent)
    with pytest.raises(CTCError, match="missing_pr11"):
        TargetEncoder(encoder().vocabulary).encode("가")


def test_aggregate_counts_and_privacy():
    report = SimpleNamespace(json_count=5, errors=(("morpheme", 1),), records=tuple(
        SimpleNamespace(target=SimpleNamespace(text=t)) for t in ("가", "1 A", "", "ㄱㄱ")))
    result = audit_crowd(report, encoder())
    assert result["converted"] == 2
    assert result["unconvertible"] == 2
    assert result["empty_strings"] == 1
    assert result["digit_rejected"] == 1
    assert result["unsupported_character_occurrences"] == {"Lu": 1, "decimal": 1, "whitespace": 1}
    assert "ㄱ" not in repr(result)


def test_pr11_integration_when_available():
    jamo = pytest.importorskip("fingerspelling.jamo")
    enc = TargetEncoder(encoder().vocabulary)
    assert enc.encode("값꽤").ids == (1, 2, 3, 4, 5)
    symbols = sorted(jamo.JAMO)
    full = TargetEncoder(TokenVocabulary({t: i + 1 for i, t in enumerate(symbols)}, blank_id=0))
    for code in range(0xac00, 0xd7a4):
        text = chr(code)
        target = full.encode(text)
        assert target.tokens == tuple(jamo.decompose_hangul(text))
        assert full.encode(unicodedata.normalize("NFD", text)).ids == target.ids
    for text in ("\u3164", "\u318e"):
        with pytest.raises(CTCError, match="decomposition_rejected"):
            full.encode(text)


def test_rejected_hangul_in_audit():
    def reject(char):
        raise ValueError("private")
    report = SimpleNamespace(json_count=1, errors=(), records=(
        SimpleNamespace(target=SimpleNamespace(text="\u3164")),))
    enc = TargetEncoder(encoder().vocabulary, decompose=reject)
    assert audit_crowd(report, enc)["unsupported_character_occurrences"] == {"Lo": 1}
