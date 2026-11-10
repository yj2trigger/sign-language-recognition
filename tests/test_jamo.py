"""표준 라이브러리만 사용: python tests/test_jamo.py"""
import sys
import unicodedata
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fingerspelling.jamo import JAMO, decompose_hangul, compose_hangul


class JamoTests(unittest.TestCase):
    def test_examples(self):
        for text, expected in [("서울", "ㅅㅓㅇㅜㄹ"), ("한국", "ㅎㅏㄴㄱㅜㄱ"),
                               ("학교", "ㅎㅏㄱㄱㅛ"), ("강남", "ㄱㅏㅇㄴㅏㅁ"),
                               ("서울역", "ㅅㅓㅇㅜㄹㅇㅕㄱ"), ("값", "ㄱㅏㅄ"),
                               ("꿩", "ㄲㅝㅇ"), ("닭", "ㄷㅏㄺ")]:
            with self.subTest(text=text):
                tokens = decompose_hangul(text)
                self.assertEqual(tokens, list(expected))
                self.assertEqual(compose_hangul(tokens), text)

    def test_all_modern_syllables_against_unicode(self):
        # 독립적인 Unicode NFD를 기준으로 11,172개 전체를 검증한다.
        for code in range(0xAC00, 0xD7A4):
            text = chr(code)
            nfd = unicodedata.normalize("NFD", text)
            # 제품 상수/인덱스를 재사용하면 상수 오타가 왕복 검사에서 상쇄된다.
            # Unicode 이름으로 독립적으로 호환 자모를 찾는다.
            expected = [unicodedata.lookup(
                "HANGUL LETTER " + unicodedata.name(char).split(" ", 2)[2]
            ) for char in nfd]
            self.assertEqual(decompose_hangul(text), expected)
            self.assertEqual(decompose_hangul(nfd), expected)
            self.assertEqual(compose_hangul(expected), text)
            self.assertEqual(compose_hangul(expected, preserve_boundaries=True), text)

    def test_all_syllables_in_sequence(self):
        text = "".join(chr(code) for code in range(0xAC00, 0xD7A4))
        for sample in (text, text[::-1]):
            for boundaries in (False, True):
                tokens = decompose_hangul(sample, preserve_boundaries=boundaries)
                self.assertEqual(compose_hangul(tokens, preserve_boundaries=boundaries), sample)

    def test_boundary_groups_are_literal(self):
        self.assertEqual(compose_hangul(list("ㄱ|ㅏ"), preserve_boundaries=True), "ㄱㅏ")
        self.assertEqual(compose_hangul(list("ㄱㅏ"), preserve_boundaries=True), "가")
        for char in JAMO:
            self.assertEqual(compose_hangul([char], preserve_boundaries=True), char)
        for tokens in (["ᄀ"], [""], ["가"], ["ㅏ", "ㄱ"], ["ㄱ", "ㅏ", "ㄸ"]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                compose_hangul(tokens, preserve_boundaries=True)

    def test_normalization_contract(self):
        self.assertEqual(decompose_hangul("ㄱㅏ"), list("ㄱㅏ"))
        self.assertEqual(decompose_hangul("가"), list("ㄱㅏ"))
        self.assertEqual(decompose_hangul("ㄱㅏ", preserve_boundaries=True), list("ㄱ|ㅏ"))
        self.assertEqual(decompose_hangul("가", preserve_boundaries=True), list("ㄱㅏ"))
        for text in ("ㄱᅡ", "가ᅡ", "ᄀㅏ", "가\u0301", "\ud800", "\u3130", "\ud7a4"):
            for boundaries in (False, True):
                with self.subTest(text=repr(text)), self.assertRaises(ValueError):
                    decompose_hangul(text, preserve_boundaries=boundaries)

    def test_empty_and_long(self):
        for text in ("", "가", "힣", "서울역학교값꿩" * 10000):
            self.assertEqual(compose_hangul(decompose_hangul(text)), text)
            tokens = decompose_hangul(text, preserve_boundaries=True)
            self.assertEqual(compose_hangul(tokens, preserve_boundaries=True), text)

    def test_boundary_roundtrip(self):
        for text in ("간ㅏ", "가나", "ㄱㅏ", "ㄳ", "".join(sorted(JAMO)),
                     "각가ㄱㅘ닭ㅏ", unicodedata.normalize("NFD", "서울")):
            tokens = decompose_hangul(text, preserve_boundaries=True)
            self.assertEqual(compose_hangul(tokens, preserve_boundaries=True),
                             unicodedata.normalize("NFC", text))

    def test_lost_information_is_explicit(self):
        self.assertEqual(decompose_hangul("간ㅏ"), decompose_hangul("가나"))
        self.assertEqual(compose_hangul(list("ㄱㅏㄴㅏ")), "가나")
        self.assertEqual(compose_hangul(list("ㄱㅏㄴ|ㅏ"), preserve_boundaries=True), "간ㅏ")
        self.assertNotEqual(decompose_hangul("간ㅏ", preserve_boundaries=True),
                            decompose_hangul("가나", preserve_boundaries=True))

    def test_invalid_text(self):
        for text in ("서울1", "a", "!", " ", "\n", "😀", "漢", "ᄀ", "ᆨ", "ㅤ", "ㆍ"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                decompose_hangul(text)
        for text in (None, [], 1):
            with self.assertRaises(TypeError):
                decompose_hangul(text)

    def test_invalid_tokens(self):
        for tokens in ([""], ["서울"], ["ᄀ"], ["1"], [" "], ["|"],
                       list("ㄱ"), list("ㅏ"), list("ㄱㄱㅏ"), list("ㄱㅏㅏ"),
                       list("ㅄㅏ"), list("ㄱㅏㄸ")):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                compose_hangul(tokens)
        for tokens in (None, "ㄱㅏ", ("ㄱ", "ㅏ"), [None], [1]):
            with self.assertRaises(TypeError):
                compose_hangul(tokens)
        for tokens in (list("|ㄱ"), list("ㄱ|"), list("ㄱ||ㅏ"), list("ㄱㅏㄴㅏ")):
            with self.assertRaises(ValueError):
                compose_hangul(tokens, preserve_boundaries=True)

    def test_determinism_and_no_mutation(self):
        tokens = list("ㅎㅏㄴㄱㅜㄱ")
        copy = tokens[:]
        self.assertEqual(compose_hangul(tokens), compose_hangul(tokens))
        self.assertEqual(tokens, copy)
        for function, value in ((decompose_hangul, "가"), (compose_hangul, tokens)):
            with self.assertRaises(TypeError):
                function(value, preserve_boundaries=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
