"""Synthetic fixtures only. Run: python -m unittest discover -s tests -p test_label_preparation.py"""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import label_preparation as lp


def fixture(text="가상누리12길"):
    return {"data": [{"start": 0.5, "end": 2.0, "attributes": [{"name": text}]}]}


def target(video="v1", signer="s1", text="가상누리길"):
    return lp.make_target(video_id=video, signer_id=signer, text=text)


class LabelTests(unittest.TestCase):
    def test_normal(self):
        doc = fixture()
        before = copy.deepcopy(doc)
        result = lp.parse_morphemes(doc)
        self.assertEqual(result, (lp.Segment(0.5, 2.0, "가상누리12길"),))
        self.assertEqual(doc, before)
        self.assertEqual(lp.target_from_morphemes(doc, video_id="v", signer_id="s").segments, result)

    def test_strings_bom_and_metadata(self):
        doc = fixture()
        doc["metaData"] = {"ignored": True}
        doc["data"][0].update(start="0.5", end="2.000")
        raw = json.dumps(doc, ensure_ascii=False).encode("utf-8")
        self.assertEqual(lp.decode_morphemes(raw), lp.decode_morphemes(b"\xef\xbb\xbf" + raw))

    def test_empty_missing_and_types(self):
        for doc in (None, [], "", {}, {"data": []}, {"data": {}}, {"data": [None]}, {"data": [{}]}):
            with self.subTest(doc=doc), self.assertRaises(lp.LabelError):
                lp.parse_morphemes(doc)
        for key in ("start", "end", "attributes"):
            doc = fixture()
            del doc["data"][0][key]
            with self.subTest(key=key), self.assertRaises(lp.LabelError):
                lp.parse_morphemes(doc)
        for attrs in (None, {}, [], [None], [{}], [{"name": 12}], [{"name": "a"}, {"name": "b"}]):
            doc = fixture()
            doc["data"][0]["attributes"] = attrs
            with self.subTest(attrs=attrs), self.assertRaises(lp.LabelError):
                lp.parse_morphemes(doc)

    def test_invalid_times(self):
        for value in (None, True, [], {}, "bad", -1, "NaN", float("inf"), 10**1000):
            doc = fixture()
            doc["data"][0]["start"] = value
            with self.subTest(value_type=type(value)), self.assertRaises(lp.LabelError):
                lp.parse_morphemes(doc)
        for start, end in ((2, 2), (3, 2), (0, -1)):
            doc = fixture()
            doc["data"][0].update(start=start, end=end)
            with self.assertRaises(lp.LabelError):
                lp.parse_morphemes(doc)

    def test_multiple_segments_never_silently_truncated(self):
        doc = fixture()
        second = copy.deepcopy(doc["data"][0])
        second.update(start=2, end=3)
        doc["data"].append(second)
        self.assertEqual(len(lp.parse_morphemes(doc)), 2)
        with self.assertRaises(lp.LabelError):
            lp.target_from_morphemes(doc, video_id="v", signer_id="s")
        for start in (0.1, 1.99):
            second["start"] = start
            with self.assertRaises(lp.LabelError):
                lp.parse_morphemes(doc)

    def test_encoding_and_json(self):
        for raw in (b"", b"{", b"\xff", fixture()["data"][0]["attributes"][0]["name"].encode("cp949"),
                    b' {"data": [], "data": []}', b'{"data":[{"start":0,"start":1}]}',
                    b'"\\ud800"', "{}", None):
            with self.subTest(raw_type=type(raw)), self.assertRaises(lp.LabelError):
                lp.decode_morphemes(raw)

    def test_preserve_numbers_punctuation_spacing_and_unicode(self):
        for text in ("가상로12-3길", " 합성 길 (2) ", "ㄱㅏ", "가", "123", "가" * 10000):
            self.assertEqual(lp.parse_morphemes(fixture(text))[0].text, text)
            self.assertEqual(target(text=text).text, text)

    def test_invalid_text_and_identity(self):
        for text in (None, 1, "", "  ", "가\n길", "가\x00길", "\ud800", "\ufffd", "가\u200b길"):
            with self.subTest(text_type=type(text)), self.assertRaises(lp.LabelError):
                lp.parse_morphemes(fixture(text))
        for video, signer in (("", "s"), ("v", ""), (None, "s"), ("v", 1)):
            with self.assertRaises(lp.LabelError):
                target(video=video, signer=signer)

    def test_duplicates(self):
        self.assertEqual(lp.validate_targets([]), ())
        self.assertEqual(len(lp.validate_targets([target(), target(video="v2")])), 2)
        for rows in ([target(), target()], [target(), target(text="다른합성길")], [{}]):
            with self.assertRaises(lp.LabelError):
                lp.validate_targets(rows)
        for rows in (None, 42):
            with self.assertRaises(lp.LabelError):
                lp.validate_targets(rows)

    def test_signer_split_leakage(self):
        lp.validate_signer_splits({"train": [target()], "val": [target("v2", "s2")], "test": []})
        for row in (target("v2", "s1"), target("v1", "s2")):
            with self.assertRaises(lp.LabelError):
                lp.validate_signer_splits({"train": [target()], "val": [row]})
        with self.assertRaises(lp.LabelError):
            lp.validate_signer_splits([])

    def test_privacy_of_errors_and_repr(self):
        secret = "합성비공개길"
        doc = fixture(secret)
        doc["data"][0]["end"] = secret
        with self.assertRaises(lp.LabelError) as caught:
            lp.parse_morphemes(doc)
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn(secret, repr(target(text=secret)))
        self.assertNotIn(secret, repr(lp.Segment(0, 1, secret)))

    def test_review_regression_direct_records_cannot_bypass_validation(self):
        for segments in ([lp.Segment(0, 1, "가")], (None,),
                         (lp.Segment(0, 1, "다"),), (lp.Segment(2, 1, "가"),),
                         (lp.Segment(0, 1, "가"), lp.Segment(1, 2, "가"))):
            with self.subTest(segments_type=type(segments)), self.assertRaises(lp.LabelError):
                lp.validate_targets([lp.Target("v", "s", "가", segments)])

    def test_review_regression_json_integer_limit(self):
        with self.assertRaises(lp.LabelError):
            lp.decode_morphemes(b'{"unused":' + b"9" * 5000 + b"}")


if __name__ == "__main__":
    unittest.main()
