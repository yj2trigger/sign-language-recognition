"""Synthetic data only; no local AI-Hub paths or source labels in fixtures."""
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import struct
import unittest
import warnings
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fingerspelling import crowd_labels as cl
import label_preparation as lp


def member(fs="0001", crowd="01"):
    return f"morpheme/{crowd}/NIA_SL_FS{fs}_CROWD{crowd}_F_morpheme.json"


def raw(text="가상누리길", fs="0001", crowd="01", change=None):
    video = f"NIA_SL_FS{fs}_CROWD{crowd}_F.mp4"
    doc = {"metaData": {"name": video, "url": "https://example.invalid/" + video},
           "data": [{"start": 0, "end": 2, "attributes": [{"name": text}]}]}
    if change:
        change(doc)
    return json.dumps(doc, ensure_ascii=False).encode("utf-8")


def archive(entries):
    stream = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
            for name, content in entries:
                z.writestr(name, content)
    stream.seek(0)
    return stream


def report(entries=None):
    return cl.read_crowd_zip(archive(entries if entries is not None else [(member(), raw())]))


class CrowdTests(unittest.TestCase):
    def test_normal(self):
        result = report()
        record = result.records[0]
        self.assertEqual(record.fs_id, "0001")
        self.assertEqual(record.crowd_id, "01")
        self.assertEqual(record.target.video_id, cl.expected_video_name(member()))
        self.assertEqual(record.target.text, "가상누리길")
        self.assertEqual(record.target.signer_id, "CROWD01")
        self.assertEqual(result.summary()["mapped_count"], 1)
        self.assertEqual(result.summary()["error_count"], 0)
        self.assertEqual(lp.decode_morphemes(raw()), record.target.segments)

    def test_bad_filename_and_path(self):
        for path in ("../" + member(), "/" + member(), member().replace("/", "\\"),
                     member().replace("FS0001", "FS001"), member().replace("_F_", "_R_"),
                     member().replace("morpheme/01/", "morpheme/02/"),
                     member().replace("0001", "０００１"), member() + "\n", None):
            with self.subTest(path_type=type(path)), self.assertRaises(cl.CrowdLabelError):
                cl.expected_video_name(path)
        result = report([("../" + member(), raw())])
        self.assertEqual(result.summary()["filename_mismatch_count"], 1)

    def test_bad_metadata(self):
        for metadata in (None, [], "x", {}):
            with self.subTest(metadata=metadata), self.assertRaises(cl.CrowdLabelError):
                cl.parse_crowd_label(member(), raw(change=lambda d: d.update(metaData=metadata)))

    def test_metadata_name_mismatch(self):
        result = report([(member(), raw(change=lambda d: d["metaData"].update(name="wrong.mp4")))])
        self.assertEqual(dict(result.errors), {"metadata_name": 1})
        self.assertEqual(result.summary()["filename_mismatch_count"], 1)

    def test_url_name_mismatch(self):
        result = report([(member(), raw(change=lambda d: d["metaData"].update(url="https://example.invalid/wrong.mp4")))])
        self.assertEqual(dict(result.errors), {"url_name": 1})

    def test_invalid_url(self):
        for url in (None, [], "", "file:///x", "https:///x", "https://[invalid/x", "https://host/\nx"):
            with self.subTest(url_type=type(url)), self.assertRaises(cl.CrowdLabelError):
                cl.parse_crowd_label(member(), raw(change=lambda d: d["metaData"].update(url=url)))

    def test_url_query_not_part_of_basename(self):
        video = cl.expected_video_name(member())
        record = cl.parse_crowd_label(member(), raw(change=lambda d: d["metaData"].update(
            url="https://example.invalid/" + video + "?private=synthetic#fragment")))
        self.assertEqual(record.target.video_id, video)

    def test_conflicts_preserve_exact_strings(self):
        for a, b in (("합성길", "다른길"), ("가", "가"), (" 합성길", "합성길")):
            result = report([(member(), raw(a)), (member(crowd="02"), raw(b, crowd="02"))])
            self.assertEqual(result.review_fs_ids, ("0001",))
            self.assertEqual([r.target.text for r in result.records], [a, b])
            self.assertEqual(result.summary()["mapped_count"], 2)

    def test_same_text_across_videos_allowed(self):
        result = report([(member(), raw()), (member(crowd="02"), raw(crowd="02"))])
        self.assertEqual(result.review_fs_ids, ())
        self.assertEqual(len(result.records), 2)

    def test_duplicate_json_all_excluded(self):
        result = report([(member(), raw()), (member(), raw("다른합성길"))])
        self.assertEqual(result.json_count, 2)
        self.assertEqual(result.records, ())
        self.assertEqual(dict(result.errors), {"duplicate_json": 2})

    def test_blank_and_invalid_json(self):
        for content in (raw(""), raw("  "), raw("가\x00길"), b"\xff", b"{", b'{"data":[],"data":[]}'):
            with self.subTest(content_length=len(content)), self.assertRaises(cl.CrowdLabelError):
                cl.parse_crowd_label(member(), content)

    def test_multiple_segments_rejected(self):
        content = raw(change=lambda d: d["data"].append(
            {"start": 2, "end": 3, "attributes": [{"name": "합성"}]}))
        with self.assertRaises(cl.CrowdLabelError):
            cl.parse_crowd_label(member(), content)

    def test_numbers_symbols_and_aggregate_statistics(self):
        result = report([(member(), raw("가상12-3길")), (member(fs="0002"), raw("123", fs="0002"))])
        self.assertEqual(result.records[0].target.text, "가상12-3길")
        self.assertEqual(result.summary(), {"json_count": 2, "mapped_count": 2, "error_count": 0,
            "filename_mismatch_count": 0, "fs_count": 2, "crowd_count": 1,
            "conflict_group_count": 0, "hangul_count": 1, "digit_count": 2,
            "road_suffix_candidate_count": 1})

    def test_video_inventory_not_provided_vs_empty(self):
        result = report()
        self.assertEqual(cl.compare_video_names(result).status, "not_provided")
        empty = cl.compare_video_names(result, [])
        self.assertEqual(empty.status, "mismatch")
        self.assertEqual(empty.missing_count, 1)

    def test_video_inventory_match_missing_duplicate_and_extra(self):
        result = report()
        name = result.records[0].target.video_id
        self.assertEqual(cl.compare_video_names(result, [name]).status, "matched")
        checked = cl.compare_video_names(result, [name, name, "NIA_SL_FS9999_CROWD99_F.mp4"])
        self.assertEqual((checked.duplicate_count, checked.unexpected_count), (1, 1))
        self.assertEqual(checked.status, "mismatch")
        self.assertEqual(checked.matched_count, 1)

    def test_video_inventory_validation_and_limit(self):
        result = report()
        for names in ("name.mp4", [None], ["../name.mp4"], 1):
            with self.assertRaises(cl.CrowdLabelError):
                cl.compare_video_names(result, names)
        name = result.records[0].target.video_id
        with self.assertRaises(cl.CrowdLabelError):
            cl.compare_video_names(result, [name, name], max_names=1)

    def test_bad_zip_and_missing_path(self):
        for source in (io.BytesIO(b"invalid zip"), Path("synthetic-nonexistent-archive.zip")):
            with self.assertRaisesRegex(cl.CrowdLabelError, "^archive_read$"):
                cl.read_crowd_zip(source)

    def test_limits(self):
        for kwargs in ({"max_json_bytes": 10}, {"max_total_bytes": 10}):
            if "max_json_bytes" in kwargs:
                result = cl.read_crowd_zip(archive([(member(), raw())]), **kwargs)
                self.assertEqual(dict(result.errors), {"json_size_limit": 1})
            else:
                with self.assertRaises(cl.CrowdLabelError):
                    cl.read_crowd_zip(archive([(member(), raw())]), **kwargs)
        with self.assertRaises(cl.CrowdLabelError):
            cl.read_crowd_zip(archive([(member(), raw()), (member(crowd="02"), raw(crowd="02"))]), max_members=1)
        with self.assertRaises(cl.CrowdLabelError):
            cl.read_crowd_zip(archive([]), max_members=True)

    def test_empty_zip_and_ignored_non_json(self):
        result = report([("morpheme/", b""), ("readme.txt", b"synthetic")])
        self.assertEqual(result.summary()["json_count"], 0)

    def test_private_repr_and_error_codes(self):
        secret = "합성비공개길"
        result = report([(member(), raw(secret))])
        self.assertNotIn(secret, repr(result))
        self.assertNotIn("0001", repr(result))
        self.assertNotIn(secret, repr(result.records[0]))
        self.assertTrue(all(type(v) is int for v in result.summary().values()))
        with self.assertRaises(cl.CrowdLabelError) as ctx:
            cl.parse_crowd_label(secret, raw(secret))
        self.assertNotIn(secret, str(ctx.exception))
        self.assertNotIn(secret, str(asdict(cl.compare_video_names(result))))

    def test_review_regression_corrupt_deflate_counted_without_leaking(self):
        content = bytearray(archive([(member(), raw())]).getvalue())
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            entry = z.infolist()[0]
            offset = entry.header_offset
            name_len, extra_len = struct.unpack_from("<HH", content, offset + 26)
            data_offset = offset + 30 + name_len + extra_len
            content[data_offset:data_offset + entry.compress_size] = b"\xff" * entry.compress_size
        result = cl.read_crowd_zip(io.BytesIO(content))
        self.assertEqual(dict(result.errors), {"member_read": 1})
        self.assertEqual(result.summary()["mapped_count"], 0)

    def test_inventory_cannot_hide_rejected_labels(self):
        result = report([(member(), raw()), (member(fs="0002"), b"invalid")])
        checked = cl.compare_video_names(result, [result.records[0].target.video_id])
        self.assertEqual(checked.status, "mismatch")
        self.assertEqual(result.summary()["error_count"], 1)


if __name__ == "__main__":
    unittest.main()
