"""Synthetic HTTP responses for document formats and access failures."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import fetch_crowd_docs as fetch


class DownloadTests(unittest.TestCase):
    def test_official_links_are_read_from_page(self):
        parser = fetch.Links()
        parser.feed('<a href="/file/down.do?fileSn=18941&amp;dataSetSn=103">guide</a>')
        self.assertEqual(parser.links, ["https://aihub.or.kr/file/down.do?fileSn=18941&dataSetSn=103"])

    def save(self, payload, content_type, disposition=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        folder = Path(directory.name)
        response = dict(status=200, content_type=content_type,
                        content_disposition=disposition,
                        declared_bytes=str(len(payload)), final_url=fetch.PAGE)
        with patch.object(fetch, "retrieve", return_value=(payload, response)):
            record = fetch.save_document(folder, "18941", fetch.PAGE)
        return folder, record, response

    def test_access_error_script_is_html_failure(self):
        payload = b"<script>alert('access denied');window.close();</script>"
        folder, record, _ = self.save(payload, "text/html;charset=UTF-8")
        self.assertEqual(record["detected_format"], "html")
        self.assertFalse(record["document_download_success"])
        self.assertEqual((folder / record["filename"]).read_bytes(), payload)
        self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())

    def test_html_login_and_unknown_payload(self):
        for payload, content_type, extension in (
                (b"<!DOCTYPE html><html>Login</html>", "text/html", ".html"),
                (b"unknown", "application/octet-stream", "")):
            with self.subTest(extension=extension):
                _, record, _ = self.save(payload, content_type)
                self.assertEqual(Path(record["filename"]).suffix, extension)
                self.assertFalse(record["document_download_success"])

    def test_header_overrides_wrong_filename_extension(self):
        _, record, _ = self.save(b"<html>Error</html>", "text/html",
                                  'attachment; filename="guide.pdf"')
        self.assertEqual(record["filename"], "guide.html")
        self.assertFalse(record["document_download_success"])

    def test_pdf_signature_requires_open_validation(self):
        _, record, _ = self.save(b"%PDF-this-is-truncated", "application/pdf")
        self.assertFalse(record["document_download_success"])
        self.assertFalse(record["document_open_verified"])
        self.assertTrue(record["requires_pdf_open_validation"])

    def test_later_network_failure_keeps_first_provenance(self):
        page = ('<a href="/file/down.do?fileSn=7969&amp;dataMnlFileSn=7969&amp;dataSetSn=103">manual</a>'
                '<a href="/file/down.do?fileSn=18941&amp;cnstcPrcuseFileSn=18941&amp;dataSetSn=103">guide</a>')
        metadata = dict(status=200, content_type="text/html", content_disposition=None,
                        declared_bytes=None, final_url=fetch.PAGE)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, "argv", ["fetch", directory]), patch.object(
                    fetch, "retrieve", side_effect=[(page.encode(), metadata),
                                                      (b"<html>denied</html>", metadata),
                                                      OSError("network failure")]):
                fetch.main()
            result = json.loads((Path(directory) / "download-provenance.json").read_text())
            self.assertEqual(len(result["documents"]), 2)
            self.assertEqual(result["documents"][1]["error"], "document_retrieval_failed")
            with patch.object(sys, "argv", ["fetch", directory]), patch.object(fetch, "retrieve") as network:
                fetch.main()
            network.assert_not_called()

    def test_filename_cannot_escape_destination(self):
        _, record, _ = self.save(b"unknown", "application/octet-stream",
                                  'attachment; filename="../elsewhere.pdf"')
        self.assertEqual(record["filename"], "aihub-103-18941")

    def test_existing_different_file_is_preserved(self):
        folder, record, response = self.save(b"original", "application/octet-stream")
        with patch.object(fetch, "retrieve", return_value=(b"changed", response)):
            with self.assertRaisesRegex(ValueError, "^existing_document_differs$"):
                fetch.save_document(folder, "18941", fetch.PAGE)
        self.assertEqual((folder / record["filename"]).read_bytes(), b"original")

    def test_reuse_manifest_with_unavailable_link(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "download-provenance.json"
            path.write_text(json.dumps({"documents": [{"identifier": "7969",
                                                     "error": "download_link_not_verified"},
                                                    {"identifier": "18941",
                                                     "error": "download_link_not_verified"}]}))
            with patch.object(sys, "argv", ["fetch", directory]), patch.object(fetch, "retrieve") as network:
                fetch.main()
            network.assert_not_called()

    def test_partial_manifest_resumes_only_missing_document(self):
        metadata = dict(status=200, content_type="text/html", content_disposition=None,
                        declared_bytes=None, final_url=fetch.PAGE)
        page = b'<a href="/file/down.do?fileSn=18941&amp;cnstcPrcuseFileSn=18941&amp;dataSetSn=103">guide</a>'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "download-provenance.json"
            path.write_text(json.dumps({"documents": [{"identifier": "7969",
                                                        "error": "download_link_not_verified"}]}))
            with patch.object(sys, "argv", ["fetch", directory]), patch.object(
                    fetch, "retrieve", side_effect=[(page, metadata), (b"<html>denied</html>", metadata)]) as network:
                fetch.main()
            self.assertEqual(network.call_count, 2)
            self.assertEqual(network.call_args_list[1].args[0],
                             "https://aihub.or.kr/file/down.do?fileSn=18941&cnstcPrcuseFileSn=18941&dataSetSn=103")
            self.assertEqual(len(json.loads(path.read_text())["documents"]), 2)


if __name__ == "__main__":
    unittest.main()
