"""Retrieve bounded public AI-Hub dataset 103 documents and provenance.

No authentication, cookies or credentials are used. HTML responses are retained
as HTML evidence and reported as unsuccessful document retrievals.
"""
import argparse
from email.message import Message
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import urllib.parse
import urllib.request
import urllib.error
import zipfile

PAGE = "https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103"
LIMIT = 32 * 1024 * 1024


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and "href" in attrs:
            self.links.append(urllib.parse.urljoin(PAGE, attrs["href"]))


def retrieve(url):
    request = urllib.request.Request(url, headers={"User-Agent": "CROWD-source-validation/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        final = urllib.parse.urlsplit(response.url)
        if final.scheme != "https" or final.hostname not in ("aihub.or.kr", "www.aihub.or.kr"):
            raise ValueError("unexpected_redirect")
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > LIMIT:
            raise ValueError("document_size")
        payload = response.read(LIMIT + 1)
        if len(payload) > LIMIT:
            raise ValueError("document_size")
        return payload, dict(status=response.status, content_type=response.headers.get("Content-Type"),
                             content_disposition=response.headers.get("Content-Disposition"),
                             declared_bytes=declared, final_url=response.url)


def save_document(folder, identifier, url):
    payload, metadata = retrieve(url)
    extension = ""
    if payload.startswith(b"%PDF-"):
        extension = ".pdf"
    elif zipfile.is_zipfile(io.BytesIO(payload)):
        extension = ".zip"
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if sum(member.file_size for member in archive.infolist()) > LIMIT:
                raise ValueError("expanded_document_size")
            if archive.testzip() is not None:
                raise ValueError("document_crc")
    elif (b"<html" in payload[:4096].lower()
          or b"<!doctype html" in payload[:4096].lower()
          or ("text/html" in (metadata["content_type"] or "").lower()
              and payload.lstrip().lower().startswith(b"<script"))):
        extension = ".html"
    elif payload.lstrip().startswith((b"{", b"[")):
        json.loads(payload)
        extension = ".json"
    header = Message()
    header["Content-Disposition"] = metadata["content_disposition"] or ""
    supplied = header.get_filename()
    if supplied:
        supplied = urllib.parse.unquote(supplied)
        # Preserve the server filename only when it is a single safe component.
        if supplied != Path(supplied).name or any(c in supplied for c in '<>:"/\\|?*'):
            supplied = None
    filename = (Path(supplied).stem if supplied else f"aihub-103-{identifier}") + extension
    target = folder / filename
    digest = hashlib.sha256(payload).hexdigest()
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("existing_document_differs")
    else:
        with target.open("xb") as stream:
            stream.write(payload)
    return dict(identifier=identifier, url=url, filename=filename, bytes=len(payload), sha256=digest,
                detected_format=extension.lstrip(".") or "unknown",
                document_download_success=extension == ".zip",
                document_open_verified=extension == ".zip",
                requires_pdf_open_validation=extension == ".pdf", **metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    folder = args.destination
    folder.mkdir(parents=True, exist_ok=True)
    # Completed provenance makes reruns read-only and avoids duplicate downloads.
    manifest = folder / "download-provenance.json"
    records = []
    completed = set()
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        for record in previous["documents"]:
            if "filename" in record:
                target = folder / record["filename"]
                if hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]:
                    raise ValueError("existing_document_hash")
            identifier = record.get("identifier")
            if identifier is None:
                query = urllib.parse.parse_qs(urllib.parse.urlsplit(record.get("url", "")).query)
                identifier = query.get("fileSn", [None])[0]
            completed.add(identifier)
        if {"7969", "18941"} <= completed:
            print(json.dumps(previous, ensure_ascii=True))
            return
        records = previous["documents"]
    page, page_metadata = retrieve(PAGE)
    links = Links()
    links.feed(page.decode("utf-8"))
    result = dict(dataset_url=PAGE, page_response=page_metadata, documents=records)
    def checkpoint():
        # Persist each response before the next request so a later network
        # failure does not lose provenance for an already downloaded file.
        pending = folder / "download-provenance.pending"
        with pending.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
        pending.replace(manifest)
    for identifier, parameter in (("7969", "dataMnlFileSn"), ("18941", "cnstcPrcuseFileSn")):
        if identifier in completed:
            continue
        matches = []
        for url in links.links:
            parsed = urllib.parse.urlsplit(url)
            query = urllib.parse.parse_qs(parsed.query)
            if (parsed.hostname in ("aihub.or.kr", "www.aihub.or.kr")
                    and parsed.path == "/file/down.do" and query.get("dataSetSn") == ["103"]
                    and query.get(parameter) == [identifier] and query.get("fileSn") == [identifier]):
                matches.append(url)
        if not matches:
            records.append(dict(identifier=identifier, document_download_success=False,
                                error="download_link_not_verified"))
            checkpoint()
            continue
        try:
            records.append(save_document(folder, identifier, matches[0]))
        except (OSError, ValueError, urllib.error.URLError):
            records.append(dict(identifier=identifier, document_download_success=False,
                                error="document_retrieval_failed"))
        checkpoint()
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
