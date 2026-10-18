import json

from knowledge_brain.ingestion import ArticleRecord
from knowledge_brain.sync import run_incremental


class FakeCrawler:
    request_delay = 0

    def fetch_sitemap(self):
        return ["https://example.test/support/solutions/articles/one", "https://example.test/support/solutions/articles/two"]

    def ingest_single_article(self, url):
        return ArticleRecord(url=url, title=url.rsplit("/", 1)[-1], text="document", retrieval_timestamp="now", content_hash=url, metadata={})


def test_limited_sync_does_not_deactivate_unvisited_records(tmp_path):
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "articles.json"
    manifest.write_text(json.dumps({"https://example.test/support/solutions/articles/two": {"active": True, "content_hash": "old"}}))
    stats = run_incremental(FakeCrawler(), manifest, output, limit=1)
    saved = json.loads(manifest.read_text())
    assert stats["removed"] == 0
    assert saved["https://example.test/support/solutions/articles/two"]["active"] is True


def test_full_sync_removes_absent_records(tmp_path):
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "articles.json"
    manifest.write_text(json.dumps({"https://example.test/support/solutions/articles/gone": {"active": True}}))
    stats = run_incremental(FakeCrawler(), manifest, output)
    assert stats["removed"] == 1
    assert len(json.loads(output.read_text())) == 2
