"""Resumable, incremental Knowledge Brain synchronization job.

Run once per day with Task Scheduler, or use ``--loop-hours 24`` for a small
local demo. The manifest is checkpointed after every page so interrupted work
resumes without duplicating records.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List

from knowledge_brain.ingestion import BrilliantDirectoriesCrawler, extract_article_urls

DEFAULT_MANIFEST = Path("knowledge_brain/data/full_articles_manifest.json")
DEFAULT_OUTPUT = Path(os.getenv("KNOWLEDGE_BRAIN_DATA_PATH", "knowledge_brain/data/full_articles.json"))


def _read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def run_incremental(crawler: BrilliantDirectoriesCrawler, manifest_path: Path = DEFAULT_MANIFEST,
                    output_path: Path = DEFAULT_OUTPUT, limit: int | None = None) -> Dict[str, int]:
    """Refresh current sitemap URLs and deactivate removed records."""
    manifest: Dict[str, Dict[str, Any]] = _read_json(manifest_path, {})
    old_records = _read_json(output_path, [])
    record_by_url = {item.get("url"): item for item in old_records if isinstance(item, dict) and item.get("url")}
    urls = list(dict.fromkeys(extract_article_urls(crawler.fetch_sitemap())))
    is_limited_run = limit is not None
    if limit is not None:
        urls = urls[:limit]
    current = set(urls)
    stats = {"new": 0, "updated": 0, "unchanged": 0, "removed": 0, "failed": 0}
    if not is_limited_run:
        for url in set(manifest).difference(current):
            if manifest[url].get("active", True):
                manifest[url]["active"] = False
                record_by_url.pop(url, None)
                stats["removed"] += 1
    for index, url in enumerate(urls):
        record = crawler.ingest_single_article(url)
        if record is None:
            manifest.setdefault(url, {})["last_status"] = "failed"
            stats["failed"] += 1
        else:
            previous = manifest.get(url, {})
            if previous.get("content_hash") == record.content_hash:
                stats["unchanged"] += 1
            else:
                stats["new" if url not in manifest else "updated"] += 1
                record_by_url[url] = record.to_dict()
            manifest[url] = {"content_hash": record.content_hash, "active": True, "last_status": "ok",
                             "retrieval_timestamp": record.retrieval_timestamp}
        # Checkpoint manifest and records together after every page. This makes
        # an interruption resumable without declaring a record indexed before it
        # has actually been persisted.
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(list(record_by_url.values()), indent=2, ensure_ascii=False), encoding="utf-8")
        if index < len(urls) - 1 and crawler.request_delay > 0:
            time.sleep(crawler.request_delay)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Run resumable incremental Knowledge Brain sync.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--loop-hours", type=float, default=0)
    args = parser.parse_args()
    while True:
        print(json.dumps(run_incremental(BrilliantDirectoriesCrawler(), limit=args.limit), indent=2))
        if not args.loop_hours:
            return 0
        time.sleep(args.loop_hours * 3600)


if __name__ == "__main__":
    raise SystemExit(main())
