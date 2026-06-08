"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Crawl nội dung HTML từ luatvietnam.vn bằng requests + BeautifulSoup.
(MarkItDown ở task3 convert được HTML → Markdown)
"""

import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

LEGAL_SOURCES = [
    {
        "filename": "luat-phong-chong-ma-tuy-2021.html",
        "url": "https://luatvietnam.vn/hinh-su/luat-phong-chong-ma-tuy-2021-211867-d1.html",
        "description": "Luat Phong, chong ma tuy 2021 (73/2021/QH15)",
    },
    {
        "filename": "nghi-dinh-57-2022.html",
        "url": "https://luatvietnam.vn/hinh-su/nghi-dinh-57-2022-nd-cp-239568-d1.html",
        "description": "Nghi dinh 57/2022/ND-CP danh muc chat ma tuy",
    },
    {
        "filename": "bo-luat-hinh-su-toi-pham-ma-tuy.html",
        "url": "https://luatvietnam.vn/hinh-su/bo-luat-hinh-su-2015-116964-d1.html",
        "description": "Bo luat Hinh su 2015 (sua doi 2017)",
    },
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def fetch_html(url: str, filepath: Path) -> bool:
    try:
        r = requests.get(url, headers=_HEADERS, timeout=20, allow_redirects=True)
        r.raise_for_status()
        if len(r.content) < 2048:
            print(f"  Response too small ({len(r.content)} bytes) - likely blocked")
            return False
        # Lưu HTML gốc để MarkItDown convert ở task3
        filepath.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def collect_legal_docs() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    success = 0

    # Đếm file đã có
    existing = {
        f.name
        for f in DATA_DIR.iterdir()
        if f.suffix.lower() in (".pdf", ".docx", ".doc", ".html")
        and f.stat().st_size > 1024
    }
    for name in existing:
        f = DATA_DIR / name
        print(f"  EXISTS: {name} ({f.stat().st_size // 1024} KB)")
        success += 1

    # Crawl các file còn thiếu
    for doc in LEGAL_SOURCES:
        if doc["filename"] in existing:
            continue
        print(f"Fetching: {doc['description']}")
        filepath = DATA_DIR / doc["filename"]
        if fetch_html(doc["url"], filepath):
            print(f"  Saved: {filepath.name} ({filepath.stat().st_size // 1024} KB)")
            success += 1
        else:
            print(f"  FAILED - lay tay tu: {doc['url']}")
        time.sleep(1)  # tránh bị rate-limit

    print(f"\nKet qua: {success} file tai {DATA_DIR}")
    if success < 3:
        print("Can toi thieu 3 file. Tai thu cong va dat vao data/landing/legal/")
    return success


if __name__ == "__main__":
    collect_legal_docs()
