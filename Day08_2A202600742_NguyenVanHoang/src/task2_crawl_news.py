"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hỗ trợ nhiều nguồn: VnExpress, VOV, Thanh Niên, Tiền Phong, VTV.
Lưu mỗi bài thành JSON với metadata: url, title, date_crawled, content_markdown.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

ARTICLE_URLS = [
    "https://vov.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-post1293496.vov",
    "https://vov.vn/giai-tri/long-nhat-truoc-khi-bi-bat-vi-ma-tuy-thi-phi-bua-vay-va-cac-scandal-gay-tranh-cai-post1293465.vov",
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    "https://thanhnien.vn/bat-tam-giam-rapper-binh-gold-185251006154630904.htm",
    "https://vnexpress.net/nha-thiet-ke-nguyen-cong-tri-bi-bat-vi-lien-quan-ma-tuy-4917929.html",
    "https://tienphong.vn/truy-to-ca-si-chi-dan-nguoi-mau-an-tay-va-225-bi-can-trong-duong-day-ma-tuy-post1832551.tpo",
    "https://vtv.vn/phap-luat/ca-si-chu-bin-bi-bat-vi-lien-quan-ma-tuy-20240607115007528.htm",
]

# Cấu hình selector theo từng domain
SITE_CONFIGS = {
    "vnexpress.net": {
        "title": ["h1.title-detail", "h1.title-news", "h1"],
        "content": ["article.fck_detail", "div.Normal", "div#article_content"],
        "desc": ["p.description"],
    },
    "vov.vn": {
        "title": ["h1.article-title", "h1[class*='title']", "h1"],
        "content": ["div.article-content", "div[class*='content-detail']", "div.entry-content"],
        "desc": ["p.article-sapo", "p[class*='sapo']"],
    },
    "thanhnien.vn": {
        "title": ["h1.detail-title", "h1[class*='title']", "h1"],
        "content": ["div#article-body", "div.detail-cmain", "div[class*='content']"],
        "desc": ["p.detail-sapo", "p[class*='sapo']"],
    },
    "tienphong.vn": {
        "title": ["h1.article__title", "h1[class*='title']", "h1"],
        "content": ["div.article__body", "div[class*='article-body']", "div[class*='cms-body']"],
        "desc": ["p.article__sapo", "p[class*='sapo']"],
    },
    "vtv.vn": {
        "title": ["h1.article-title", "h1[class*='title']", "h1"],
        "content": ["div.article-content", "div[class*='article-content']", "div[class*='content']"],
        "desc": ["p.article-sapo", "p[class*='sapo']"],
    },
}


def _get_domain(url: str) -> str:
    return urlparse(url).netloc.replace("www.", "")


def _find_first(soup: BeautifulSoup, selectors: list[str]) -> BeautifulSoup | None:
    for sel in selectors:
        tag = soup.select_one(sel)
        if tag:
            return tag
    return None


def fetch_article(url: str, timeout: int = 15) -> dict | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        domain = _get_domain(url)
        cfg = SITE_CONFIGS.get(domain, {
            "title": ["h1"],
            "content": ["article", "div[class*='content']", "div[class*='article']"],
            "desc": ["p[class*='sapo']", "p[class*='description']"],
        })

        # Title
        title_tag = _find_first(soup, cfg["title"])
        title = title_tag.get_text(strip=True) if title_tag else "Unknown"

        # Description/sapo
        desc_tag = _find_first(soup, cfg["desc"])
        description = desc_tag.get_text(strip=True) if desc_tag else ""

        # Main content
        content_div = _find_first(soup, cfg["content"])
        paragraphs = []
        if content_div:
            for p in content_div.find_all("p"):
                text = p.get_text(strip=True)
                if text and len(text) > 30:
                    paragraphs.append(text)

        content = (description + "\n\n" + "\n\n".join(paragraphs)).strip()

        # Fallback: quét toàn bộ <p> nếu content quá ngắn
        if len(content) < 200:
            all_p = []
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                if text and len(text) > 50:
                    all_p.append(text)
            content = "\n\n".join(all_p[:30])

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": content,
        }

    except Exception as e:
        print(f"  ✗ Lỗi crawl {url}: {e}")
        return None


def crawl_all():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url[:80]}...")
        article = fetch_article(url)

        if article:
            content_len = len(article.get("content_markdown", ""))
            filename = f"article_{i:02d}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  ✓ Saved: {filepath.name} ({content_len} chars)")
            saved += 1
        else:
            print(f"  ✗ Bỏ qua bài {i}")

        time.sleep(1)

    print(f"\n✓ Crawled {saved}/{len(ARTICLE_URLS)} articles → {DATA_DIR}")


if __name__ == "__main__":
    crawl_all()
