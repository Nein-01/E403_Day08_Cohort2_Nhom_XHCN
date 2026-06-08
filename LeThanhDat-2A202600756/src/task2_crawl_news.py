"""
Task 2 — Crawl bai bao ve nghe si lien quan toi ma tuy.

Dung Playwright de render JavaScript va lay noi dung day du tu cac bao
VnExpress, Tuoi Tre, Thanh Nien (nhung site nay deu render JS).
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    # Current articles from RSS (confirmed alive)
    "https://tuoitre.vn/hot-girl-dieu-hanh-duong-day-cung-cap-nuoc-vui-thuoc-lac-cho-nguoi-nuoc-ngoai-20260602145446559.htm",
    "https://vnexpress.net/hai-thanh-nien-duong-tinh-voi-ma-tuy-thong-chot-dam-nga-csgt-5082931.html",
    "https://vnexpress.net/nhieu-nguoi-nuoc-ngoai-phe-ma-tuy-trong-khach-san-o-tp-hcm-5082175.html",
    "https://vnexpress.net/duong-ham-buon-ma-tuy-bi-mat-duoi-san-cua-hang-o-my-5081471.html",
    "https://tuoitre.vn/nha-trang-tang-cuong-ra-soat-xet-nghiem-ma-tuy-gom-ca-nguoi-nuoc-ngoai-20260607143603699.htm",
    "https://vnexpress.net/ca-si-chau-viet-long-bi-ket-an-20-nam-tu-4043031.html",
    "https://vnexpress.net/nghe-si-chi-tai-bi-bat-vi-co-ma-tuy-4176888.html",
]

# CSS selectors theo tung domain
_TITLE_SELECTORS = [
    "h1.title-detail",        # VnExpress
    "h1.article-title",
    "h1.detail__headline",    # Thanh Nien
    "h1#article-title",       # Tuoi Tre
    "h1",
]

_CONTENT_SELECTORS = [
    "article.fck_detail",         # VnExpress
    "div.article-body",           # VnExpress
    "div#main-detail-body",       # Tuoi Tre
    "div.detail__cmain",          # Thanh Nien
    "div.detail-content",
    "div.content-detail",
    "div[data-role='content']",
    "article",
]


def extract(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    for sel in _TITLE_SELECTORS:
        tag = soup.select_one(sel)
        if tag:
            title = tag.get_text(strip=True)
            break

    content = ""
    for sel in _CONTENT_SELECTORS:
        tag = soup.select_one(sel)
        if tag:
            for rm in tag.select("script, style, .ads, .advertisement, figure, .sidebar"):
                rm.decompose()
            text = tag.get_text(separator="\n", strip=True)
            if len(text) > len(content):
                content = text

    return {
        "url": url,
        "title": title or "Unknown",
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content,
    }


async def crawl_with_playwright(url: str) -> str:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            )
        )
        page = await ctx.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        # Cho JS render xong
        await page.wait_for_timeout(3000)
        html = await page.content()
        await browser.close()
        return html


async def crawl_all():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Xoa file cu de crawl lai sach
    for old in DATA_DIR.glob("article_*.json"):
        old.unlink()

    success = 0
    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] {url}")
        try:
            html = await crawl_with_playwright(url)
            article = extract(html, url)
            content_len = len(article["content_markdown"])

            filepath = DATA_DIR / f"article_{i:02d}.json"
            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"  Saved: article_{i:02d}.json | {content_len} chars | {article['title'][:50]}")
            success += 1
        except Exception as e:
            print(f"  SKIP: {e}")

    print(f"\nDone: {success}/{len(ARTICLE_URLS)} articles -> {DATA_DIR}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
