"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft + BeautifulSoup cho HTML.
PDF từ chinhphu.vn là scanned image -> dùng HTML tương ứng từ luatvietnam.vn.

    pip install "markitdown[pdf]" beautifulsoup4
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from bs4 import BeautifulSoup
from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _html_to_markdown(html_bytes: bytes) -> str:
    """Extract main text content from HTML using BeautifulSoup."""
    soup = BeautifulSoup(html_bytes, "html.parser")

    # Xoa script, style, nav, footer, quang cao
    for tag in soup.select("script, style, nav, footer, header, .ads, .advertisement, .sidebar"):
        tag.decompose()

    # Tim noi dung chinh
    content = ""
    for sel in [
        "div.content-document",   # luatvietnam.vn
        "div.vbContent",
        "div#vbContent",
        "div.content",
        "div.main-content",
        "article",
        "div#content",
        "body",
    ]:
        tag = soup.select_one(sel)
        if tag:
            text = tag.get_text(separator="\n", strip=True)
            if len(text) > len(content):
                content = text

    return content


def convert_legal_docs() -> int:
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()
    converted = 0

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc", ".html"):
            continue
        output_path = output_dir / f"{filepath.stem}.md"
        if output_path.exists() and output_path.stat().st_size > 500:
            print(f"  EXISTS: {output_path.name}")
            converted += 1
            continue

        print(f"Converting: {filepath.name}")
        try:
            if filepath.suffix.lower() == ".html":
                # HTML: dung BeautifulSoup extract text truc tiep
                content = _html_to_markdown(filepath.read_bytes())
                if len(content) < 200:
                    # Fallback: MarkItDown
                    result = md.convert(str(filepath))
                    content = result.text_content
            else:
                # PDF/DOCX: thu MarkItDown truoc
                result = md.convert(str(filepath))
                content = result.text_content

            if len(content) < 100:
                print(f"  SKIP: {filepath.name} -> {len(content)} chars (scanned PDF, no text layer)")
                continue

            output_path.write_text(content, encoding="utf-8")
            print(f"  Saved: {output_path.name} ({len(content):,} chars)")
            converted += 1
        except Exception as e:
            print(f"  FAILED: {e}")

    return converted


def convert_news_articles() -> int:
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue
        output_path = output_dir / f"{filepath.stem}.md"
        if output_path.exists() and output_path.stat().st_size > 100:
            print(f"  EXISTS: {output_path.name}")
            converted += 1
            continue
        print(f"Converting: {filepath.name}")
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            header = (
                f"# {data.get('title', 'Unknown')}\n\n"
                f"**Source:** {data.get('url', 'N/A')}\n"
                f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
            )
            content = header + data.get("content_markdown", "")
            output_path.write_text(content, encoding="utf-8")
            print(f"  Saved: {output_path.name} ({len(content):,} chars)")
            converted += 1
        except Exception as e:
            print(f"  FAILED: {e}")

    return converted


def convert_all():
    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    legal_count = convert_legal_docs()

    print("\n--- News Articles ---")
    news_count = convert_news_articles()

    print(f"\nDone: {legal_count} legal + {news_count} news -> {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
