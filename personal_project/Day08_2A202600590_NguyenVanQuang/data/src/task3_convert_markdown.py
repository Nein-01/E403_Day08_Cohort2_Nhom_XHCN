"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Dùng MarkItDown (Microsoft) cho PDF/DOCX.
Fallback: antiword (có sẵn trên hệ thống) cho file .doc cũ (OLE2 format).
JSON news articles được extract content_markdown field trực tiếp.
"""

import json
import shutil
import subprocess
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Vị trí antiword (đi kèm Git for Windows)
ANTIWORD = shutil.which("antiword") or r"C:\Program Files\Git\mingw64\bin\antiword.exe"


def _convert_with_antiword(filepath: Path) -> str:
    """Dùng antiword để extract text từ .doc (OLE2 binary format)."""
    try:
        result = subprocess.run(
            [ANTIWORD, str(filepath)],
            capture_output=True,
            timeout=30,
        )
        text = result.stdout.decode("utf-8", errors="replace").strip()
        if not text:
            text = result.stderr.decode("utf-8", errors="replace").strip()
        return text
    except Exception as e:
        raise RuntimeError(f"antiword failed: {e}") from e


def convert_legal_docs():
    """Convert PDF/DOCX/DOC trong data/landing/legal/ sang markdown."""
    from markitdown import MarkItDown

    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()
    converted = 0

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue

        print(f"  Converting: {filepath.name}")
        text = ""

        # Ưu tiên MarkItDown
        try:
            result = md.convert(str(filepath))
            text = result.text_content.strip()
        except Exception:
            pass

        # Fallback: antiword cho .doc cũ
        if len(text) < 100 and filepath.suffix.lower() == ".doc":
            try:
                text = _convert_with_antiword(filepath)
                if text:
                    print(f"    (via antiword)")
            except Exception as e:
                print(f"    ✗ antiword cũng lỗi: {e}")

        if len(text) < 100:
            print(f"    ⚠ Output quá ngắn ({len(text)} chars), bỏ qua")
            continue

        output_path = output_dir / f"{filepath.stem}.md"
        header = f"# {filepath.stem}\n\n"
        header += f"**Nguồn:** {filepath.name}\n\n---\n\n"
        output_path.write_text(header + text, encoding="utf-8")
        print(f"    ✓ Saved: {output_path.name} ({len(text)} chars)")
        converted += 1

    print(f"  → Converted {converted} legal documents")
    return converted


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue

        print(f"  Converting: {filepath.name}")
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))

            title = data.get("title", "Unknown")
            url = data.get("url", "N/A")
            date_crawled = data.get("date_crawled", "N/A")
            content = data.get("content_markdown", "")

            if len(content) < 50:
                content = f"Xem bài gốc tại: {url}"

            header = f"# {title}\n\n"
            header += f"**Nguồn:** {url}\n"
            header += f"**Ngày crawl:** {date_crawled}\n\n---\n\n"

            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(header + content, encoding="utf-8")
            print(f"    ✓ Saved: {output_path.name} ({len(content)} chars)")
            converted += 1

        except Exception as e:
            print(f"    ✗ Lỗi: {e}")

    print(f"  → Converted {converted} news articles")
    return converted


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 60)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 60)

    print("\n--- Legal Documents ---")
    n_legal = convert_legal_docs()

    print("\n--- News Articles ---")
    n_news = convert_news_articles()

    print(f"\n✓ Done! Total: {n_legal} legal + {n_news} news → {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
