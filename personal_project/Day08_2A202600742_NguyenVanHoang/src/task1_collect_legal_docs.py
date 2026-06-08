"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Các văn bản đã tải về data/landing/legal/:
    1. 73_2021_QH14_445185.doc  — Luật Phòng, chống ma tuý 2021 (Luật số 73/2021/QH14)
       Ngày ban hành: 30/03/2021  |  141 KB
    2. 69_2025_QH15_603983.doc  — Luật sửa đổi, bổ sung Luật Phòng, chống ma tuý 69/2025/QH15
       Ngày ban hành: 2025  |  156 KB
    3. 28_2026_ND-CP_690473.doc — Nghị định 28/2026/NĐ-CP hướng dẫn thi hành Luật Phòng, chống ma tuý
       Ngày ban hành: 2026  |  1044 KB

Nguồn: thuvienphapluat.vn
"""

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

LEGAL_FILES = [
    {
        "filename": "73_2021_QH14_445185.doc",
        "name": "Luật Phòng, chống ma tuý 2021",
        "number": "73/2021/QH14",
        "date": "2021-03-30",
        "source": "thuvienphapluat.vn",
    },
    {
        "filename": "69_2025_QH15_603983.doc",
        "name": "Luật sửa đổi, bổ sung Luật Phòng, chống ma tuý",
        "number": "69/2025/QH15",
        "date": "2025-01-01",
        "source": "thuvienphapluat.vn",
    },
    {
        "filename": "28_2026_ND-CP_690473.doc",
        "name": "Nghị định hướng dẫn thi hành Luật Phòng, chống ma tuý",
        "number": "28/2026/NĐ-CP",
        "date": "2026-01-01",
        "source": "thuvienphapluat.vn",
    },
]


def setup_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def verify_files() -> bool:
    """Kiểm tra tất cả file đã có đầy đủ."""
    missing = []
    for info in LEGAL_FILES:
        path = DATA_DIR / info["filename"]
        if not path.exists() or path.stat().st_size < 1024:
            missing.append(info["filename"])
    if missing:
        print(f"Thiếu file: {missing}")
        return False
    print(f"OK: {len(LEGAL_FILES)} file pháp luật đã có tại {DATA_DIR}")
    return True


if __name__ == "__main__":
    setup_directory()
    verify_files()
