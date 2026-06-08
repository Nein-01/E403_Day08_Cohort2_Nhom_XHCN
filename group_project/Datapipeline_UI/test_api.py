"""Test API endpoint via httpx (không cần TestClient)."""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import httpx

# Start server in subprocess for testing
import subprocess, time, threading

proc = subprocess.Popen(
    [r".venv\Scripts\python.exe", "-m", "uvicorn", "app:app", "--port", "8765", "--no-access-log"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
time.sleep(3)  # wait for startup

try:
    # Test health
    r = httpx.get("http://localhost:8765/api/health", timeout=10)
    print("Health:", r.json())

    # Test query
    r = httpx.post(
        "http://localhost:8765/api/query",
        json={"question": "Rapper Binh Gold bi bat vi ly do gi?"},
        timeout=30
    )
    data = r.json()
    print(f"\nStatus: {r.status_code}")
    print(f"Answer (200 chars): {data.get('answer', '')[:200]}")
    print(f"Sources: {len(data.get('sources', []))}")
    for s in data.get('sources', [])[:3]:
        print(f"  - [{s['score']}%] {s['title'][:60]}")
    print("\n[PASS] API test OK!")
finally:
    proc.terminate()
    proc.wait()
