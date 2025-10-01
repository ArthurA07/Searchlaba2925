#!/usr/bin/env python3
import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import requests


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch evaluate folder via /api/v1/infer")
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--api", default="http://127.0.0.1:8000/api/v1/infer")
    ap.add_argument("--out", type=Path, default=Path("out_eval"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    for p in sorted(args.images.glob("*.*")):
        if p.suffix.lower() not in {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}:
            continue
        t0 = time.time()
        with p.open("rb") as f:
            resp = requests.post(args.api, files={"file": (p.name, f, "image/jpeg")}, timeout=120)
        dt = (time.time() - t0) * 1000.0
        if resp.status_code != 200:
            rows.append({"name": p.name, "status": resp.status_code, "ms": dt})
            continue
        js = resp.json()
        found = len(js.get("detections", []))
        avg_score = 0.0
        if found:
            avg_score = sum(d.get("score", 0.0) for d in js["detections"]) / found
        rows.append({"name": p.name, "status": 200, "ms": dt, "found": found, "avg_score": avg_score})

    # CSV
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["name","status","ms","found","avg_score"])
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    # JSON
    (args.out / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[ok] wrote:", args.out / "summary.csv", args.out / "summary.json")


if __name__ == "__main__":
    main()


