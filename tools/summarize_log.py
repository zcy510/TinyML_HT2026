"""Summarise one or more client CSV logs, grouped by run setting (resolution + JPEG quality).

python -m tools.summarize_log logs/remote_client_log.csv --deadline-ms 500
"""
from __future__ import annotations

import argparse

import pandas as pd


def summarize(df: pd.DataFrame, deadline_ms: float) -> pd.DataFrame:
    df = df.copy()
    df["ok"] = df["ok"].astype(str).str.lower() == "true"
    # group by the run setting (max width + JPEG quality); older logs: by resolution
    if "setting" in df.columns:
        df["res"] = df["setting"].astype(str)
    else:
        df["res"] = df["sent_w"].astype(str) + "x" + df["sent_h"].astype(str)
    df["in_deadline"] = df["ok"] & (df["e2e_ms"] <= deadline_ms)
    ok = df[df["ok"]]
    g = ok.groupby("res")
    out = pd.DataFrame({
        "frames": df.groupby("res").size(),
        "success_rate": df.groupby("res")["ok"].mean(),
        f"within_{int(deadline_ms)}ms": df.groupby("res")["in_deadline"].mean(),
        "e2e_mean": g["e2e_ms"].mean(),
        "e2e_p50": g["e2e_ms"].median(),
        "e2e_p95": g["e2e_ms"].quantile(0.95),
        "infer_mean": g["inference_ms"].mean(),
        "network_mean": g["network_ms"].mean(),
        "upload_kb": df.groupby("res")["upload_bytes"].mean() / 1024,
        "dets_per_frame": g["num_detections"].mean(),
    })
    return out.round(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--deadline-ms", type=float, default=500.0)
    args = ap.parse_args()
    df = pd.concat([pd.read_csv(p) for p in args.logs], ignore_index=True)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summarize(df, args.deadline_ms))


if __name__ == "__main__":
    main()
