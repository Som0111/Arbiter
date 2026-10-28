#!/usr/bin/env python
"""Ping /health every N minutes. An experiment to reduce Render cold starts, not a fix.

Usage: python scripts/keep_alive.py --url https://arbiter-dzrv.onrender.com
Or:    use cron-job.org (free) to call GET /health every 10 minutes.
"""
import argparse
import time
import urllib.request


def ping(url: str) -> None:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=10) as r:
            print(f"[{time.strftime('%H:%M:%S')}] {r.status}")
    except Exception as exc:  # noqa: BLE001
        print(f"[{time.strftime('%H:%M:%S')}] FAIL - {exc}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="https://arbiter-dzrv.onrender.com")
    p.add_argument("--interval", type=int, default=10, help="minutes between pings")
    args = p.parse_args()
    print(f"Pinging {args.url}/health every {args.interval} min. Ctrl-C to stop.")
    while True:
        ping(args.url)
        time.sleep(args.interval * 60)
