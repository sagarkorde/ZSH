"""Rate-limited, cached, resumable Esplora client (mempool.space / blockstream.info).

Every response body is stored gzip-compressed in one SQLite file so that the
prospective sample can be re-parsed without contacting the API again.
"""
import gzip
import json
import queue
import sqlite3
import threading
import time

import requests

from .config import CFG

USER_AGENT = "ZSH-v2-research/1.0 (+https://github.com/sagarkorde/ZSH)"


class Cache:
    def __init__(self, path):
        self.lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS resp (path TEXT PRIMARY KEY, host TEXT, "
                        "fetched REAL, status INTEGER, body BLOB)")
        self.db.commit()

    def get(self, path):
        with self.lock:
            row = self.db.execute("SELECT status, body FROM resp WHERE path=?", (path,)).fetchone()
        if row is None:
            return None
        return row[0], gzip.decompress(row[1]).decode("utf-8")

    def put(self, path, host, status, text):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO resp VALUES (?,?,?,?,?)",
                            (path, host, time.time(), status, gzip.compress(text.encode("utf-8"))))
            self.db.commit()

    def count(self):
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM resp").fetchone()[0]


class HostWorker:
    def __init__(self, base, min_interval, log):
        self.base = base.rstrip("/")
        self.min_interval = min_interval
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.last = 0.0
        self.log = log
        self.base_interval = min_interval
        self.ok_streak = 0

    def fetch(self, path, max_tries=8, yield_on_429=False):
        """Return (status, text); (429, None) when yield_on_429 and the host rate-limits."""
        delay = 5.0
        for attempt in range(max_tries):
            wait = self.min_interval - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                r = self.session.get(self.base + path, timeout=60)
            except requests.RequestException as e:
                self.log(f"  {self.base}{path}: {type(e).__name__}, retry in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 600)
                continue
            if r.status_code in (200, 404):
                if self.min_interval > self.base_interval and r.status_code == 200:
                    self.ok_streak += 1
                    if self.ok_streak >= 200:
                        self.min_interval = max(self.base_interval, self.min_interval / 2)
                        self.ok_streak = 0
                return r.status_code, r.text
            if r.status_code == 429:
                self.ok_streak = 0
                if self.min_interval < 8.0:
                    self.min_interval = min(self.min_interval * 2, 8.0)
                    self.log(f"  {self.base}: rate limited, interval now {self.min_interval:.1f}s")
                if yield_on_429:
                    return 429, None
            self.log(f"  {self.base}{path}: HTTP {r.status_code}, retry in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 600)
        return None, None


class Esplora:
    """Fetch many paths with one worker thread per host; results cached."""

    def __init__(self, cache_path, log, hosts=None, min_interval=None):
        fc = CFG["future"]
        self.hosts = hosts or [fc["api_primary"], fc["api_fallback"]]
        self.min_interval = fc["min_interval_s"] if min_interval is None else min_interval
        self.cache = Cache(cache_path)
        self.log = log

    def get_one(self, path, host_index=0):
        hit = self.cache.get(path)
        if hit is not None:
            return hit
        w = HostWorker(self.hosts[host_index], self.min_interval, self.log)
        status, text = w.fetch(path)
        if status is not None:
            self.cache.put(path, w.base, status, text)
        return status, text

    def get_many(self, paths, progress_every=500, hosts=None):
        todo = [p for p in dict.fromkeys(paths) if self.cache.get(p) is None]
        self.log(f"  {len(paths):,} paths, {len(todo):,} not cached")
        q = queue.Queue()
        for p in todo:
            q.put(p)
        done = [0]
        inflight = [0]
        failed = []
        lock = threading.Lock()
        t0 = time.time()

        def run(base):
            w = HostWorker(base, self.min_interval, self.log)
            pause = 30.0
            while True:
                with lock:
                    try:
                        p = q.get_nowait()
                        inflight[0] += 1
                    except queue.Empty:
                        if inflight[0] == 0:
                            return
                        p = None
                if p is None:
                    time.sleep(1.0)
                    continue
                status, text = w.fetch(p, yield_on_429=True)
                with lock:
                    inflight[0] -= 1
                    if status == 429:
                        q.put(p)  # hand the path to the other host and pause this one
                if status == 429:
                    time.sleep(pause)
                    pause = min(pause * 2, 900)
                    continue
                pause = 30.0
                if status is None:
                    with lock:
                        failed.append(p)
                else:
                    self.cache.put(p, w.base, status, text)
                with lock:
                    done[0] += 1
                    if done[0] % progress_every == 0:
                        rate = done[0] / (time.time() - t0)
                        self.log(f"  fetched {done[0]:,}/{len(todo):,} ({rate:.2f}/s)")

        threads = [threading.Thread(target=run, args=(h,), daemon=True) for h in (hosts or self.hosts)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if failed:
            self.log(f"  {len(failed)} paths failed; second pass, one host at a time")
            retry, failed = failed, []
            for base in (hosts or self.hosts):
                w = HostWorker(base, self.min_interval, self.log)
                still = []
                for p in retry:
                    status, text = w.fetch(p, max_tries=4)
                    if status is None:
                        still.append(p)
                    else:
                        self.cache.put(p, w.base, status, text)
                retry = still
            failed = retry
            if failed:
                self.log(f"  {len(failed)} paths failed after the second pass")
        return failed

    def json(self, path):
        hit = self.cache.get(path)
        if hit is None or hit[0] != 200:
            return None
        return json.loads(hit[1])

    def text(self, path):
        hit = self.cache.get(path)
        if hit is None or hit[0] != 200:
            return None
        return hit[1]
