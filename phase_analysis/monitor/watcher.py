"""Polling-based change detector with debounce/coalescing.

Deliberately no filesystem-event library (inotify/FSEvents/watchdog): a
plain directory listing works unmodified on network shares and any OS,
and the data rate here (new file every few seconds at most) makes polling
plenty responsive.

Backpressure handling: analysis is holistic, so there's no such thing as
"process just the new file" — every run reprocesses the whole current
set. That makes queuing a run per new file both wasteful and wrong (by
the time run #3 started, run #1 and #2 are already obsolete). Instead:

  1. Wait for the folder to stop changing for `quiet_seconds` (debounce) —
     avoids kicking off a run against a half-written frame.
  2. Run the (possibly slow) analysis against that stable snapshot.
  3. Re-snapshot immediately when it finishes, with no sleep. If new
     files landed while the run was in flight, that's the next snapshot
     to process; if not, go back to idle polling.

This way there is never more than one run in flight, and a slow run never
leaves a backlog — acquisition can keep racing ahead and the monitor
simply always analyzes the most recent stable state once it gets a
chance to.
"""

import os
import time


def snapshot(data_folder: str, extensions) -> tuple:
    """Cheap signature of folder contents: (name, size, mtime_ns) per
    matching file, sorted for a stable comparison."""
    exts = {e.lower() for e in extensions}
    entries = []
    for f in os.listdir(data_folder):
        if f.startswith('.') or os.path.splitext(f)[1].lower() not in exts:
            continue
        path = os.path.join(data_folder, f)
        try:
            st = os.stat(path)
        except OSError:
            continue
        entries.append((f, st.st_size, st.st_mtime_ns))
    return tuple(sorted(entries))


def watch(data_folder: str, extensions, poll_interval: float, quiet_seconds: float,
          on_stable_change, should_stop=lambda: False) -> None:
    """Block (polling) until should_stop(), calling on_stable_change(sig)
    once per stable snapshot that differs from the last one processed."""

    last_processed = None
    last_seen = None
    last_change_time = time.monotonic()

    while not should_stop():
        sig = snapshot(data_folder, extensions)
        now = time.monotonic()

        if sig != last_seen:
            last_seen = sig
            last_change_time = now

        stable = (now - last_change_time) >= quiet_seconds
        if stable and sig and sig != last_processed:
            on_stable_change(sig)
            last_processed = sig
            continue  # re-check immediately rather than sleeping

        time.sleep(poll_interval)
