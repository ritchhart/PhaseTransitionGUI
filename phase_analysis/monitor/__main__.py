"""Entry point: python -m phase_analysis.monitor path/to/config.json"""

import signal
import sys
import traceback

from phase_analysis.monitor.config import MonitorConfig
from phase_analysis.monitor.report import render
from phase_analysis.monitor.runner import PreprocessCache, run_once
from phase_analysis.monitor.watcher import watch


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m phase_analysis.monitor <config.json>", file=sys.stderr)
        return 2

    config = MonitorConfig.load(argv[0])
    preprocess_cache = PreprocessCache()
    state = {'run_count': 0, 'stop': False}

    def request_stop(signum, frame):
        state['stop'] = True

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, request_stop)

    def on_stable_change(sig):
        print(f"[monitor] change detected ({len(sig)} files) — running analysis...")
        last_error = None
        try:
            result = run_once(config, preprocess_cache)
        except Exception:
            last_error = traceback.format_exc()
            result = None
            print(f"[monitor] run failed:\n{last_error}", file=sys.stderr)

        if result is not None:
            state['run_count'] += 1
            render(result, config, config.output_html, state['run_count'],
                   last_error=last_error)
            print(f"[monitor] wrote {config.output_html} "
                  f"({result.n_times} frames, run #{state['run_count']})")
        elif last_error is None:
            print("[monitor] no matching files yet, nothing to report")

    print(f"[monitor] watching {config.data_folder} -> {config.output_html}")
    print(f"[monitor] poll every {config.poll_interval_seconds}s, "
          f"quiet period {config.quiet_seconds}s before each run")

    watch(config.data_folder, config.file_extensions,
          config.poll_interval_seconds, config.quiet_seconds,
          on_stable_change, should_stop=lambda: state['stop'])

    print("[monitor] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
