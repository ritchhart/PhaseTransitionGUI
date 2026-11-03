"""Renders a RunResult as a single self-contained HTML file.

No JS, no server — the plots are embedded as base64 PNGs and the file is
written atomically (temp file + rename) so a browser refresh never sees a
half-written page. An optional <meta refresh> tag can auto-reload the
page; without it, the user just hits refresh whenever they want the
latest run.
"""

import base64
import html as html_lib
import io
import os

import numpy as np
from matplotlib.figure import Figure

from phase_analysis.monitor.runner import RunResult
from phase_analysis.monitor.config import MonitorConfig


def _fig_to_data_uri(fig: Figure, dpi: int = 110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('ascii')
    return f"data:image/png;base64,{encoded}"


def _segmentation_figure(result: RunResult) -> Figure:
    n_times = result.n_times
    t_axis = np.arange(n_times)
    transitions_sorted = sorted(result.segmentation['transitions'])
    seg = result.segmentation

    fig = Figure(figsize=(11, 11), dpi=100)
    axes = fig.subplots(4, 1, gridspec_kw={'height_ratios': [1, 2, 2, 2.5]})

    # Panel 0: temperature trace
    ax = axes[0]
    temps = result.temperatures
    if any(t is not None for t in temps):
        plot_temps = [t if t is not None else np.nan for t in temps]
        ax.plot(t_axis, plot_temps, color='firebrick', lw=1.2)
        ax.set_ylabel('T (°C)')
    else:
        ax.text(0.5, 0.5, 'No temperature metadata found',
                ha='center', va='center', transform=ax.transAxes, fontsize=9)
    for tr in transitions_sorted:
        ax.axvline(tr, color='lime', alpha=0.6, lw=1)
    ax.set_xlim(0, max(n_times - 1, 1))
    ax.set_title('Sample temperature')

    # Panel 1: dI/dt heatmap
    ax = axes[1]
    dI_dt = seg['dI_dt']
    vlim = np.percentile(np.abs(dI_dt), 99) or 1.0
    ax.imshow(dI_dt.T, aspect='auto', cmap='RdBu_r', vmin=-vlim, vmax=vlim,
              extent=[0, n_times - 1, result.two_theta[-1], result.two_theta[0]],
              origin='upper')
    for tr in transitions_sorted:
        ax.axvline(tr, color='lime', alpha=0.7, lw=1.2)
    ax.set_ylabel('2θ (°)')
    ax.set_title('dI/dt map')

    # Panel 2: data heatmap
    ax = axes[2]
    vmax = np.percentile(result.data, 99) or 1.0
    ax.imshow(result.data, aspect='auto', cmap='bone_r', vmin=0, vmax=vmax,
              extent=[result.two_theta[0], result.two_theta[-1], n_times, 0],
              origin='upper')
    for tr in transitions_sorted:
        ax.axhline(tr, color='lime', alpha=0.7, lw=1.2)
    ax.set_xlabel('2θ (°)')
    ax.set_ylabel('Frame')
    ax.set_title('Data with transitions')

    # Panel 3: indicator curves
    ax = axes[3]
    styles = [
        ('dissimilarity', 'Dissimilarity', '#e41a1c', '///'),
        ('intensity_change', '|dI/dt|', '#377eb8', '\\\\\\'),
        ('channel_change', '# channels', '#4daf4a', '|||'),
        ('rank_change', 'Rank change', '#984ea3', '---'),
    ]
    for key, label, color, hatch in styles:
        ax.fill_between(t_axis, seg[key], facecolor='none', edgecolor=color,
                         hatch=hatch, linewidth=0.0, label=label)
        ax.plot(t_axis, seg[key], color=color, lw=0.8, alpha=0.7)
    ax.plot(t_axis, seg['combined_score'], 'k-', lw=1.5, alpha=0.9, label='Combined')
    ax.axhline(seg['threshold'], color='orange', lw=2, label='Threshold')
    for tr in transitions_sorted:
        ax.axvline(tr, color='red', alpha=0.5, linestyle='--')
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Score')
    ax.set_title('Phase transition indicators')
    ax.legend(loc='upper right', fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _search_figure(result: RunResult) -> Figure:
    fig = Figure(figsize=(11, 8.5), dpi=100)
    if result.search_result is not None:
        result.search_result.plot(
            fig, transitions=sorted(result.segmentation['transitions']),
            file_name=os.path.basename(result.data_folder))
    else:
        ax = fig.add_subplot(111)
        msg = result.search_error or "No search result available"
        ax.text(0.5, 0.5, msg, ha='center', va='center', wrap=True, fontsize=11)
        ax.axis('off')
    return fig


def _temperature_summary(temperatures) -> str:
    valid = [t for t in temperatures if t is not None]
    if not valid:
        return "no temperature metadata found"
    missing = len(temperatures) - len(valid)
    note = f" ({missing} frame(s) missing metadata)" if missing else ""
    return f"{min(valid):.1f}–{max(valid):.1f} °C{note}"


def render(result: RunResult, config: MonitorConfig, out_path: str,
           run_count: int, last_error: str = None) -> None:
    seg_fig = _segmentation_figure(result)
    seg_img = _fig_to_data_uri(seg_fig)

    search_fig = _search_figure(result)
    search_img = _fig_to_data_uri(search_fig)

    title = html_lib.escape(config.title or os.path.basename(
        result.data_folder.rstrip(os.sep)) or "Phase Transition Monitor")

    n_transitions = len(result.segmentation['transitions'])
    n_segments = len(set(result.segment_labels.tolist()))

    skipped_html = ""
    if result.skipped_files:
        items = "".join(
            f"<li><code>{html_lib.escape(os.path.basename(p))}</code> — "
            f"{html_lib.escape(reason)}</li>"
            for p, reason in result.skipped_files)
        skipped_html = f"""
        <details>
          <summary>{len(result.skipped_files)} file(s) skipped this run</summary>
          <ul>{items}</ul>
        </details>"""

    error_html = ""
    if last_error:
        error_html = (f'<p class="error">Last run failed: '
                       f'{html_lib.escape(last_error)}</p>')

    refresh_tag = ""
    if config.html_refresh_seconds:
        refresh_tag = (f'<meta http-equiv="refresh" '
                        f'content="{int(config.html_refresh_seconds)}">')

    from datetime import datetime
    run_time = datetime.fromtimestamp(result.timestamp).strftime('%Y-%m-%d %H:%M:%S')

    doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
{refresh_tag}
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 2em auto;
         max-width: 1100px; color: #222; }}
  h1 {{ margin-bottom: 0.2em; }}
  .meta {{ color: #555; font-size: 0.95em; margin-bottom: 1.5em; }}
  .meta table {{ border-collapse: collapse; }}
  .meta td {{ padding: 2px 10px 2px 0; vertical-align: top; }}
  .meta td:first-child {{ color: #777; white-space: nowrap; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  .error {{ color: #a33; font-weight: 600; }}
  section {{ margin-bottom: 2.5em; }}
  code {{ background: #f2f2f2; padding: 1px 4px; border-radius: 3px; }}
  footer {{ color: #888; font-size: 0.85em; margin-top: 3em; }}
</style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">
    <table>
      <tr><td>Data folder</td><td><code>{html_lib.escape(result.data_folder)}</code></td></tr>
      <tr><td>Frames analyzed</td><td>{result.n_times}</td></tr>
      <tr><td>Temperature range</td><td>{_temperature_summary(result.temperatures)}</td></tr>
      <tr><td>Transitions / segments</td><td>{n_transitions} / {n_segments}</td></tr>
      <tr><td>Search method</td><td>{html_lib.escape(result.method_display_name)}</td></tr>
      <tr><td>Last run</td><td>{run_time} (run #{run_count})</td></tr>
    </table>
    {error_html}
    {skipped_html}
  </div>

  <section>
    <h2>Segmentation</h2>
    <img src="{seg_img}" alt="segmentation plot">
  </section>

  <section>
    <h2>{html_lib.escape(result.method_display_name)}</h2>
    <p>{html_lib.escape(result.search_result.summary) if result.search_result else ''}</p>
    <img src="{search_img}" alt="search result plot">
  </section>

  <footer>
    Refresh this page for the latest run.
    {"Auto-refreshes every " + str(config.html_refresh_seconds) + "s." if config.html_refresh_seconds else ""}
  </footer>
</body>
</html>
"""

    out_dir = os.path.dirname(out_path) or '.'
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = out_path + '.tmp'
    with open(tmp_path, 'w') as f:
        f.write(doc)
    os.replace(tmp_path, out_path)
