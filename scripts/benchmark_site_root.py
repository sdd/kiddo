#!/usr/bin/env python3
"""Render the top-level benchmark site index and latest v5-v6 comparison page."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Any

from benchmark_site import RunSummary, load_run_summary, render_html, site_join, write_text
from chart_benchmark_results import (
    NS_PER_US,
    Point,
    SeriesKey,
    change_score,
    import_matplotlib,
    result_series,
    slug,
    subbenchmark_name,
)


RESULT_GLOB = "bench_result-*.json"
LINE_ORDER = ("v6", "v5")


def canonical_suite_name(suite: str) -> str:
    return re.sub(r"^v[0-9]+-", "", suite)


def latest_baseline_summary(line_root: Path) -> RunSummary | None:
    path = line_root / "baseline" / "latest" / "run.json"
    if not path.is_file():
        return None
    return load_run_summary(path)


def load_result_sets(
    results_dir: Path,
) -> dict[str, tuple[str, dict[SeriesKey, list[Point]]]]:
    sets: dict[str, tuple[str, dict[SeriesKey, list[Point]]]] = {}
    for path in sorted(results_dir.glob(RESULT_GLOB)):
        name = path.name
        if not name.startswith("bench_result-") or not name.endswith("-baseline.json"):
            continue
        suite = name[len("bench_result-") : -len("-baseline.json")]
        sets[canonical_suite_name(suite)] = (suite, result_series(path))
    return sets


def comparison_chart_path(
    output_dir: Path,
    suite: str,
    key: SeriesKey,
    left_label: str,
    right_label: str,
    seen: set[Path],
) -> Path:
    base = (
        f"bench_result-{slug(suite)}-{subbenchmark_name(key)}-"
        f"{slug(left_label)}-vs-{slug(right_label)}"
    )
    path = output_dir / f"{base}.png"
    suffix = 1
    while path in seen:
        path = output_dir / f"{base}-{suffix}.png"
        suffix += 1
    seen.add(path)
    return path


def render_comparison_chart(
    plt: Any,
    title: str,
    left_label: str,
    left: list[Point],
    right_label: str,
    right: list[Point],
    output_path: Path,
) -> None:
    if len(left) != len(right):
        raise RuntimeError(f"point count mismatch for {title}")
    for left_point, right_point in zip(left, right):
        if left_point.tree_log2 != right_point.tree_log2:
            raise RuntimeError(f"tree size mismatch for {title}")

    figure, axis = plt.subplots(figsize=(10, 6))
    for label, points, color in (
        (left_label, left, "#3264a8"),
        (right_label, right, "#d35400"),
    ):
        x = [point.tree_log2 for point in points]
        y = [point.duration_ns / NS_PER_US for point in points]
        lower = [point.lower_ns / NS_PER_US for point in points]
        upper = [point.upper_ns / NS_PER_US for point in points]
        axis.plot(x, y, marker="o", linewidth=2, label=label, color=color)
        axis.fill_between(x, lower, upper, color=color, alpha=0.14)

    x_ticks = sorted({point.tree_log2 for point in left} | {point.tree_log2 for point in right})
    axis.set_xticks(x_ticks)
    axis.set_xticklabels([f"{value:g}" for value in x_ticks])
    axis.set_yscale("log")
    axis.set_xlabel("log2(tree size)")
    axis.set_ylabel("Mean duration per query (us, log scale)")
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()

    y_min, y_max = axis.get_ylim()
    upward_factor = 10**0.009
    downward_factor = 10**-0.0125
    lower_safe = y_min * (10**0.01)
    upper_safe = y_max * (10**-0.01)
    for left_point, right_point in zip(left, right):
        delta_fraction = (right_point.duration_ns - left_point.duration_ns) / left_point.duration_ns
        delta_percent = round(delta_fraction * 100)
        label = f"{delta_percent:+.0f}%"
        left_y_us = left_point.duration_ns / NS_PER_US
        right_y_us = right_point.duration_ns / NS_PER_US
        if delta_fraction < 0:
            anchor_y = right_y_us
            preferred_above = False
        else:
            anchor_y = left_y_us
            preferred_above = True

        preferred_y = anchor_y * (upward_factor if preferred_above else downward_factor)
        if lower_safe <= preferred_y <= upper_safe:
            place_above = preferred_above
            text_y = preferred_y
        else:
            place_above = not preferred_above
            text_y = anchor_y * (upward_factor if place_above else downward_factor)

        axis.text(
            right_point.tree_log2,
            text_y,
            label,
            color="#1f8f3a" if delta_fraction < 0 else "#c0392b",
            fontsize=9,
            ha="center",
            va="bottom" if place_above else "top",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
                "pad": 0.25,
            },
        )

    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def render_latest_comparison(pages_root: Path, site_url_base: str | None) -> str | None:
    v5_root = pages_root / "v5"
    v6_root = pages_root / "v6"
    v5_summary = latest_baseline_summary(v5_root)
    v6_summary = latest_baseline_summary(v6_root)
    if v5_summary is None or v6_summary is None:
        return None

    v5_results = load_result_sets(v5_root / "baseline" / "latest" / "results")
    v6_results = load_result_sets(v6_root / "baseline" / "latest" / "results")
    shared_suites = sorted(v5_results.keys() & v6_results.keys())
    compare_root = pages_root / "compare" / "v5-v6-latest"
    charts_dir = compare_root / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    for path in charts_dir.glob("*.png"):
        path.unlink()

    _, plt = import_matplotlib()
    chart_rows: list[tuple[str, str, float]] = []
    seen: set[Path] = set()
    for canonical_suite in shared_suites:
        _, v5_series = v5_results[canonical_suite]
        _, v6_series = v6_results[canonical_suite]
        common_keys = sorted(
            v5_series.keys() & v6_series.keys(),
            key=lambda key: (key.group_id, key.function_id),
        )
        for key in common_keys:
            title = f"{canonical_suite}: {key.group_id} / {key.function_id}"
            chart_path = comparison_chart_path(
                charts_dir,
                canonical_suite,
                key,
                "v5",
                "v6",
                seen,
            )
            render_comparison_chart(
                plt,
                title,
                "v5",
                v5_series[key],
                "v6",
                v6_series[key],
                chart_path,
            )
            chart_rows.append((title, chart_path.name, change_score(v5_series[key], v6_series[key])))

    chart_rows.sort(key=lambda row: row[2], reverse=True)
    sections = []
    for title, file_name, _ in chart_rows:
        sections.append(
            f"<section><h2>{html.escape(title)}</h2>"
            f"<img src=\"charts/{file_name}\" alt=\"{html.escape(title, quote=True)}\"></section>"
        )

    v5_url = site_join(site_url_base, "v5/baseline/latest/index.html")
    v6_url = site_join(site_url_base, "v6/baseline/latest/index.html")
    body = f"""
<p><a href="../../index.html">Home</a></p>
<h1>Latest v5 vs latest v6 baseline comparison</h1>
<p class="meta">v5 baseline: <code>{v5_summary.run_id}</code> · <code>{v5_summary.sha[:12]}</code></p>
<p class="meta">v6 baseline: <code>{v6_summary.run_id}</code> · <code>{v6_summary.sha[:12]}</code></p>
<p>{f'<a href="{v5_url}">View latest v5 baseline</a>' if v5_url else ''}</p>
<p>{f'<a href="{v6_url}">View latest v6 baseline</a>' if v6_url else ''}</p>
<p>{len(chart_rows)} shared charts generated from the latest published baselines.</p>
{''.join(sections) if sections else '<section><p class="meta">No shared benchmark suites are published yet.</p></section>'}
"""
    write_text(compare_root / "index.html", render_html("Latest v5 vs latest v6 benchmark comparison", body))
    return "compare/v5-v6-latest/index.html"


def render_root_index(pages_root: Path, site_url_base: str | None) -> None:
    line_rows = []
    for line in LINE_ORDER:
        line_root = pages_root / line
        summary = latest_baseline_summary(line_root)
        if summary is None:
            continue
        line_rows.append(
            "<tr>"
            f"<td><a href=\"{line}/index.html\">{line}</a></td>"
            f"<td><code>{summary.baseline_ref_name}</code></td>"
            f"<td><a href=\"{line}/baseline/latest/index.html\">{summary.run_id}</a></td>"
            f"<td><code>{summary.sha[:12]}</code></td>"
            f"<td>{summary.created_at}</td>"
            "</tr>"
        )

    compare_rel = render_latest_comparison(pages_root, site_url_base)
    compare_link = (
        f'<p><a href="{compare_rel}">Compare latest v5 and v6 baselines</a></p>'
        if compare_rel
        else "<p class=\"meta\">The latest v5-v6 comparison page will appear after both lines publish a baseline.</p>"
    )
    body = f"""
<h1>Kiddo benchmark reports</h1>
<p>This index links the line-specific benchmark sites and the latest v5-v6 parity comparison.</p>
<section>
  <h2>Benchmark lines</h2>
  <table>
    <thead><tr><th>Line</th><th>Baseline branch</th><th>Latest baseline run</th><th>SHA</th><th>Created</th></tr></thead>
    <tbody>{''.join(line_rows) if line_rows else '<tr><td colspan="5">No line baselines published yet.</td></tr>'}</tbody>
  </table>
</section>
<section>
  <h2>Cross-line comparison</h2>
  {compare_link}
</section>
"""
    write_text(pages_root / "index.html", render_html("Kiddo benchmark reports", body))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages-root", type=Path, required=True)
    parser.add_argument("--site-url-base")
    args = parser.parse_args()
    render_root_index(args.pages_root.resolve(), args.site_url_base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
