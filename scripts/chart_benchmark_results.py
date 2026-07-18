#!/usr/bin/env python3
"""Chart matching baseline and variant Criterion result exports."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RESULT_PREFIX = "bench_result-"
BASELINE_SUFFIX = "-baseline.json"
SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]*$")
NS_PER_US = 1_000.0


@dataclass(frozen=True)
class SeriesKey:
    group_id: str
    function_id: str


@dataclass(frozen=True)
class Point:
    tree_log2: float
    duration_ns: float
    lower_ns: float
    upper_ns: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare every matching bench_result-*-baseline.json and "
            "bench_result-*-[VARIANT_KEY].json pair."
        )
    )
    parser.add_argument("variant_key", help="result key to compare with baseline")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path.cwd(),
        help="directory containing result JSON files (default: current directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="directory for charts and HTML (default: current directory)",
    )
    return parser.parse_args()


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_")
    return cleaned or "benchmark"


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("results"), list):
        raise RuntimeError(f"{path} is not a Criterion result export")
    return value


def finite_positive(value: Any, description: str, path: Path) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid {description} in {path}: {value!r}") from error
    if not math.isfinite(number) or number <= 0:
        raise RuntimeError(f"invalid {description} in {path}: {value!r}")
    return number


def result_series(path: Path) -> dict[SeriesKey, list[Point]]:
    series: dict[SeriesKey, list[Point]] = {}
    for result in load_json(path)["results"]:
        metadata = result.get("metadata")
        estimates = result.get("estimates")
        if not isinstance(metadata, dict) or not isinstance(estimates, dict):
            raise RuntimeError(
                f"{path} lacks benchmark metadata; regenerate it with the current exporter"
            )

        group_id = metadata.get("group_id")
        function_id = metadata.get("function_id") or "benchmark"
        tree_size = finite_positive(metadata.get("value_str"), "tree size", path)
        if not isinstance(group_id, str) or not isinstance(function_id, str):
            raise RuntimeError(f"invalid benchmark identity in {path}: {metadata!r}")

        throughput = metadata.get("throughput")
        query_count = throughput.get("Elements") if isinstance(throughput, dict) else None
        query_count = finite_positive(query_count, "query count/Elements throughput", path)

        mean = estimates.get("mean")
        interval = mean.get("confidence_interval") if isinstance(mean, dict) else None
        if not isinstance(mean, dict) or not isinstance(interval, dict):
            raise RuntimeError(f"missing mean confidence interval in {path}")

        duration = finite_positive(mean.get("point_estimate"), "mean duration", path)
        lower = finite_positive(interval.get("lower_bound"), "lower duration bound", path)
        upper = finite_positive(interval.get("upper_bound"), "upper duration bound", path)
        key = SeriesKey(group_id, function_id)
        series.setdefault(key, []).append(
            Point(
                tree_log2=math.log2(tree_size),
                duration_ns=duration / query_count,
                lower_ns=lower / query_count,
                upper_ns=upper / query_count,
            )
        )

    for key, points in series.items():
        points.sort(key=lambda point: point.tree_log2)
        x_values = [point.tree_log2 for point in points]
        if len(x_values) != len(set(x_values)):
            raise RuntimeError(f"duplicate tree sizes for {key} in {path}")
    return series


def subbenchmark_name(key: SeriesKey) -> str:
    group_parts = key.group_id.split("/")
    parts = group_parts[1:] + [key.function_id]
    return slug("-".join(part for part in parts if part))


def unique_chart_path(
    output_dir: Path,
    suite: str,
    key: SeriesKey,
    variant_key: str,
    paths_seen: set[Path],
) -> Path:
    base = f"bench_result-{slug(suite)}-{subbenchmark_name(key)}-{slug(variant_key)}"
    path = output_dir / f"{base}.png"
    if path in paths_seen:
        identity = f"{key.group_id}\0{key.function_id}".encode()
        path = output_dir / f"{base}-{hashlib.sha256(identity).hexdigest()[:8]}.png"
    paths_seen.add(path)
    return path


def render_chart(
    plt: Any,
    suite: str,
    key: SeriesKey,
    baseline: list[Point],
    variant: list[Point],
    variant_key: str,
    output_path: Path,
) -> None:
    if len(baseline) != len(variant):
        raise RuntimeError(
            f"baseline/variant point count mismatch for {suite}: "
            f"{key.group_id} / {key.function_id}"
        )
    for baseline_point, variant_point in zip(baseline, variant):
        if baseline_point.tree_log2 != variant_point.tree_log2:
            raise RuntimeError(
                f"baseline/variant tree sizes differ for {suite}: "
                f"{key.group_id} / {key.function_id}"
            )

    figure, axis = plt.subplots(figsize=(10, 6))
    for label, points, color in (
        ("baseline", baseline, "#3264a8"),
        (variant_key, variant, "#d35400"),
    ):
        x = [point.tree_log2 for point in points]
        y = [point.duration_ns / NS_PER_US for point in points]
        lower = [point.lower_ns / NS_PER_US for point in points]
        upper = [point.upper_ns / NS_PER_US for point in points]
        axis.plot(x, y, marker="o", linewidth=2, label=label, color=color)
        axis.fill_between(x, lower, upper, color=color, alpha=0.14)

    x_ticks = sorted(
        {point.tree_log2 for point in baseline} | {point.tree_log2 for point in variant}
    )
    axis.set_xticks(x_ticks)
    axis.set_xticklabels([f"{value:g}" for value in x_ticks])
    axis.set_yscale("log")
    axis.set_xlabel("log2(tree size)")
    axis.set_ylabel("Mean duration per query (us, log scale)")
    axis.set_title(f"{suite}: {key.group_id} / {key.function_id}")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()

    y_min, y_max = axis.get_ylim()
    upward_factor = 10**0.009
    downward_factor = 10**-0.0125
    lower_safe = y_min * (10**0.01)
    upper_safe = y_max * (10**-0.01)
    for baseline_point, variant_point in zip(baseline, variant):
        delta_fraction = (variant_point.duration_ns - baseline_point.duration_ns) / (
            baseline_point.duration_ns
        )
        delta_percent = round(delta_fraction * 100)
        label = f"{delta_percent:+.0f}%"
        baseline_y_us = baseline_point.duration_ns / NS_PER_US
        variant_y_us = variant_point.duration_ns / NS_PER_US
        if delta_fraction < 0:
            anchor_y = variant_y_us
            preferred_above = False
        else:
            anchor_y = baseline_y_us
            preferred_above = True

        preferred_y = anchor_y * (upward_factor if preferred_above else downward_factor)
        if lower_safe <= preferred_y <= upper_safe:
            place_above = preferred_above
            text_y = preferred_y
        else:
            place_above = not preferred_above
            text_y = anchor_y * (upward_factor if place_above else downward_factor)

        axis.text(
            variant_point.tree_log2,
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


def change_score(baseline: list[Point], variant: list[Point]) -> float:
    if len(baseline) != len(variant):
        raise RuntimeError("cannot score unmatched baseline/variant series")
    score = 0.0
    for baseline_point, variant_point in zip(baseline, variant):
        if baseline_point.tree_log2 != variant_point.tree_log2:
            raise RuntimeError("cannot score series with mismatched tree sizes")
        delta_fraction = (
            variant_point.duration_ns - baseline_point.duration_ns
        ) / baseline_point.duration_ns
        score += delta_fraction * delta_fraction
    return score


def write_html(
    output_path: Path,
    variant_key: str,
    charts: list[tuple[str, Path]],
) -> None:
    sections = []
    for title, chart_path in charts:
        encoded = base64.b64encode(chart_path.read_bytes()).decode("ascii")
        sections.append(
            "<section>"
            f"<h2>{html.escape(title)}</h2>"
            f'<img src="data:image/png;base64,{encoded}" '
            f'alt="{html.escape(title, quote=True)}">'
            f"<p><code>{html.escape(chart_path.name)}</code></p>"
            "</section>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Benchmark run: baseline vs {html.escape(variant_key)}</title>
  <style>
    body {{ margin: 0 auto; max-width: 1100px; padding: 2rem; font-family: system-ui, sans-serif; background: #f6f7f9; color: #202124; }}
    section {{ margin: 2rem 0; padding: 1rem; border-radius: .5rem; background: white; box-shadow: 0 1px 5px #0002; }}
    img {{ display: block; width: 100%; height: auto; }}
    h1, h2 {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Benchmark run: baseline vs {html.escape(variant_key)}</h1>
  <p>{len(charts)} charts generated from matching result-file pairs.</p>
  {''.join(sections)}
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def import_matplotlib() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("charting requires Python's matplotlib package") from error
    return matplotlib, plt


def generate_charts(
    variant_key: str,
    results_dir: Path,
    output_dir: Path,
) -> list[dict[str, str | float]]:
    if variant_key == "baseline":
        raise RuntimeError("VARIANT_KEY must identify a non-baseline result")
    if not SAFE_KEY.fullmatch(variant_key):
        raise RuntimeError(
            "VARIANT_KEY must contain only letters, digits, '.', '_', '+', ':', or '-'"
        )

    _, plt = import_matplotlib()
    results_dir = results_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[str, Path, Path]] = []
    for baseline_path in sorted(results_dir.glob(f"{RESULT_PREFIX}*{BASELINE_SUFFIX}")):
        suite = baseline_path.name[len(RESULT_PREFIX) : -len(BASELINE_SUFFIX)]
        variant_path = results_dir / f"{RESULT_PREFIX}{suite}-{variant_key}.json"
        if variant_path.is_file():
            pairs.append((suite, baseline_path, variant_path))

    if not pairs:
        raise RuntimeError(
            f"no baseline/result pairs found for variant {variant_key!r} in {results_dir}"
        )

    charts: list[tuple[str, Path]] = []
    metadata: list[dict[str, str]] = []
    paths_seen: set[Path] = set()
    for suite, baseline_path, variant_path in pairs:
        baseline_series = result_series(baseline_path)
        variant_series = result_series(variant_path)
        common_keys = sorted(
            baseline_series.keys() & variant_series.keys(),
            key=lambda key: (key.group_id, key.function_id),
        )
        for key in common_keys:
            chart_path = unique_chart_path(output_dir, suite, key, variant_key, paths_seen)
            render_chart(
                plt,
                suite,
                key,
                baseline_series[key],
                variant_series[key],
                variant_key,
                chart_path,
            )
            title = f"{suite}: {key.group_id} / {key.function_id}"
            charts.append((title, chart_path))
            metadata.append(
                {
                    "title": title,
                    "file_name": chart_path.name,
                    "suite": suite,
                    "group_id": key.group_id,
                    "function_id": key.function_id,
                    "change_score": change_score(baseline_series[key], variant_series[key]),
                }
            )

    if not charts:
        raise RuntimeError("matching result files contained no common benchmark series")

    html_path = output_dir / "latest_benchmark_run.html"
    write_html(html_path, variant_key, charts)
    return metadata


def main() -> int:
    args = parse_args()
    generate_charts(args.variant_key, args.results_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
