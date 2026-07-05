#!/usr/bin/env python3
"""Build an OWOD-style result table from experiment logs.

The script reads real training/evaluation outputs from each run's log.txt.
Use --demo only for a local layout smoke test with generated fake logs.
"""

import argparse
import csv
import html
import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path


TASK_LAYOUT = {
    1: [
        ("u_recall", "U-Recall (↑)"),
        ("current_map", "mAP Current known (↑)"),
    ],
    2: [
        ("u_recall", "U-Recall (↑)"),
        ("previous_map", "mAP Previously known (↑)"),
        ("current_map", "mAP Current known (↑)"),
        ("both_map", "mAP Both (↑)"),
    ],
    3: [
        ("u_recall", "U-Recall (↑)"),
        ("previous_map", "mAP Previously known (↑)"),
        ("current_map", "mAP Current known (↑)"),
        ("both_map", "mAP Both (↑)"),
    ],
    4: [
        ("previous_map", "mAP Previously known (↑)"),
        ("current_map", "mAP Current known (↑)"),
        ("both_map", "mAP Both (↑)"),
    ],
}

METRIC_MAP = {
    "u_recall": "U_R50",
    "previous_map": "PK_AP50",
    "current_map": "CK_AP50",
    "both_map": "K_AP50",
}


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def coerce_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def format_metric(value):
    value = coerce_float(value)
    if value is None:
        return "-"
    return f"{value:.1f}"


def format_params(params):
    params = coerce_float(params)
    if params is None:
        return "-"
    return f"{params / 1_000_000:.2f}M"


def format_param_delta(params, baseline_params):
    params = coerce_float(params)
    baseline_params = coerce_float(baseline_params)
    if params is None or baseline_params in (None, 0):
        return "-"
    delta = (params - baseline_params) / baseline_params * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def task_from_path(path):
    match = re.search(r"(?:^|/|\\)t([1-4])(?:_|$|/|\\)", str(path))
    if match:
        return int(match.group(1))
    return None


def latest_metrics_from_log(run_dir):
    log_path = Path(run_dir) / "log.txt"
    if not log_path.exists():
        raise FileNotFoundError(f"missing log file: {log_path}")

    latest = None
    with log_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {log_path}:{line_no}") from exc
            metrics = record.get("test_metrics")
            if isinstance(metrics, dict) and metrics:
                latest = record

    if latest is None:
        raise ValueError(f"no non-empty test_metrics found in {log_path}")
    return latest


def load_runs(manifest):
    rows = []
    for item in manifest.get("runs", []):
        run_dir = Path(item["path"])
        task = int(item.get("task") or task_from_path(run_dir) or 0)
        if task not in TASK_LAYOUT:
            raise ValueError(f"run {run_dir} has invalid or missing task: {task}")

        record = latest_metrics_from_log(run_dir)
        metrics = record["test_metrics"]
        row = {
            "method": item["method"],
            "task": task,
            "path": str(run_dir),
            "epoch": record.get("epoch"),
            "n_parameters": item.get("n_parameters", record.get("n_parameters")),
            "gpus": item.get("gpus", manifest.get("gpus", "")),
            "trained_at": item.get("trained_at", manifest.get("trained_at", "")),
            "duration": item.get("duration", record.get("training_time", "")),
            "metrics": metrics,
        }
        rows.append(row)
    return rows


def grouped_results(rows):
    methods = []
    seen = set()
    result = {}
    for row in rows:
        method = row["method"]
        if method not in seen:
            methods.append(method)
            seen.add(method)
        result.setdefault(method, {})[row["task"]] = row
    return methods, result


def baseline_params(methods, result, baseline_name):
    if baseline_name not in result:
        return None
    values = [
        coerce_float(row.get("n_parameters"))
        for row in result[baseline_name].values()
        if coerce_float(row.get("n_parameters")) is not None
    ]
    return values[0] if values else None


def best_values(methods, result):
    best = {}
    for task, columns in TASK_LAYOUT.items():
        for key, _ in columns:
            values = []
            for method in methods:
                row = result.get(method, {}).get(task)
                if not row:
                    continue
                values.append(coerce_float(row["metrics"].get(METRIC_MAP[key])))
            values = [value for value in values if value is not None]
            if values:
                best[(task, key)] = max(values)
    return best


def collect_meta(rows, manifest):
    gpus = sorted({str(row["gpus"]) for row in rows if row.get("gpus")})
    trained = sorted({str(row["trained_at"]) for row in rows if row.get("trained_at")})
    source = manifest.get("source", "experiment log.txt files")
    gpus_label = ", ".join(gpus) if gpus else str(manifest.get("gpus", "unknown"))
    trained_label = ", ".join(trained) if trained else str(manifest.get("trained_at", "unknown"))
    return source, gpus_label, trained_label


def render_html(manifest, rows, output_path):
    methods, result = grouped_results(rows)
    baseline_name = manifest.get("baseline", "PROB")
    baseline = baseline_params(methods, result, baseline_name)
    best = best_values(methods, result)
    source, gpus_label, trained_label = collect_meta(rows, manifest)
    title = manifest.get("title", "OWOD Result Summary")

    col_count = 3 + sum(len(cols) for cols in TASK_LAYOUT.values())
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:Arial,'Noto Sans SC',sans-serif;margin:24px;color:#111;background:#fff;}",
        ".wrap{max-width:1800px;margin:0 auto;}",
        "h1{font-size:24px;margin:0 0 6px 0;font-weight:700;}",
        ".meta{font-size:13px;color:#333;margin:0 0 16px 0;line-height:1.5;}",
        "table{border-collapse:collapse;width:100%;table-layout:fixed;border-top:3px solid #111;border-bottom:3px solid #111;}",
        "th,td{padding:8px 8px;text-align:center;vertical-align:middle;border-right:1px solid #444;white-space:normal;word-break:keep-all;}",
        "th:last-child,td:last-child{border-right:0;}",
        "thead th{font-weight:700;font-size:15px;line-height:1.2;border-bottom:1px solid #111;}",
        "thead tr.group th{font-size:18px;background:#fff;}",
        "thead tr.sub th{font-size:13px;background:#eaf4ff;}",
        "thead tr.sub th.u{background:#fffde9;}",
        "tbody td{font-size:15px;line-height:1.2;border-bottom:1px solid #ddd;}",
        "tbody tr.baseline td{font-weight:700;}",
        "tbody tr:nth-child(even){background:#fafafa;}",
        "tbody tr.section td{border-top:2px double #111;}",
        "td.method{text-align:left;font-weight:700;}",
        "td.source{font-size:12px;color:#333;text-align:left;word-break:break-all;}",
        ".best{font-weight:800;background:#fff4c2;}",
        ".missing{color:#999;}",
        ".foot{font-size:12px;color:#444;margin-top:10px;}",
        "</style></head><body><div class='wrap'>",
        f"<h1>{html.escape(title)}</h1>",
        f"<div class='meta'>GPU: {html.escape(gpus_label)} | Trained at: {html.escape(trained_label)} | Baseline: {html.escape(baseline_name)} | Source: {html.escape(source)}</div>",
        "<table>",
        "<colgroup>",
        "<col style='width:180px'><col style='width:95px'><col style='width:105px'>",
    ]
    for task in TASK_LAYOUT:
        for key, _ in TASK_LAYOUT[task]:
            width = "90px" if key == "u_recall" else "100px"
            parts.append(f"<col style='width:{width}'>")
    parts.extend(["</colgroup>", "<thead>", "<tr class='group'>"])
    parts.append("<th rowspan='2'>Method</th><th rowspan='2'>Params</th><th rowspan='2'>Params vs PROB</th>")
    for task, columns in TASK_LAYOUT.items():
        parts.append(f"<th colspan='{len(columns)}'>Task {task}</th>")
    parts.extend(["</tr>", "<tr class='sub'>"])
    for columns in TASK_LAYOUT.values():
        for key, label in columns:
            cls = " class='u'" if key == "u_recall" else ""
            parts.append(f"<th{cls}>{html.escape(label)}</th>")
    parts.extend(["</tr>", "</thead>", "<tbody>"])

    for method in methods:
        row_class = "baseline" if method == baseline_name else ""
        params_values = [
            coerce_float(row.get("n_parameters"))
            for row in result.get(method, {}).values()
            if coerce_float(row.get("n_parameters")) is not None
        ]
        params = params_values[0] if params_values else None
        parts.append(f"<tr class='{row_class}'>")
        parts.append(f"<td class='method'>{html.escape(method)}</td>")
        parts.append(f"<td>{format_params(params)}</td>")
        parts.append(f"<td>{format_param_delta(params, baseline)}</td>")

        for task, columns in TASK_LAYOUT.items():
            task_row = result.get(method, {}).get(task)
            for key, _ in columns:
                if not task_row:
                    parts.append("<td class='missing'>-</td>")
                    continue
                value = coerce_float(task_row["metrics"].get(METRIC_MAP[key]))
                cell_class = "best" if value is not None and value == best.get((task, key)) else ""
                parts.append(f"<td class='{cell_class}'>{format_metric(value)}</td>")
        parts.append("</tr>")

    parts.extend(["</tbody>", "</table>"])
    parts.append(f"<div class='foot'>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Values are parsed from each run's latest non-empty <code>test_metrics</code> in <code>log.txt</code>.</div>")
    parts.extend(["</div></body></html>"])
    Path(output_path).write_text("\n".join(parts), encoding="utf-8")


def write_csv(rows, output_path):
    fieldnames = [
        "method", "task", "path", "epoch", "gpus", "trained_at", "n_parameters",
        "U_R50", "PK_AP50", "CK_AP50", "K_AP50", "WI", "AOSA",
    ]
    with Path(output_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in fieldnames}
            out.update({
                "U_R50": row["metrics"].get("U_R50", ""),
                "PK_AP50": row["metrics"].get("PK_AP50", ""),
                "CK_AP50": row["metrics"].get("CK_AP50", ""),
                "K_AP50": row["metrics"].get("K_AP50", ""),
                "WI": row["metrics"].get("WI", ""),
                "AOSA": row["metrics"].get("AOSA", ""),
            })
            writer.writerow(out)


def fake_record(epoch, params, task, base):
    metrics = {
        "U_R50": base + task * 1.7,
        "CK_AP50": base + task * 2.4 + 30,
        "K_AP50": base + task * 2.0 + 28,
        "WI": max(0.0, 0.12 - task * 0.01),
        "AOSA": int(900 - task * 80 - base * 4),
    }
    if task > 1:
        metrics["PK_AP50"] = base + task * 2.2 + 42
    return {
        "train_loss": 1.0 / task,
        "test_metrics": metrics,
        "epoch": epoch,
        "n_parameters": params,
    }


def make_demo(output_dir):
    demo_root = Path(output_dir) / "fake_runs"
    if demo_root.exists():
        shutil.rmtree(demo_root)
    methods = [
        ("PROB", 41_200_000, 8.0),
        ("MyModel-A", 43_000_000, 10.5),
        ("MyModel-B", 39_800_000, 9.0),
    ]
    runs = []
    for method, params, base in methods:
        for task in range(1, 5):
            run_dir = demo_root / method / f"t{task}"
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / "log.txt").open("w", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": 0, "n_parameters": params, "test_metrics": {}}) + "\n")
                f.write(json.dumps(fake_record(40 + task, params, task, base)) + "\n")
            runs.append({
                "method": method,
                "task": task,
                "path": str(run_dir),
                "gpus": "4 cards",
                "trained_at": "2026-07-05 18:10",
            })
    return {
        "title": "OWOD Final Result Table Demo",
        "baseline": "PROB",
        "gpus": "4 cards",
        "trained_at": "2026-07-05 18:10",
        "source": str(demo_root),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate an OWOD-style result table.")
    parser.add_argument("--manifest", help="JSON manifest listing real experiment run directories.")
    parser.add_argument("--output-dir", default="results/tables", help="directory for generated table files")
    parser.add_argument("--html-name", default="owod_results.html")
    parser.add_argument("--csv-name", default="owod_results.csv")
    parser.add_argument("--demo", action="store_true", help="generate fake logs under output-dir and render a layout demo")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        manifest = make_demo(output_dir)
        manifest_path = output_dir / "demo_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    elif args.manifest:
        manifest = read_json(args.manifest)
        manifest_path = Path(args.manifest)
    else:
        raise SystemExit("Provide --manifest for real runs or --demo for a local smoke test.")

    rows = load_runs(manifest)
    html_path = output_dir / args.html_name
    csv_path = output_dir / args.csv_name
    render_html(manifest, rows, html_path)
    write_csv(rows, csv_path)

    print(f"manifest: {manifest_path}")
    print(f"html: {html_path}")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
