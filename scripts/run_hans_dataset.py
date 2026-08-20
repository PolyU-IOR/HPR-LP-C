#!/usr/bin/env python3
"""Run HPR-LP-C over Hans and maintain a resumable paper-style CSV."""

import argparse
import csv
import math
import os
import queue
import shlex
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path


DEFAULT_HPRLP_ENVIRONMENT = {
    "HPRLP_ENABLE_COMPRESSIBLE_MEMORY": "1",
    "HPRLP_USE_ROW_REDUCTION": "1",
    "HPRLP_USE_ROW_COMPRESSED_AUTOTUNE": "1",
    "HPRLP_USE_REDUCED_COMPRESSED_AUTOTUNE": "1",
    "HPRLP_DEFER_REDUCED_EMPTY_ROWS_TO_CHECK": "1",
    "HPRLP_USE_REDUCED_NONEMPTY_CUSPARSE": "1",
    "HPRLP_REDUCED_RESET_MASK_ON_RESTART": "1",
    "HPRLP_REDUCED_RESTART_MASK_MIN_RECOVERY": "0.25",
    "HPRLP_REDUCED_RESTART_MASK_MIN_SAVED_COLUMNS": "25000",
    "HPRLP_REDUCED_RESTART_MASK_MIN_CURRENT_COLUMNS": "95000",
}
for name, value in DEFAULT_HPRLP_ENVIRONMENT.items():
    os.environ.setdefault(name, value)


CSV_HEADER = [
    "name", "iter", "total_time", "presolve_time", "setup_time",
    "scaling_time", "analyze_time", "power_iteration_time", "solve_time",
    "folding_time",
    "res", "primal_obj", "gap", "status", "iter_4", "time_4",
    "iter_6", "time_6", "iter_8", "time_8",
    "reduced_active_iteration_ratio", "reduced_average_column_ratio",
    "reduced_average_nnz_ratio", "reduced_minimum_column_ratio",
    "reduced_minimum_nnz_ratio",
]
TIME_FIELDS = [
    "total_time", "presolve_time", "setup_time", "scaling_time",
    "analyze_time", "power_iteration_time", "solve_time", "folding_time",
]
SUMMARY_LABELS = {
    "Status": "status",
    "Iterations": "iter",
    "Primal Objective": "primal_obj",
    "Primal Residual": "res",
    "Objective Gap": "gap",
    "Residual": "res",
    "Gap": "gap",
    "Presolve Time": "presolve_time",
    "Setup Time": "setup_time",
    "Scaling Time": "scaling_time",
    "Analyze Time": "analyze_time",
    "Power Time": "power_iteration_time",
    "Solve Time": "solve_time",
    "Total Time": "total_time",
    "Folding Time": "folding_time",
    "Reduced Active Iteration Ratio": "reduced_active_iteration_ratio",
    "Reduced Average Column Ratio": "reduced_average_column_ratio",
    "Reduced Average NNZ Ratio": "reduced_average_nnz_ratio",
    "Reduced Minimum Column Ratio": "reduced_minimum_column_ratio",
    "Reduced Minimum NNZ Ratio": "reduced_minimum_nnz_ratio",
}
REQUIRED_SUMMARY_FIELDS = {
    "status", "iter", "primal_obj", "res", "gap", "total_time",
}
SUMMARY_NAMES = {"SGM10", "solved"}
FAILED_HEADER = ["index", "total", "name", "exit_code", "start", "end", "reason", "log"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HPR-LP-C on a Hans dataset and generate HPRLP_result.csv."
    )
    parser.add_argument("--data-dir", default="/data/lp_data/Hans")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--log-file", default="")
    parser.add_argument(
        "--logs-dir", default="",
        help="Per-instance log directory (default: <output directory>/logs)",
    )
    parser.add_argument("--failed-out", default="", help="Optional failed-instance CSV path")
    parser.add_argument("--solver", default="./build/solve_mps_file")
    parser.add_argument("--presolver", default=os.environ.get("HPRLP_PRESOLVER", "gpu"))
    parser.add_argument("--gpu-folding", default=os.environ.get("HPRLP_GPU_FOLDING", "true"))
    parser.add_argument(
        "--reduced-matrix",
        default=os.environ.get("HPRLP_REDUCED_MATRIX", "true"),
        help="Enable adaptive row/column reduction in manual mode (true/false)",
    )
    parser.add_argument(
        "--auto-memory-policy",
        default=os.environ.get("HPRLP_AUTO_MEMORY_POLICY", "false"),
        help="Select reduced/compression from presolved dimensions (true/false)",
    )
    parser.add_argument(
        "--devices", default=os.environ.get("HPRLP_DEVICES", ""),
        help="Comma-separated CUDA device ids; the next instance goes to the next free device",
    )
    parser.add_argument("--device", default=os.environ.get("HPRLP_DEVICE", "0"))
    parser.add_argument("--time-limit", default=os.environ.get("HPRLP_TIME_LIMIT", "1000"))
    parser.add_argument("--tol", default=os.environ.get("HPRLP_TOL", "1e-6"))
    parser.add_argument("--check-iter", default=os.environ.get("HPRLP_CHECK_ITER", "150"))
    parser.add_argument(
        "--max-iter", default=os.environ.get("HPRLP_MAX_ITER", "2147483647")
    )
    parser.add_argument(
        "--print-debug-info",
        default=os.environ.get("HPRLP_PRINT_DEBUG_INFO", "false"),
        help="Print detailed solver diagnostics (true/false)",
    )
    parser.add_argument("--pattern", default="*.mps.gz")
    parser.add_argument("--limit", type=int, default=0, help="0 means all matched files")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def timestamp():
    return datetime.now().isoformat(timespec="seconds")


def safe_name(value):
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)
    return cleaned or "run"


def default_out_dir(presolver):
    return Path("Results") / f"hans_{safe_name(presolver)}_{datetime.now():%m%d}"


def output_paths(args):
    out_dir = Path(args.out_dir) if args.out_dir else None
    if not args.out and out_dir is None:
        out_dir = default_out_dir(args.presolver)
    out = Path(args.out) if args.out else out_dir / "HPRLP_result.csv"
    log = Path(args.log_file) if args.log_file else out.parent / "HPRLP_log.txt"
    logs = Path(args.logs_dir) if args.logs_dir else out.parent / "logs"
    failed = Path(args.failed_out) if args.failed_out else None
    return out, log, logs, failed


def normalize_devices(devices, fallback):
    text = devices.strip() or str(fallback).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    try:
        normalized = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"invalid CUDA device list: {devices or fallback!r}") from error
    if not normalized:
        raise ValueError("devices must contain at least one CUDA device id")
    if any(device < 0 for device in normalized):
        raise ValueError("CUDA device ids must be nonnegative")
    if len(set(normalized)) != len(normalized):
        raise ValueError("devices must not contain duplicate CUDA device ids")
    return normalized


def instance_name(path):
    name = path.name
    lower = name.lower()
    for suffix in (".mps.gz", ".mps", ".h5"):
        if lower.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


def instance_log_name(path):
    name = instance_name(path)
    return f"{safe_name(name)}.log"


def read_instance_rows(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return [
            {field: row.get(field, "") for field in CSV_HEADER}
            for row in csv.DictReader(handle)
            if row.get("name") and row["name"] not in SUMMARY_NAMES
        ]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def ensure_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        write_rows(path, [])


def shifted_geomean(rows, field, shift=10.0):
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    if not values:
        return 0.0
    result = math.exp(sum(math.log(value + shift) for value in values) / len(values)) - shift
    return 0.0 if abs(result) < 1e-12 else result


def parse_summary_line(line, summary, in_summary):
    stripped = line.strip()
    if stripped == "=== Solution Summary ===":
        summary.clear()
        return True
    if not in_summary or ":" not in stripped:
        return in_summary
    label, value = stripped.split(":", 1)
    field = SUMMARY_LABELS.get(label)
    if field:
        summary[field] = value.strip().removesuffix(" seconds")
    return in_summary


def is_solved(status):
    return status == "OPTIMAL" or str(status).startswith("REDUCED_OPTIMAL_")


def write_summary(path):
    rows = read_instance_rows(path)
    if not rows:
        return
    summary = {field: "" for field in CSV_HEADER}
    summary["name"] = "SGM10"
    for field in ["iter"] + TIME_FIELDS:
        summary[field] = f"{shifted_geomean(rows, field):.15g}"
    solved_count = sum(is_solved(row["status"]) for row in rows)
    solved = {field: "" for field in CSV_HEADER}
    solved.update({"name": "solved", "solve_time": str(solved_count), "total_time": str(solved_count)})
    write_rows(path, rows + [summary, solved])


def append_failed(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILED_HEADER)
        if needs_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FAILED_HEADER})


def log_section(handle, title):
    handle.write("\n" + "=" * 80 + "\n" + title + "\n" + "=" * 80 + "\n")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    solver = Path(args.solver)
    out, combined_log_path, logs_dir, failed = output_paths(args)
    try:
        devices = normalize_devices(args.devices, args.device)
    except ValueError as error:
        print(f"Invalid --devices/--device value: {error}", file=sys.stderr)
        return 2

    if not data_dir.is_dir():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        return 2
    if not solver.is_file():
        print(f"Solver executable not found: {solver}", file=sys.stderr)
        return 2

    files = sorted(data_dir.glob(args.pattern))
    if args.limit > 0:
        files = files[:args.limit]
    if not files:
        print(f"No files matched {data_dir / args.pattern}", file=sys.stderr)
        return 2

    ensure_csv(out)
    old_rows = read_instance_rows(out)
    done = {row["name"] for row in old_rows} if args.resume else set()
    if not args.dry_run:
        write_rows(out, old_rows)
    logs_dir.mkdir(parents=True, exist_ok=True)
    combined_log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[hans-run] matched={len(files)} already_done={len(done)}")
    print(f"[hans-run] out={out}")
    print(f"[hans-run] log={combined_log_path}")
    print(f"[hans-run] logs={logs_dir}")
    print(
        f"[hans-run] presolver={args.presolver} gpu_folding={args.gpu_folding} "
        f"reduced_matrix={args.reduced_matrix} "
        f"auto_memory_policy={args.auto_memory_policy} "
        f"devices={','.join(map(str, devices))} "
        f"time_limit={args.time_limit} tol={args.tol}"
    )

    jobs = queue.Queue()
    for index, mps_path in enumerate(files, start=1):
        if mps_path.name in done:
            print(f"[hans-run] skip {index}/{len(files)} {mps_path.name}")
        else:
            jobs.put((index, mps_path))

    state_lock = threading.Lock()
    log_lock = threading.Lock()
    stop_event = threading.Event()
    state = {"had_failure": False}

    with combined_log_path.open("a") as combined_log:
        with log_lock:
            log_section(combined_log, f"HPRLP Hans run start {timestamp()}")
            combined_log.write(
                f"data_dir: {data_dir}\nout: {out}\nfailed_out: {failed or 'disabled'}\n"
                f"logs_dir: {logs_dir}\npresolver: {args.presolver}\n"
                f"gpu_folding: {args.gpu_folding}\n"
                f"reduced_matrix: {args.reduced_matrix}\n"
                f"auto_memory_policy: {args.auto_memory_policy}\n"
                f"devices: {','.join(map(str, devices))}\n"
                f"time_limit: {args.time_limit}\ntol: {args.tol}\n"
                f"matched: {len(files)}\nalready_done: {len(done)}\n"
            )
            combined_log.flush()

        def run_instance(index, mps_path, device):
            name = mps_path.name
            cmd = [
                str(solver), "-i", str(mps_path),
                "--device", str(device),
                "--time-limit", str(args.time_limit),
                "--tol", str(args.tol),
                "--check-iter", str(args.check_iter),
                "--presolver", str(args.presolver),
                "--gpu-folding", str(args.gpu_folding),
                "--reduced-matrix", str(args.reduced_matrix),
                "--auto-memory-policy", str(args.auto_memory_policy),
                "--print-debug-info", str(args.print_debug_info),
            ]
            if args.max_iter:
                cmd.extend(["--max-iter", str(args.max_iter)])
            command_text = " ".join(shlex.quote(part) for part in cmd)
            print(
                f"[hans-run] run {index}/{len(files)} {name} on device {device}",
                flush=True,
            )
            print(f"[hans-run] cmd {command_text}", flush=True)
            with log_lock:
                log_section(
                    combined_log,
                    f"[{index}/{len(files)}] {name} (device {device})",
                )
                combined_log.write(f"start: {timestamp()}\ncommand: {command_text}\n\n")
                combined_log.flush()
            if args.dry_run:
                return

            instance_log_path = logs_dir / instance_log_name(mps_path)
            start = timestamp()
            reason = ""
            summary = {}
            in_summary = False
            return_code = -1
            try:
                with instance_log_path.open("w", buffering=1) as instance_log:
                    proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1,
                    )
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        in_summary = parse_summary_line(line, summary, in_summary)
                        lower = line.lower()
                        if not reason and (
                            "out of memory" in lower or "cuda error 2" in lower
                        ):
                            reason = "OOM"
                        elif not reason and "terminate called" in lower:
                            reason = "ABORT"
                        instance_log.write(line)
                        with log_lock:
                            combined_log.write(f"[{name} gpu={device}] {line}")
                            combined_log.flush()
                    return_code = proc.wait()
            except Exception as error:
                reason = f"RUNNER_ERROR:{type(error).__name__}:{error}"

            end = timestamp()
            with log_lock:
                combined_log.write(f"\nend: {end} exit={return_code}\n")
                combined_log.flush()
            missing = sorted(REQUIRED_SUMMARY_FIELDS - summary.keys())
            parse_failed = return_code == 0 and bool(missing)
            if parse_failed:
                reason = "MISSING_SUMMARY_FIELDS:" + ",".join(missing)

            if return_code != 0 or parse_failed:
                with state_lock:
                    state["had_failure"] = True
                    if failed:
                        append_failed(failed, {
                            "index": index, "total": len(files), "name": name,
                            "exit_code": return_code, "start": start,
                            "end": end, "reason": reason or f"EXIT_{return_code}",
                            "log": instance_log_path,
                        })
                failure_detail = reason or f"exit={return_code}"
                print(f"[hans-run] failed {name}: {failure_detail}", file=sys.stderr)
                if args.stop_on_failure:
                    stop_event.set()
            else:
                row = {field: summary.get(field, "") for field in CSV_HEADER}
                row["name"] = name
                with state_lock:
                    old_rows.append(row)
                    write_rows(out, old_rows)
                print(f"[hans-run] done {name} on device {device}", flush=True)

        def device_worker(device):
            while not stop_event.is_set():
                try:
                    index, mps_path = jobs.get_nowait()
                except queue.Empty:
                    return
                try:
                    run_instance(index, mps_path, device)
                finally:
                    jobs.task_done()

        with ThreadPoolExecutor(
            max_workers=len(devices), thread_name_prefix="hprlp-gpu",
        ) as executor:
            futures = [executor.submit(device_worker, device) for device in devices]
            for future in futures:
                future.result()

        with log_lock:
            log_section(combined_log, f"HPRLP Hans run finished {timestamp()}")

    if not args.dry_run:
        write_summary(out)
    print(f"[hans-run] finished: {out}")
    return 1 if state["had_failure"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
