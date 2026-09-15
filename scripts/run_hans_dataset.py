#!/usr/bin/env python3
"""Run HPR-LP-C over Hans and maintain a resumable paper-style CSV."""

import argparse
import ctypes
import csv
import math
import multiprocessing
import os
import queue
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


DEFAULT_HPRLP_ENVIRONMENT = (
    ("HPRLP_ENABLE_COMPRESSIBLE_MEMORY", "1"),
    ("HPRLP_USE_ROW_REDUCTION", "1"),
    ("HPRLP_USE_ROW_COMPRESSED_AUTOTUNE", "1"),
    ("HPRLP_USE_REDUCED_COMPRESSED_AUTOTUNE", "1"),
    ("HPRLP_DEFER_REDUCED_EMPTY_ROWS_TO_CHECK", "1"),
    ("HPRLP_USE_REDUCED_NONEMPTY_CUSPARSE", "1"),
    ("HPRLP_REDUCED_RESET_MASK_ON_RESTART", "1"),
    ("HPRLP_REDUCED_RESTART_MASK_MIN_RECOVERY", "0.25"),
    ("HPRLP_REDUCED_RESTART_MASK_MIN_SAVED_COLUMNS", "25000"),
    ("HPRLP_REDUCED_RESTART_MASK_MIN_CURRENT_COLUMNS", "95000"),
)
for environment_name, default_value in DEFAULT_HPRLP_ENVIRONMENT:
    os.environ.setdefault(environment_name, default_value)


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
SUMMARY_NAMES = {"SGM10", "solved"}
FAILED_HEADER = ["index", "total", "name", "exit_code", "start", "end", "reason", "log"]
PARAMETER_OPTIONS = (
    ("--max-iter", "HPRLP_MAX_ITER", 0),
    ("--tol", "HPRLP_TOL", 1),
    ("--time-limit", "HPRLP_TIME_LIMIT", 2),
    ("--check-iter", "HPRLP_CHECK_ITER", 4),
    ("--presolver", "HPRLP_PRESOLVER", 19),
    ("--gpu-folding", "HPRLP_GPU_FOLDING", 20),
    ("--reduced-matrix", "HPRLP_REDUCED_MATRIX", 21),
    ("--auto-memory-policy", "HPRLP_AUTO_MEMORY_POLICY", 22),
    ("--print-debug-info", "HPRLP_PRINT_DEBUG_INFO", 23),
)


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Run HPR-LP-C on a Hans dataset and generate HPRLP_result.csv."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing the Hans *.mps.gz instances",
    )
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--log-file", default="")
    parser.add_argument(
        "--logs-dir", default="",
        help="Per-instance log directory (default: <output directory>/logs)",
    )
    parser.add_argument("--failed-out", default="", help="Optional failed-instance CSV path")
    parser.add_argument(
        "--solver", default="", help=argparse.SUPPRESS,
    )
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
    args = parser.parse_args(argv)
    args.explicit_options = {
        token.split("=", 1)[0] for token in argv if token.startswith("--")
    }
    return args


def parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"expected true or false, got {value!r}")


def native_configuration(args, forward_device):
    presolver = str(args.presolver).strip().lower()
    if presolver not in {"pslp", "gpu", "none"}:
        raise ValueError("--presolver must be pslp, gpu, or none")
    specified_parameter_mask = (1 << 3) if forward_device else 0
    for option, environment_name, bit in PARAMETER_OPTIONS:
        if option in args.explicit_options or environment_name in os.environ:
            specified_parameter_mask |= 1 << bit
    return {
        "max_iter": int(args.max_iter),
        "stop_tol": float(args.tol),
        "time_limit": float(args.time_limit),
        "check_iter": int(args.check_iter),
        "presolver": presolver,
        "gpu_folding": parse_bool(args.gpu_folding),
        "use_reduced_matrix": parse_bool(args.reduced_matrix),
        "auto_reduced_compression_policy": parse_bool(args.auto_memory_policy),
        "print_debug_info": parse_bool(args.print_debug_info),
        "specified_parameter_mask": specified_parameter_mask,
    }


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
    for suffix in (".mps.gz", ".hdf5", ".mps", ".h5"):
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
        return None
    result = math.exp(sum(math.log(value + shift) for value in values) / len(values)) - shift
    return 0.0 if abs(result) < 1e-12 else result


def native_result_row(path, result):
    timing = result.timing
    return {
        "name": instance_name(Path(path)),
        "iter": result.iter,
        "total_time": timing.total_time,
        "presolve_time": timing.presolve_time,
        "setup_time": timing.setup_time,
        "scaling_time": timing.scaling_time,
        "analyze_time": timing.analyze_time,
        "power_iteration_time": timing.power_iteration_time,
        "solve_time": timing.solve_time,
        "folding_time": result.folding_time,
        "res": result.residuals,
        "primal_obj": result.primal_obj,
        "gap": result.gap,
        "status": result.status,
        "iter_4": result.iter4,
        "time_4": result.time4,
        "iter_6": result.iter6,
        "time_6": result.time6,
        "iter_8": result.iter8,
        "time_8": result.time8,
        "reduced_active_iteration_ratio": result.reduced_active_iteration_ratio,
        "reduced_average_column_ratio": result.reduced_average_column_ratio,
        "reduced_average_nnz_ratio": result.reduced_average_nnz_ratio,
        "reduced_minimum_column_ratio": result.reduced_minimum_column_ratio,
        "reduced_minimum_nnz_ratio": result.reduced_minimum_nnz_ratio,
    }


def native_parameters(hprlp, configuration, device):
    parameters = hprlp.Parameters()
    parameters.max_iter = configuration["max_iter"]
    parameters.stop_tol = configuration["stop_tol"]
    parameters.time_limit = configuration["time_limit"]
    parameters.device_number = device
    parameters.check_iter = configuration["check_iter"]
    parameters.use_presolve = configuration["presolver"] != "none"
    parameters.presolver = {"pslp": 0, "gpu": 1, "none": 2}[
        configuration["presolver"]
    ]
    parameters.enable_gpu_folding = configuration["gpu_folding"]
    parameters.use_reduced_matrix = configuration["use_reduced_matrix"]
    parameters.auto_reduced_compression_policy = configuration[
        "auto_reduced_compression_policy"
    ]
    parameters.print_debug_info = configuration["print_debug_info"]
    parameters.specified_parameter_mask = configuration["specified_parameter_mask"]
    return parameters


@contextmanager
def redirect_native_output(path):
    """Redirect Python, C, and C++ process output into one instance log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", buffering=1) as stream:
        sys.stdout.flush()
        sys.stderr.flush()
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        try:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            yield stream
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            try:
                ctypes.CDLL(None).fflush(None)
            except (AttributeError, OSError):
                pass
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def native_binding_preflight():
    import hprlp
    from hprlp import _hprlp_core

    result = _hprlp_core.Results()
    required_fields = (
        "folding_time",
        "reduced_active_iteration_ratio",
        "reduced_average_column_ratio",
        "reduced_average_nnz_ratio",
        "reduced_minimum_column_ratio",
        "reduced_minimum_nnz_ratio",
    )
    missing = [field for field in required_fields if not hasattr(result, field)]
    if missing:
        raise RuntimeError(
            "installed hprlp binding is missing native result fields: "
            + ", ".join(missing)
            + "; reinstall it with `python -m pip install --force-reinstall "
              "./bindings/python`"
        )
    return hprlp.__version__


def solve_native_instance(index, total, path, log_path, device, configuration):
    start = timestamp()
    model = None
    try:
        with redirect_native_output(log_path):
            import hprlp

            os.environ["HPRLP_PRINT_DEBUG_INFO"] = (
                "1" if configuration["print_debug_info"] else "0"
            )
            print(f"Reading file {path}", flush=True)
            model = hprlp.Model.from_mps(path)
            print(f"Reading time: {model.read_time:.2f}s", flush=True)
            parameters = native_parameters(hprlp, configuration, device)
            result = model.solve(parameters, copy_solution=False)
            row = native_result_row(path, result)
        return {
            "index": index,
            "total": total,
            "device": device,
            "name": Path(path).name,
            "row": row,
            "status": result.status,
            "success": True,
            "reason": "",
            "start": start,
            "end": timestamp(),
            "log": log_path,
        }
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        lower = message.lower()
        reason = "OOM" if "out of memory" in lower or "cuda error 2" in lower else message
        with Path(log_path).open("a") as stream:
            stream.write(f"\nRUNNER_ERROR: {message}\n")
        return {
            "index": index,
            "total": total,
            "device": device,
            "name": Path(path).name,
            "row": None,
            "status": "SOLVE_ERROR",
            "success": False,
            "reason": reason,
            "start": start,
            "end": timestamp(),
            "log": log_path,
        }
    finally:
        if model is not None:
            model.free()


def is_solved(status):
    return status == "OPTIMAL" or str(status).startswith("REDUCED_OPTIMAL_")


def write_summary(path):
    rows = read_instance_rows(path)
    if not rows:
        return
    summary = {field: "" for field in CSV_HEADER}
    summary["name"] = "SGM10"
    for field in ["iter"] + TIME_FIELDS:
        value = shifted_geomean(rows, field)
        if value is not None:
            summary[field] = f"{value:.15g}"
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
    out, combined_log_path, logs_dir, failed = output_paths(args)
    forward_device = bool(
        {"--devices", "--device"} & args.explicit_options
        or "HPRLP_DEVICES" in os.environ
        or "HPRLP_DEVICE" in os.environ
    )
    try:
        devices = normalize_devices(args.devices, args.device)
        configuration = native_configuration(args, forward_device)
    except ValueError as error:
        print(f"Invalid runner option: {error}", file=sys.stderr)
        return 2

    if not data_dir.is_dir():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
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
    print(f"[hans-run] devices={','.join(map(str, devices))}")
    print("[hans-run] backend=native Python binding")
    if args.solver:
        print("[hans-run] warning: --solver is deprecated and ignored")
    print(
        "[hans-run] parameters="
        + ",".join(f"{key}={value}" for key, value in configuration.items())
    )

    jobs = queue.Queue()
    for index, mps_path in enumerate(files, start=1):
        if instance_name(mps_path) in done:
            print(f"[hans-run] skip {index}/{len(files)} {mps_path.name}")
        else:
            jobs.put((index, mps_path))

    state_lock = threading.Lock()
    log_lock = threading.Lock()
    stop_event = threading.Event()
    state = {"had_failure": False}
    native_executors = {}

    if not args.dry_run and not jobs.empty():
        context = multiprocessing.get_context("spawn")
        try:
            for device in devices:
                native_executors[device] = ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=context,
                )
            checks = [
                executor.submit(native_binding_preflight)
                for executor in native_executors.values()
            ]
            versions = {check.result() for check in checks}
            print(f"[hans-run] hprlp={','.join(sorted(versions))}")
        except Exception as error:
            for executor in native_executors.values():
                executor.shutdown(wait=True, cancel_futures=True)
            print(
                "Unable to load the native Python binding in dataset workers: "
                f"{error}\nInstall it with: "
                "python -m pip install --force-reinstall ./bindings/python",
                file=sys.stderr,
            )
            return 2

    try:
        with combined_log_path.open("a") as combined_log:
            with log_lock:
                log_section(combined_log, f"HPRLP Hans run start {timestamp()}")
                combined_log.write(
                    f"data_dir: {data_dir}\nout: {out}\n"
                    f"failed_out: {failed or 'disabled'}\nlogs_dir: {logs_dir}\n"
                    f"devices: {','.join(map(str, devices))}\n"
                    "backend: native Python binding\n"
                    f"parameters: {configuration}\n"
                    f"matched: {len(files)}\nalready_done: {len(done)}\n"
                )
                combined_log.flush()

            def record_outcome(outcome):
                name = outcome["name"]
                log_path = Path(outcome["log"])
                with log_lock:
                    log_section(
                        combined_log,
                        f"[{outcome['index']}/{outcome['total']}] {name} "
                        f"(device {outcome['device']})",
                    )
                    combined_log.write(f"start: {outcome['start']}\n")
                    if log_path.exists():
                        with log_path.open(errors="replace") as instance_log:
                            for line in instance_log:
                                combined_log.write(
                                    f"[{name} gpu={outcome['device']}] {line}"
                                )
                    combined_log.write(
                        f"\nend: {outcome['end']} status={outcome['status']}\n"
                    )
                    combined_log.flush()

                if not outcome["success"]:
                    with state_lock:
                        state["had_failure"] = True
                        if failed:
                            append_failed(failed, {
                                "index": outcome["index"],
                                "total": outcome["total"],
                                "name": name,
                                "exit_code": 1,
                                "start": outcome["start"],
                                "end": outcome["end"],
                                "reason": outcome["reason"],
                                "log": log_path,
                            })
                    print(
                        f"[hans-run] failed {name}: {outcome['reason']}",
                        file=sys.stderr,
                    )
                    if args.stop_on_failure:
                        stop_event.set()
                    return

                with state_lock:
                    old_rows.append(outcome["row"])
                    write_rows(out, old_rows)
                print(
                    f"[hans-run] done {name} on device {outcome['device']}",
                    flush=True,
                )

            def run_instance(index, mps_path, device):
                name = mps_path.name
                print(
                    f"[hans-run] run {index}/{len(files)} {name} "
                    f"on device {device}",
                    flush=True,
                )
                if args.dry_run:
                    return
                log_path = logs_dir / instance_log_name(mps_path)
                try:
                    outcome = native_executors[device].submit(
                        solve_native_instance,
                        index,
                        len(files),
                        str(mps_path),
                        str(log_path),
                        device,
                        configuration,
                    ).result()
                except Exception as error:
                    outcome = {
                        "index": index,
                        "total": len(files),
                        "device": device,
                        "name": name,
                        "row": None,
                        "status": "WORKER_ERROR",
                        "success": False,
                        "reason": f"WORKER_ERROR:{type(error).__name__}:{error}",
                        "start": timestamp(),
                        "end": timestamp(),
                        "log": str(log_path),
                    }
                record_outcome(outcome)

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
    finally:
        for executor in native_executors.values():
            executor.shutdown(wait=True, cancel_futures=True)

    if not args.dry_run:
        write_summary(out)
    print(f"[hans-run] finished: {out}")
    return 1 if state["had_failure"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
