#!/usr/bin/env python3
"""Compare CPU presolver and PSLP with a cuPDLPx-aligned timing window.

The headline timing window follows cuPDLPx's PSLP integration: it starts after
the MPS file has been read into the host model representation and measures the
presolver lifecycle itself. For PSLP this includes settings allocation,
new_presolver, and run_presolver, and excludes MPS parsing, CSR preparation,
reduced-problem conversion, and Julia startup/JIT warmup.
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CpuRun:
    model: str
    run: int
    parse_seconds: float
    aligned_seconds: float
    rows: int
    cols: int
    nnz: int


@dataclass
class PslpRun:
    model: str
    run: int
    aligned_seconds: float
    stats_seconds: float
    rows: int
    cols: int
    nnz: int


def strip_model_suffix(path: Path) -> str:
    name = path.name
    for suffix in (".mps.gz", ".MPS.gz", ".mps", ".MPS"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def resolve_model(token: str, data_dir: Path) -> tuple[str, Path]:
    path = Path(token)
    if path.is_file():
        return strip_model_suffix(path), path
    for suffix in (".mps.gz", ".mps", ".MPS.gz", ".MPS"):
        candidate = data_dir / f"{token}{suffix}"
        if candidate.is_file():
            return token, candidate
    raise FileNotFoundError(f"cannot find model {token!r} under {data_dir}")


def parse_cpu_stdout(stdout: str, model: str, run: int) -> CpuRun:
    fields: dict[str, str] = {}
    for line in stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            fields[parts[0]] = parts[1]
    required = [
        "parse_seconds",
        "presolve_seconds",
        "reduced_rows",
        "reduced_cols",
        "reduced_nnz",
    ]
    missing = [key for key in required if key not in fields]
    if missing:
        raise RuntimeError(f"CPU output for {model} is missing fields: {', '.join(missing)}")
    return CpuRun(
        model=model,
        run=run,
        parse_seconds=float(fields["parse_seconds"]),
        aligned_seconds=float(fields["presolve_seconds"]),
        rows=int(fields["reduced_rows"]),
        cols=int(fields["reduced_cols"]),
        nnz=int(fields["reduced_nnz"]),
    )


def run_cpu(cpu_bin: Path, model: str, path: Path, run: int) -> CpuRun:
    if path.name.endswith(".gz"):
        gzip = subprocess.Popen(["gzip", "-cd", str(path)], stdout=subprocess.PIPE)
        try:
            completed = subprocess.run(
                [str(cpu_bin), "-"],
                stdin=gzip.stdout,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        finally:
            if gzip.stdout is not None:
                gzip.stdout.close()
        gzip_status = gzip.wait()
        if gzip_status != 0:
            raise RuntimeError(f"gzip failed for {path} with exit code {gzip_status}")
    else:
        completed = subprocess.run(
            [str(cpu_bin), str(path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"CPU presolver failed for {model} with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return parse_cpu_stdout(completed.stdout, model, run)


PSLP_JULIA = r'''
using HPRLP
using SparseArrays
const PSLP = HPRLP.PSLP

function parse_model_specs(specs::String)
    result = Tuple{String,String}[]
    for item in split(specs, ';')
        isempty(item) && continue
        parts = split(item, '=', limit=2)
        length(parts) == 2 || error("bad model spec: " * item)
        push!(result, (parts[1], parts[2]))
    end
    return result
end

function run_one(model_name::String, path::String, run_idx::Int)
    model = HPRLP.build_from_mps(path)
    A = model.A
    c = model.c
    l = model.l
    u = model.u
    lhs = model.AL
    rhs = model.AU
    m, n = size(A)
    nnz_val = nnz(A)

    # cuPDLPx receives CSR arrays before its presolve timer starts. Prepare the
    # same layout outside the aligned PSLP timing window.
    A_trans = sparse(transpose(A))
    Ax_c = Vector{Float64}(A_trans.nzval)
    Ai_c = Vector{Cint}(A_trans.rowval .- 1)
    Ap_c = Vector{Cint}(A_trans.colptr .- 1)

    GC.gc()
    t0 = time_ns()
    settings = PSLP.Settings(verbose=false)
    stgs_ptr = PSLP._allocate_settings_ptr(settings)
    ptr = GC.@preserve Ax_c Ai_c Ap_c lhs rhs l u c begin
        ccall(
            (:new_presolver, PSLP.LIB_PATH),
            Ptr{PSLP.PresolverStruct},
            (
                Ptr{Float64},
                Ptr{Cint},
                Ptr{Cint},
                Csize_t,
                Csize_t,
                Csize_t,
                Ptr{Float64},
                Ptr{Float64},
                Ptr{Float64},
                Ptr{Float64},
                Ptr{Float64},
                Ptr{PSLP.Settings},
            ),
            Ax_c,
            Ai_c,
            Ap_c,
            Csize_t(m),
            Csize_t(n),
            Csize_t(nnz_val),
            lhs,
            rhs,
            l,
            u,
            c,
            stgs_ptr,
        )
    end
    if ptr == C_NULL
        ccall((:free_settings, PSLP.LIB_PATH), Cvoid, (Ptr{PSLP.Settings},), stgs_ptr)
        error("PSLP new_presolver returned NULL for " * model_name)
    end
    info = PSLP.PresolverModel(ptr, stgs_ptr)
    status = PSLP._with_silent_stdio() do
        ccall((:run_presolver, PSLP.LIB_PATH), Cint, (Ptr{PSLP.PresolverStruct},), info.ptr)
    end
    t1 = time_ns()

    presolver_data = unsafe_load(info.ptr)
    stats_seconds = NaN
    if presolver_data.stats != C_NULL
        stats = unsafe_load(Ptr{PSLP.PresolveStats}(presolver_data.stats))
        stats_seconds = stats.time_init + stats.time_presolve
    end
    if presolver_data.reduced_prob == C_NULL
        PSLP.free_presolver_wrapper(info)
        error("PSLP reduced_prob is NULL for " * model_name)
    end
    red_prob = unsafe_load(presolver_data.reduced_prob)
    println(
        "PSLP_UNIFIED\t",
        model_name, "\t",
        run_idx, "\t",
        (t1 - t0) / 1.0e9, "\t",
        stats_seconds, "\t",
        Int(red_prob.m), "\t",
        Int(red_prob.n), "\t",
        Int(red_prob.nnz), "\t",
        status,
    )
    PSLP.free_presolver_wrapper(info)
end

warmup = get(ENV, "UNIFIED_PSLP_WARMUP", "")
if !isempty(warmup)
    run_one("__warmup__", warmup, 0)
end

runs = parse(Int, ENV["UNIFIED_RUNS"])
for (model_name, path) in parse_model_specs(ENV["UNIFIED_MODELS"])
    for run_idx in 1:runs
        run_one(model_name, path, run_idx)
    end
end
'''


def parse_pslp_stdout(stdout: str) -> list[PslpRun]:
    runs: list[PslpRun] = []
    for line in stdout.splitlines():
        if not line.startswith("PSLP_UNIFIED\t"):
            continue
        _, model, run, elapsed, stats, rows, cols, nnz, _status = line.split("\t")
        if model == "__warmup__":
            continue
        runs.append(
            PslpRun(
                model=model,
                run=int(run),
                aligned_seconds=float(elapsed),
                stats_seconds=float(stats),
                rows=int(rows),
                cols=int(cols),
                nnz=int(nnz),
            )
        )
    return runs


def run_pslp(
    julia: Path,
    hprlp_project: Path,
    model_paths: list[tuple[str, Path]],
    runs: int,
    warmup: Path | None,
) -> list[PslpRun]:
    env = os.environ.copy()
    env["UNIFIED_RUNS"] = str(runs)
    env["UNIFIED_MODELS"] = ";".join(f"{name}={path}" for name, path in model_paths)
    if warmup is not None:
        env["UNIFIED_PSLP_WARMUP"] = str(warmup)
    completed = subprocess.run(
        [str(julia), f"--project={hprlp_project}", "-e", PSLP_JULIA],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"PSLP benchmark failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    parsed = parse_pslp_stdout(completed.stdout)
    expected = len(model_paths) * runs
    if len(parsed) != expected:
        raise RuntimeError(
            f"PSLP output produced {len(parsed)} result rows, expected {expected}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return parsed


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def group_by_model(items):
    grouped = {}
    for item in items:
        grouped.setdefault(item.model, []).append(item)
    return grouped


def print_summary(cpu_runs: list[CpuRun], pslp_runs: list[PslpRun]) -> None:
    cpu_by_model = group_by_model(cpu_runs)
    pslp_by_model = group_by_model(pslp_runs)
    print(
        "model,cpu_cupdlpx_aligned_seconds,pslp_cupdlpx_aligned_seconds,"
        "cpu_over_pslp,pslp_stats_seconds,cpu_rows,cpu_cols,cpu_nnz,"
        "pslp_rows,pslp_cols,pslp_nnz,cpu_parse_seconds"
    )
    for model in cpu_by_model:
        cpu_items = cpu_by_model[model]
        pslp_items = pslp_by_model[model]
        cpu_time = mean([item.aligned_seconds for item in cpu_items])
        pslp_time = mean([item.aligned_seconds for item in pslp_items])
        pslp_stats = mean([item.stats_seconds for item in pslp_items])
        cpu_last = cpu_items[-1]
        pslp_last = pslp_items[-1]
        cpu_parse = mean([item.parse_seconds for item in cpu_items])
        print(
            f"{model},"
            f"{cpu_time:.6f},"
            f"{pslp_time:.6f},"
            f"{cpu_time / pslp_time:.3f},"
            f"{pslp_stats:.6f},"
            f"{cpu_last.rows},"
            f"{cpu_last.cols},"
            f"{cpu_last.nnz},"
            f"{pslp_last.rows},"
            f"{pslp_last.cols},"
            f"{pslp_last.nnz},"
            f"{cpu_parse:.6f}"
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Compare CPU presolver and PSLP with cuPDLPx-aligned presolve timing."
    )
    parser.add_argument("models", nargs="+", help="Model names under --data-dir or MPS paths.")
    parser.add_argument("--runs", type=int, default=1, help="Runs per model.")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--cpu-bin", type=Path, default=Path("build/cpu_presolve_mps"))
    parser.add_argument(
        "--hprlp-project",
        type=Path,
        required=True,
    )
    parser.add_argument("--julia", type=Path, required=True)
    parser.add_argument(
        "--pslp-warmup",
        type=Path,
        default=None,
        help="Optional warmup model used to keep Julia JIT out of PSLP timings.",
    )
    args = parser.parse_args(argv)

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if not args.cpu_bin.is_file():
        raise FileNotFoundError(f"CPU binary not found: {args.cpu_bin}")
    if not args.julia.is_file():
        raise FileNotFoundError(f"Julia binary not found: {args.julia}")
    if not args.hprlp_project.is_dir():
        raise FileNotFoundError(f"HPRLP project not found: {args.hprlp_project}")

    model_paths = [resolve_model(model, args.data_dir) for model in args.models]
    warmup = args.pslp_warmup

    cpu_runs: list[CpuRun] = []
    for name, path in model_paths:
        for run in range(1, args.runs + 1):
            cpu_runs.append(run_cpu(args.cpu_bin, name, path, run))

    pslp_runs = run_pslp(args.julia, args.hprlp_project, model_paths, args.runs, warmup)
    print_summary(cpu_runs, pslp_runs)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
