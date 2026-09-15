#!/usr/bin/env python3

import importlib.util
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_hans_dataset", REPO_ROOT / "scripts" / "run_hans_dataset.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def fake_result():
    timing = SimpleNamespace(
        total_time=8.0,
        presolve_time=1.0,
        setup_time=2.0,
        scaling_time=3.0,
        analyze_time=4.0,
        power_iteration_time=5.0,
        solve_time=6.0,
    )
    return SimpleNamespace(
        timing=timing,
        folding_time=0.5,
        residuals=1e-7,
        primal_obj=12.0,
        gap=2e-8,
        status="OPTIMAL",
        iter=800,
        iter4=400,
        iter6=600,
        iter8=800,
        time4=4.0,
        time6=6.0,
        time8=8.0,
        reduced_active_iteration_ratio=0.8,
        reduced_average_column_ratio=0.7,
        reduced_average_nnz_ratio=0.6,
        reduced_minimum_column_ratio=0.5,
        reduced_minimum_nnz_ratio=0.4,
    )


def test_native_result_row_is_complete():
    row = RUNNER.native_result_row(Path("example.mps.gz"), fake_result())
    assert list(row) == RUNNER.CSV_HEADER
    assert row["name"] == "example"
    assert all(value != "" for value in row.values())


def test_missing_summary_input_stays_missing():
    assert RUNNER.shifted_geomean([], "solve_time") is None


def test_native_parameters_match_configuration():
    class Parameters:
        pass

    hprlp = SimpleNamespace(Parameters=Parameters)
    configuration = {
        "max_iter": 123,
        "stop_tol": 1e-6,
        "time_limit": 1000.0,
        "check_iter": 150,
        "presolver": "gpu",
        "gpu_folding": True,
        "use_reduced_matrix": True,
        "auto_reduced_compression_policy": False,
        "print_debug_info": False,
        "specified_parameter_mask": 7,
    }
    parameters = RUNNER.native_parameters(hprlp, configuration, device=2)
    assert parameters.device_number == 2
    assert parameters.presolver == 1
    assert parameters.use_presolve is True
    assert parameters.enable_gpu_folding is True
    assert parameters.specified_parameter_mask == 7


def test_spawned_native_worker_writes_complete_csv():
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        fake_package = temporary / "python" / "hprlp"
        fake_package.mkdir(parents=True)
        (fake_package / "_hprlp_core.py").write_text(
            "class Results:\n"
            "    folding_time = 0.0\n"
            "    reduced_active_iteration_ratio = 0.0\n"
            "    reduced_average_column_ratio = 1.0\n"
            "    reduced_average_nnz_ratio = 1.0\n"
            "    reduced_minimum_column_ratio = 1.0\n"
            "    reduced_minimum_nnz_ratio = 1.0\n"
        )
        (fake_package / "__init__.py").write_text(
            "from types import SimpleNamespace\n"
            "from . import _hprlp_core\n"
            "__version__ = 'test'\n"
            "class Parameters:\n"
            "    pass\n"
            "class Model:\n"
            "    read_time = 0.25\n"
            "    @classmethod\n"
            "    def from_mps(cls, path):\n"
            "        return cls()\n"
            "    def solve(self, parameters, copy_solution=True):\n"
            "        timing = SimpleNamespace(total_time=8.0, "
            "presolve_time=1.0, setup_time=2.0, scaling_time=3.0, "
            "analyze_time=4.0, power_iteration_time=5.0, solve_time=6.0)\n"
            "        return SimpleNamespace(timing=timing, folding_time=0.5, "
            "residuals=1e-7, primal_obj=12.0, gap=2e-8, status='OPTIMAL', "
            "iter=800, iter4=400, iter6=600, iter8=800, time4=4.0, "
            "time6=6.0, time8=8.0, reduced_active_iteration_ratio=0.8, "
            "reduced_average_column_ratio=0.7, reduced_average_nnz_ratio=0.6, "
            "reduced_minimum_column_ratio=0.5, reduced_minimum_nnz_ratio=0.4)\n"
            "    def free(self):\n"
            "        pass\n"
        )
        data_directory = temporary / "data"
        data_directory.mkdir()
        (data_directory / "sample.mps.gz").touch()
        output_directory = temporary / "output"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(temporary / "python")
        completed = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_hans_dataset.py"),
                "--data-dir", str(data_directory),
                "--out-dir", str(output_directory),
                "--no-resume",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        with (output_directory / "HPRLP_result.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert rows[0]["name"] == "sample"
        assert all(rows[0][field] != "" for field in RUNNER.CSV_HEADER)


if __name__ == "__main__":
    test_native_result_row_is_complete()
    test_missing_summary_input_stays_missing()
    test_native_parameters_match_configuration()
    test_spawned_native_worker_writes_complete_csv()
    print("run_hans_dataset_test: PASS")
