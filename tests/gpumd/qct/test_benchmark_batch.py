import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_batch", REPO_ROOT / "tools/qct/benchmark_batch.py"
)
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def benchmark_row(replicas, throughput, return_code=0, memory_safe=True):
    return {
        "replicas": replicas,
        "run_replica_steps_per_s": throughput,
        "return_code": return_code,
        "memory_safe_at_85_percent": memory_safe,
    }


def test_recommend_smallest_batch_near_peak():
    rows = [
        benchmark_row(1024, 4.6e6),
        benchmark_row(8192, 18.3e6),
        benchmark_row(65536, 22.0e6),
        benchmark_row(131072, 22.7e6),
    ]
    recommended, peak = BENCHMARK.recommend_candidate(rows, 0.95)
    assert peak["replicas"] == 131072
    assert recommended["replicas"] == 65536


def test_recommend_ignores_failed_or_memory_unsafe_batches():
    rows = [
        benchmark_row(1024, 4.0e6),
        benchmark_row(2048, 8.0e6, return_code=1),
        benchmark_row(4096, 12.0e6, memory_safe=False),
    ]
    recommended, peak = BENCHMARK.recommend_candidate(rows, 0.95)
    assert peak == recommended
    assert recommended["replicas"] == 1024


def test_recommend_rejects_unknown_memory_status():
    rows = [benchmark_row(1024, 4.0e6, memory_safe="unknown")]
    with pytest.raises(ValueError, match="No successful and memory-safe"):
        BENCHMARK.recommend_candidate(rows)


def test_run_one_fake_executable_uses_fallback_and_cleans_workspace(tmp_path):
    (tmp_path / "run.in").write_text("ensemble qct replicas 1\nrun 2\n", encoding="utf-8")

    row = BENCHMARK.run_one(2, tmp_path, "/bin/true", 3, False, None)

    assert row["return_code"] == 0
    assert row["run_replica_steps_per_s"] > 0
    assert row["memory_safe_at_85_percent"] == "unknown"
