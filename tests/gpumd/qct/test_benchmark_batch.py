import importlib.util
from pathlib import Path


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
