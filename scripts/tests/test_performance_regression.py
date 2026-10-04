"""Keep the explicit two-CPU allocation guard strict without platform dependencies."""

import contextlib
import importlib.util
import io
import pathlib
import unittest


PATH = pathlib.Path(__file__).resolve().parents[1] / "check-performance-regression.py"
SPEC = importlib.util.spec_from_file_location("performance_regression", PATH)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class LowCpuAllocationTests(unittest.TestCase):
    def fixture(self):
        return dict(available_processors=2, lines=16, detector_limit_side=320,
                    warmup=10, iterations=5, backend="vector", parallelism_policy="auto",
                    allocated_bytes_per_ocr=12_000, gc_count_delta=0)

    def check(self, result):
        with contextlib.redirect_stdout(io.StringIO()):
            CHECK.check_low_cpu_allocation(result, "test.json")

    def test_passing_and_exact_limit(self):
        self.check(self.fixture())
        self.check(dict(self.fixture(), allocated_bytes_per_ocr=1_000_000, gc_count_delta=1))

    def test_observed_boxing_and_gc_fail(self):
        for updates in (dict(allocated_bytes_per_ocr=66_428_400),
                        dict(allocated_bytes_per_ocr=1_000_001), dict(gc_count_delta=2)):
            with self.subTest(updates=updates), self.assertRaises(SystemExit):
                self.check(dict(self.fixture(), **updates))

    def test_missing_or_unsupported_metrics_fail(self):
        for key in ("allocated_bytes_per_ocr", "gc_count_delta"):
            result = self.fixture()
            del result[key]
            with self.subTest(key=key), self.assertRaises(SystemExit):
                self.check(result)
            with self.assertRaises(SystemExit):
                self.check(dict(self.fixture(), **{key: -1}))

    def test_wrong_workload_cannot_hide_regression(self):
        for updates in (dict(available_processors=8), dict(lines=0), dict(backend="scalar"),
                        dict(detector_limit_side=960), dict(iterations=1),
                        dict(warmup=100), dict(parallelism_policy="manual")):
            with self.subTest(updates=updates), self.assertRaises(SystemExit):
                self.check(dict(self.fixture(), **updates))


if __name__ == "__main__":
    unittest.main()
