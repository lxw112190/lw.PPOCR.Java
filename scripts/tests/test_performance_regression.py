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


class StemTimingTests(unittest.TestCase):
    def fixture(self, median=1.5):
        return dict(benchmark="det-stride-two-conv", backend="vector", workload="stem",
                    input_width=320, input_height=320, input_channels=3,
                    output_width=160, output_height=160, output_channels=16,
                    requested_warmup=100, warmup=750, warmup_min_ms=1000,
                    warmup_elapsed_ms=1001.0, iterations=30, median_ms=median,
                    checksum="10973022362502076197", samples_ms=[median]*30)

    def check(self, result):
        CHECK.check_stem_contract(result, "stem.json")
        with contextlib.redirect_stdout(io.StringIO()):
            CHECK.require_lt(result, "median_ms", 5.0, "stem.json")

    def test_steady_protocol_and_unchanged_strict_limit(self):
        self.check(self.fixture())
        self.check(self.fixture(4.999))
        for median in (5.0, 8.525):
            with self.subTest(median=median), self.assertRaises(SystemExit):
                self.check(self.fixture(median))

    def test_short_or_incomplete_warmup_is_not_accepted(self):
        for updates in (dict(requested_warmup=10), dict(warmup=99),
                        dict(requested_warmup=800), dict(warmup_min_ms=0),
                        dict(warmup_elapsed_ms=999.9), dict(iterations=10),
                        dict(warmup=float("nan")), dict(warmup_min_ms=float("inf")),
                        dict(warmup_elapsed_ms=float("nan")), dict(requested_warmup=True)):
            with self.subTest(updates=updates), self.assertRaises(SystemExit):
                self.check(dict(self.fixture(), **updates))
        for key in ("requested_warmup", "warmup_elapsed_ms", "samples_ms"):
            result=self.fixture(); del result[key]
            with self.subTest(key=key), self.assertRaises(SystemExit):
                self.check(result)

    def test_fixture_backend_and_shape_cannot_be_reduced(self):
        for updates in (dict(input_width=160), dict(input_channels=1),
                        dict(output_channels=8), dict(workload="downsample"),
                        dict(backend="scalar"), dict(checksum="wrong")):
            with self.subTest(updates=updates), self.assertRaises(SystemExit):
                self.check(dict(self.fixture(), **updates))

    def test_all_chronological_samples_must_back_reported_median(self):
        for samples in ([1.5]*29, [1.5]*31, [1.0]*30,
                        [True]+[1.5]*29, [0]+[1.5]*29,
                        [float("nan")]+[1.5]*29, [float("inf")]+[1.5]*29):
            with self.subTest(samples=samples), self.assertRaises(SystemExit):
                self.check(dict(self.fixture(), samples_ms=samples))
        # A minority of fast samples must never hide a slow upper median.
        with self.assertRaises(SystemExit):
            self.check(dict(self.fixture(), samples_ms=[1.5]*12+[8.525]*18))


if __name__ == "__main__":
    unittest.main()
