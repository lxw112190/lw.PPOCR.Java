import copy
import importlib.util
import pathlib
import unittest
import contextlib
import io
import json
import tempfile

spec = importlib.util.spec_from_file_location('comparison', pathlib.Path(__file__).with_name('summarize-csharp-comparison.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ContractTest(unittest.TestCase):
    def setUp(self):
        self.java = dict(schema=1, runtime='java', variant='medium', cpu=8, line_workers=4,
                         det_threads=4, rec_threads=2, warmup=5, iterations=10,
                         width=500, height=500, det_width=512, det_height=512,
                         settings={'det_limit': 960}, bgr_sha256='a'*64, fma=False,
                         mean_ms=30., median_ms=29., p95_ms=33.,
                         lines=[dict(text='test', rotation=0, rec_width=192, box=[0.]*8)])
        self.csharp = dict(copy.deepcopy(self.java), runtime='csharp', width_hook=True, fp32=True, mean_ms=20.)

    def test_matching_contract(self):
        result = module.compare(self.java, self.csharp)
        self.assertEqual(1.5, result['java_over_csharp'])
        self.assertTrue(result['identical_ordered_crops'])
        self.assertEqual(1, result['exact_text_lines'])

    def test_features_are_reported_without_changing_input_contract(self):
        self.java['features'] = {'parallel_prepared_epilogue': True}
        self.assertEqual(self.java['features'], module.compare(self.java, self.csharp)['java_features'])

    def test_fma_only_run_has_no_invented_default_baseline(self):
        java = dict(self.java, fma=True, replica=1)
        peer = dict(self.csharp, replica=1)
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / 'medium-java-fma-r1.json').write_text(json.dumps(java), encoding='utf-8')
            (root / 'medium-csharp-r1.json').write_text(json.dumps(peer), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                module.main(root)
            summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))[0]
            self.assertNotIn('fma_reduction_percent', summary)
            self.assertEqual(1.5, summary['java_over_csharp'])

    def test_each_contract_mismatch_fails(self):
        for key in ('cpu', 'line_workers', 'det_threads', 'rec_threads', 'warmup',
                    'iterations', 'width', 'height', 'det_width', 'det_height', 'settings', 'variant', 'schema', 'bgr_sha256'):
            peer = copy.deepcopy(self.csharp)
            peer[key] = 'different'
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.compare(self.java, peer)

    def test_changed_geometry_is_not_equal_work(self):
        self.csharp['lines'][0]['box'][0] = .125
        result = module.compare(self.java, self.csharp)
        self.assertFalse(result['identical_ordered_crops'])
        self.assertEqual(.125, result['max_ordered_box_coordinate_error'])

    def test_text_and_width_differences_remain_visible(self):
        self.csharp['lines'][0].update(text='other', rec_width=320, rotation=180)
        result = module.compare(self.java, self.csharp)
        self.assertEqual(0, result['exact_text_lines'])
        self.assertFalse(result['ordered_width_parity'])
        self.assertFalse(result['rotation_parity'])

    def test_invalid_output_fails(self):
        for change in ({'lines': []}, {'mean_ms': float('nan')}, {'p95_ms': 0},
                       {'width_hook': False}, {'fp32': False}, {'runtime': 'java'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                module.compare(self.java, dict(self.csharp, **change))
        self.csharp['lines'][0]['rec_width'] = 256
        with self.assertRaises(ValueError):
            module.compare(self.java, self.csharp)


if __name__ == '__main__':
    unittest.main()
