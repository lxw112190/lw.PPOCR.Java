"""Validate workload contracts; never confuse a same-image comparison with CER."""
import collections
import json
import math
import pathlib
import sys


def compare(java, csharp):
    if java['runtime'] != 'java' or csharp['runtime'] != 'csharp':
        raise ValueError('invalid runtime pairing')
    for key in ('schema', 'variant', 'cpu', 'line_workers', 'det_threads', 'rec_threads',
                'warmup', 'iterations', 'width', 'height', 'det_width', 'det_height', 'settings', 'bgr_sha256'):
        if java[key] != csharp[key]:
            raise ValueError(f'comparison contract mismatch: {key}')
    if not csharp['width_hook'] or not csharp['fp32']:
        raise ValueError('C# must use FP32 and the pinned REC bucket hook')
    for result in (java, csharp):
        if not result['lines']:
            raise ValueError('empty OCR output')
        for key in ('mean_ms', 'median_ms', 'p95_ms'):
            if not math.isfinite(result[key]) or result[key] <= 0:
                raise ValueError('invalid measurement')
        for line in result['lines']:
            if line['rec_width'] not in (192, 320, 480, 640, 960):
                raise ValueError('unexpected REC width')
            if len(line['box']) != 8 or not all(math.isfinite(x) for x in line['box']):
                raise ValueError('invalid detection box')
    j, c = java['lines'], csharp['lines']
    equal_count = len(j) == len(c)
    histogram = lambda lines: dict(sorted(collections.Counter(x['rec_width'] for x in lines).items()))
    ordered_widths = [x['rec_width'] for x in j] == [x['rec_width'] for x in c]
    boxes = max((abs(a-b) for jl, cl in zip(j, c) for a,b in zip(jl['box'],cl['box'])), default=None) if equal_count else None
    texts = sum(jl['text'] == cl['text'] for jl,cl in zip(j,c)) if equal_count else None
    rotations = equal_count and all(jl['rotation'] == cl['rotation'] for jl,cl in zip(j,c))
    return dict(variant=java['variant'], fma=java['fma'],
                java_mean_ms=java['mean_ms'], csharp_mean_ms=csharp['mean_ms'],
                java_over_csharp=java['mean_ms']/csharp['mean_ms'],
                java_lines=len(j), csharp_lines=len(c), exact_text_lines=texts,
                rotation_parity=rotations, java_widths=histogram(j), csharp_widths=histogram(c),
                ordered_width_parity=ordered_widths, max_ordered_box_coordinate_error=boxes,
                identical_ordered_crops=equal_count and ordered_widths and boxes == 0,
                warning='Same settings/pixels, but different crops are different workloads. No CER claim.')


def main(root):
    summaries = []
    for path in sorted(root.glob('*-java*-r*.json')):
        java = json.loads(path.read_text(encoding='utf-8-sig'))
        if ('-java-fma-' in path.name) != java['fma']:
            raise ValueError('FMA flag does not match result filename')
        peer = root / f"{java['variant']}-csharp-r{java['replica']}.json"
        csharp = json.loads(peer.read_text(encoding='utf-8-sig'))
        summary = dict(replica=java['replica'], **compare(java, csharp))
        if java['fma']:
            baseline = json.loads((root / f"{java['variant']}-java-r{java['replica']}.json").read_text(encoding='utf-8-sig'))
            summary['fma_reduction_percent'] = 100*(1-java['mean_ms']/baseline['mean_ms'])
            summary['fma_ordered_text_rotation_width_parity'] = [
                (line['text'], line['rotation'], line['rec_width']) for line in java['lines']
            ] == [(line['text'], line['rotation'], line['rec_width']) for line in baseline['lines']]
        summaries.append(summary)
    if not summaries:
        raise ValueError('no paired benchmark results')
    (root / 'summary.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    for s in summaries:
        print(f"{s['variant']} FMA={s['fma']}: Java {s['java_mean_ms']:.3f}ms / C# {s['csharp_mean_ms']:.3f}ms "
              f"= {s['java_over_csharp']:.2f}x; exact text {s['exact_text_lines']}/{s['java_lines']}, "
              f"identical crops={s['identical_ordered_crops']}")


if __name__ == '__main__':
    main(pathlib.Path(sys.argv[1]))
