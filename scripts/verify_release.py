"""Verify archived hashes, exact result arithmetic and manuscript references.

Uses only the Python standard library; does not load patient data or train models.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/20260804_aug_jitter005_mask010_full_paper_v1'


def main():
    errors = []
    manifest = json.loads((ROOT / 'docs/provenance/source_files.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        path = ROOT / entry['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['release_sha256']:
            errors.append(f"Hash mismatch or missing: {entry['path']}")
    python_paths = list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'scripts').glob('*.py')) + list((ROOT / 'analysis').glob('*.py')) + list((ROOT / 'tests').glob('*.py'))
    for path in python_paths:
        try:
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        except SyntaxError as error:
            errors.append(f'Syntax error: {path.relative_to(ROOT)}:{error.lineno}')
    checked_results = 0
    for path in sorted(RESULTS.rglob('exact_seed_metrics.csv')):
        rows = list(csv.DictReader(path.open(encoding='utf-8', newline='')))
        seeds = [int(row['seed']) for row in rows]
        if sorted(seeds) != [42, 2026, 3407]:
            errors.append(f'Unexpected seeds: {path.relative_to(ROOT)}')
        summary_path = path.with_name('exact_summary.csv')
        summary = next(csv.DictReader(summary_path.open(encoding='utf-8', newline='')))
        for metric in ['primary_shd_auroc', 'primary_shd_auprc', 'macro_auroc_11', 'macro_auroc_12']:
            values = [float(row[metric]) for row in rows]
            for suffix, actual in [('mean', statistics.mean(values)), ('std', statistics.stdev(values))]:
                expected = float(summary[f'{metric}_{suffix}'])
                if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
                    errors.append(f'Arithmetic mismatch: {path.relative_to(ROOT)}:{metric}_{suffix}')
        checked_results += 1
    references = 0
    graphic_roots = [ROOT / 'paper']
    for macro_path in (ROOT / 'paper').rglob('*.tex'):
        text = macro_path.read_text(encoding='utf-8')
        for block in re.findall(r'\\graphicspath\{((?:\{[^}]+\})+)\}', text):
            graphic_roots.extend(ROOT / 'paper' / name for name in re.findall(r'\{([^}]+)\}', block))
    for path in (ROOT / 'paper').rglob('*.tex'):
        content = path.read_text(encoding='utf-8')
        for command, name in re.findall(r'\\(input|include|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}', content):
            if '\\' in name or '#' in name:
                continue
            roots = [ROOT / 'paper'] if command in ['input', 'include'] else graphic_roots
            candidates = [root / name for root in roots]
            extensions = ['.tex'] if command in ['input', 'include'] else ['.pdf', '.png', '.jpg', '.eps']
            candidates += [Path(str(candidate) + ext) for candidate in list(candidates) for ext in extensions]
            if not any(candidate.is_file() for candidate in candidates):
                errors.append(f'Missing manuscript asset: {name} in {path.name}')
            references += 1
    report = {'status': 'PASS' if not errors else 'FAIL', 'source_hashes_checked': len(manifest['files']),
              'python_files_parsed': len(python_paths), 'result_groups_checked': checked_results,
              'manuscript_references_checked': references, 'errors': errors}
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
