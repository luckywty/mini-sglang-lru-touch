"""Copy completed campaign evidence for Git; preserve large raw files locally."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source, destination):
    progress = json.loads((source / 'PROGRESS.json').read_text())
    audit = json.loads((source / 'report/delivery-audit.json').read_text())
    if progress.get('phase') != 'measure complete' or not audit.get('measurement_complete'):
        raise RuntimeError('Campaign is not complete')
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite: {destination}')
    config = json.loads((source / 'frozen-config.json').read_text())
    paths = set()

    def include(relative):
        path = source / relative
        if path.is_file():
            paths.add(path)

    for name in ('REPORT.md', 'RESUME_EVIDENCE.md', 'frozen-config.json', 'PROGRESS.json'):
        include(name)
    for path in (source / 'report').rglob('*'):
        if path.is_file() and path.suffix in {'.json', '.csv', '.html', '.png', '.svg', '.py'}:
            paths.add(path)
    for path in (source / 'environment').glob('*'):
        if path.is_file() and path.suffix in {'.json', '.txt', '.lock', '.log'}:
            paths.add(path)
    for folder in ('config-history', 'amendments'):
        paths.update((source / 'environment' / folder).glob('*.json'))
    for label in ('H', 'S'):
        fingerprint = config[f'runtime_{label}_manifest_sha256']
        for suffix in ('.json', '.tar.gz'):
            include(f'environment/sources/{fingerprint}{suffix}')
        if sha(source / f'environment/runtime-{label}-files.json') != fingerprint:
            raise RuntimeError(f'Frozen manifest mismatch: {label}')
    for suffix in ('.json', '.tar.gz'):
        include(f'environment/tooling/{config["tooling_sha256"]}{suffix}')
    paths.update(p for p in (source / 'patches').rglob('*') if p.is_file())
    paths.update((source / 'traces').glob('*.manifest.json'))
    for experiment in (source / 'measure').iterdir():
        if experiment.is_dir():
            for name in ('run.json', 'summary.json', 'extension-decision.json'):
                paths.update(experiment.rglob(name))
    paths.update((source / 'preflight').rglob('acceptance.json'))
    paths.update((source / 'preflight/numerics').glob('*.json'))
    for name in ('natural-hf-reference.json', 'abort-gpu-incident.json',
                 'github-publication-validation.txt',
                 'github-publication-validation-unconfigured-env.txt'):
        include(f'preflight/{name}')
    for path in paths:
        if path.stat().st_size > 25 * 1024**2:
            raise RuntimeError(f'Unexpectedly large input: {path}')
    destination.mkdir(parents=True)
    copied = {}
    for path in sorted(paths):
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        digest = sha(path)
        if sha(target) != digest:
            raise RuntimeError(f'Copy verification failed: {relative}')
        copied[str(relative)] = digest
    manifest = dict(exported_utc=datetime.now(timezone.utc).isoformat(),
                    source_results_directory=str(source), measurement_complete=True,
                    formal_attempts=audit['formal_attempts'], eligible_runs=audit['eligible_runs'],
                    copied_files=copied, raw_evidence_checksums='report/artifact-sha256.json',
                    excluded=['weights', 'full trace JSONL', 'client.json', 'requests.jsonl',
                              'tokens.bin', 'measurement.pkl', 'per-run GPU/CPU telemetry',
                              'superseded source archives'],
                    scope='Derived evidence and frozen sources, not the complete raw dataset; '
                          'copied bytes and historical machine paths are unchanged.')
    (destination / 'export-manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    sums = [(str(p.relative_to(destination)), sha(p))
            for p in sorted(destination.rglob('*')) if p.is_file()]
    (destination / 'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name, digest in sums))
    return dict(files=len(sums) + 1, copied_bytes=sum(p.stat().st_size for p in paths),
                destination=str(destination), formal_attempts=audit['formal_attempts'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=REPO.parent / 'benchmark-results')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.results.resolve(), args.output.resolve()), ensure_ascii=False))
