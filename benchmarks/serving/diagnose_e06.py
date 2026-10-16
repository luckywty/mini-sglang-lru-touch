"""Retrospective E06 mechanism accounting from archived local CPU records."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
from statistics import mean, median


def diagnose(root):
    records = []
    for variant in ['H0000', 'S1000', 'S1100', 'S1011', 'S1111']:
        for seed in range(31723, 31728):
            directory = root / 'measure/E06' / f'seed-{seed}' / variant
            snapshot = pickle.loads((directory / 'measurement.pkl').read_bytes())
            client = json.loads((directory / 'client.json').read_text())
            requests = [json.loads(line) for line in
                        (directory / 'requests.jsonl').read_text().splitlines()]
            start = client['start_ns']
            early = [b for b in snapshot['batches'] if start <= b[0] < start + 60_000_000_000]
            decodes = [b for b in early if b[1] == 'decode']
            competing_prefills = [b for b in early if b[1] == 'prefill' and b[4] > 0]
            chunks = Counter()
            for batch in snapshot['batches']:
                if batch[1] == 'prefill':
                    chunks.update(row[0] for row in batch[12])
            long_chunks = [chunks[r['uid']] for r in requests
                           if r['class'] == 'long' and not r['audit_errors']]
            records.append(dict(
                variant=variant, seed=seed, source_directory=str(directory),
                early_window_seconds=60,
                early_decode_batch_mean=mean(b[2] for b in decodes),
                early_competing_prefill_tokens_median=median(b[7] for b in competing_prefills),
                early_pending_max=max(b[5] for b in early), early_pending_last=early[-1][5],
                successful_long_prefill_chunks_median=median(long_chunks),
                successful_long_prefill_chunks_max=max(long_chunks),
                first_abort_s=(min(x[1] for x in snapshot['aborts']) - start) / 1e9
                              if snapshot['aborts'] else None,
                total_scheduler_decision_cpu_s=sum(snapshot['decision_prepare_ns'][::2]) / 1e9))
    return dict(scope='Retrospective diagnosis from archived data; not a new experiment or '
                      'kernel profile. First60s avoids timeout Abort effects. Long chunk '
                      'counts condition on completed audited long requests.', records=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path,
                        default=Path(__file__).resolve().parents[3] / 'benchmark-results')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.results.resolve())
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Accounted for {len(report["records"])} archived runs; no GPU execution.')
