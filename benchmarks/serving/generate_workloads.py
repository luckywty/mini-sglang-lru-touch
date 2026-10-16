"""Offline natural-text workload construction; replay never invokes a tokenizer."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import string

from transformers import AutoTokenizer

MODEL = '/home/wangwei/ai-infra/models/Qwen3-0.6B'
TEXT = (
    'The engineering team reviews a distributed service and records its behavior. '
    'Requests arrive independently, and each worker processes a small part of the task. '
    'A useful experiment fixes the input data and measures both waiting time and throughput. '
    'The report should include failed requests, resource use, and the limits of the evidence. '
    'Consider a library containing books about science, history, mathematics, and the arts. '
    'A reader can compare several explanations before choosing a clear and accurate answer. '
    'Please describe the topic in complete sentences with concrete examples and careful reasoning. '
)


class Factory:
    def __init__(self, model=MODEL, seed=31723):
        self.tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.rng = random.Random(seed)
        self.seed = seed

    def marker(self):
        return ''.join(self.rng.choices(string.ascii_letters, k=12))

    def prompt(self, length, prefix=''):
        for _ in range(100):
            text = prefix + self.marker() + '. ' + TEXT * (length//50 + 4)
            ids = self.tok.encode(text)[:length]
            prompt = self.tok.decode(ids, skip_special_tokens=False)
            if len(ids) == length and self.tok.encode(prompt) == ids:
                return prompt, ids
        raise ValueError('Could not construct an exact tokenizer round trip')

    def row(self, i, arrival, length, output, kind, group='', prefix='', shared=0, ignore=True):
        prompt, ids = self.prompt(length, prefix)
        if prefix:
            pids = self.tok.encode(prefix)
            assert ids[:len(pids)] == pids, 'Prefix changed at concatenation boundary'
        return dict(schema_version=1, request_id=f's{self.seed}-{i:06d}', arrival_s=arrival,
                    prompt=prompt, input_ids=ids, input_tokens=length, max_output_tokens=output,
                    prefix_group=group, shared_prefix_tokens=shared, **{'class': kind},
                    temperature=0.0, top_k=1, top_p=1.0, ignore_eos=ignore, timeout_s=120)

    def prefix(self,length=512):
        # End the common token region at a stable newline boundary. Joining a
        # marker directly to a cut word can change its last BPE token.
        for n in range(length-5,length+1):
            base, _ = self.prompt(n)
            for sep in ('\n', '\n\n', '.\n', '.\n\n', ' .\n'):
                prefix=base+sep
                ids=self.tok.encode(prefix)
                if len(ids)==length and all(self.tok.encode(prefix+s)[:length]==ids
                                        for s in ('Alpha','beta','Test')):
                    return prefix
        raise ValueError(f'No stable {length}-token prefix boundary')

    def make(self, workload, requests, rate, arrival_mode='poisson', cycle_s=6.0, background_output=256):
        rows, prime = [], []
        prefixes = {}
        if workload in ('W2', 'W3', 'W6'):
            for g in range(36):
                prefixes[g] = self.prefix()
            for g in range(4):
                prime.append(self.row(g, 0, 576, 2, 'prime', f'g{g}', prefixes[g], 512))
        if workload=='W4-X':
            prefixes={g:self.prefix(64) for g in range(4)}
            prime=[self.row(g,0,128,2,'prime',f'g{g}',prefixes[g],64) for g in range(4)]
        arrival = 0.0
        for i in range(requests):
            if workload == 'W5':
                # 36-request fixed wall-clock cycles: 4/8/12 background decode
                # requests and four long-prefill interferers per wave.
                cycle, j = divmod(i, 36)
                wave = 0 if j < 8 else (1 if j < 20 else 2)
                local = j if wave == 0 else (j-8 if wave == 1 else j-20)
                n = (4, 8, 12)[wave]
                start = cycle*cycle_s + wave*cycle_s/3
                if local < n:
                    arrival, length, output, kind = start, 128, background_output, 'background'
                else:
                    arrival = start + 0.12 + (local-n)*0.12
                    length, output, kind = 2048, 32, 'long'
                rows.append(self.row(i, arrival, length, output, kind))
                continue
            if arrival_mode == 'poisson' and i:
                arrival += self.rng.expovariate(rate)
            elif arrival_mode == 'burst':
                arrival = (i//32)*32/rate
            elif arrival_mode == 'closed':
                arrival = 0.0
            if workload in ('W2', 'W3', 'W6'):
                # Cold first within each burst; exactly 20% cold over 5 requests.
                hot = (i % 5 != 0)
                if workload=='W6':
                    start=(i//32)*32
                    end=min(start+32,requests)
                    cold_in_burst=end//5-start//5
                    hot=(i-start)>=cold_in_burst
                g = self.rng.randrange(4) if hot else (4 + (i//5) % 32)
                if workload == 'W6' and not hot:
                    # New cold prefix, never deliberately revisited.
                    prefix = self.prefix()
                    group = f'unique-{i}'
                else:
                    prefix, group = prefixes[g], f'g{g}'
                rows.append(self.row(i, arrival, 576, 32, 'hot' if hot else 'cold', group, prefix, 512))
            elif workload in ('W4','W4-X'):
                long = (i % 5 == 4)
                if workload=='W4-X' and not long and i%10<4:
                    g=self.rng.randrange(4)
                    rows.append(self.row(i,arrival,128,256,'short_hot',f'g{g}',prefixes[g],64))
                else:
                    rows.append(self.row(i, arrival, 2048 if long else 128,
                                         32 if long else 256, 'long' if long else 'short'))
            elif workload == 'W0':
                length, output = ((128, 32), (512, 128), (2048, 128))[i % 3]
                rows.append(self.row(i, arrival, length, output, 'correctness'))
            else:
                rows.append(self.row(i, arrival, 512, 128, 'control'))
        rows.sort(key=lambda r:r['arrival_s'])
        return rows, prime


def save(path, rows, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows).encode()
    if path.exists():
        assert path.read_bytes() == data, f'Refusing to overwrite frozen trace {path}'
    else:
        path.write_bytes(data)
    manifest = {**metadata, 'sha256':hashlib.sha256(data).hexdigest(), 'requests':len(rows),
                'classes':{k:sum(r['class']==k for r in rows) for k in set(r['class'] for r in rows)},
                'input_lengths':sorted(set(r['input_tokens'] for r in rows)),
                'output_lengths':sorted(set(r['max_output_tokens'] for r in rows))}
    path.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', default=MODEL)
    p.add_argument('--workload', required=True)
    p.add_argument('--seed', type=int, default=31723)
    p.add_argument('--requests', type=int, default=1000)
    p.add_argument('--rate', type=float, required=True)
    p.add_argument('--arrival-mode', choices=['poisson','burst','closed'], default='poisson')
    p.add_argument('--cycle-s', type=float, default=6.)
    p.add_argument('--background-output', type=int, default=256)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    rows, prime = Factory(a.model_path,a.seed).make(a.workload,a.requests,a.rate,a.arrival_mode,
                                                  a.cycle_s,a.background_output)
    print(json.dumps(save(a.output,rows,vars(a))))
    if prime:
        save(Path(a.output).with_suffix('.prime.jsonl'),prime,{'role':'prime'})
