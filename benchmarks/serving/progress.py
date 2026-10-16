"""Read-only terminal progress; standard library only, no server/GPU calls."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time


EXPERIMENTS = [('E02', 3), ('E03', 3), ('E04', 4), ('E05', 3),
               ('E01', 2), ('E02-low', 2), ('E06', 7)]


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def bar(done, total, width=20):
    filled = min(width, int(width * done / total)) if total else 0
    return '[' + '#' * filled + '-' * (width - filled) + ']'


def render(root):
    config = read_json(root / 'frozen-config.json')
    campaign = read_json(root / 'CAMPAIGN_STATE.json')
    watcher = read_json(root / 'WATCHER_STATE.json')
    progress = read_json(root / 'PROGRESS.json')
    pid = campaign.get('driver_pid')
    try:
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
        live = b'suite.py' in cmd and b'measure' in cmd
    except OSError:
        live = False
    lines = [datetime.now().astimezone().isoformat(timespec='seconds'),
             f"Controller PID {pid}: {'RUNNING' if live else 'NOT RUNNING'}; "
             f"watcher: {watcher.get('status', 'unknown')}",
             'Progress counts completed configurations, including failed outcomes; '
             'quality-valid counts are separate.']
    total_done = total_low = total_high = 0
    for experiment, variants in EXPERIMENTS:
        directory = root / 'measure' / experiment
        cells = set()
        selected = set()
        completed_attempts = 0
        for path in directory.rglob('run.json'):
            info = read_json(path)
            if (not info or info.get('superseded_by_protocol') or
                    info.get('protocol_version') != config.get('protocol_version')):
                continue
            cell = (info.get('seed'), info.get('variant'))
            if info.get('status') == 'complete' or info.get('formal_outcome_recorded'):
                cells.add(cell)
            if info.get('status') == 'complete':
                completed_attempts += 1
            if info.get('selected_for_primary'):
                selected.add(cell)
        decision = read_json(directory / 'extension-decision.json')
        low = variants * 5
        high = variants * 10
        if 'extend_to_10' in decision:
            low = high = variants * (10 if decision['extend_to_10'] else 5)
        target = str(low) if low == high else f'{low}-{high}'
        # Before the extension decision the bar uses the five-repeat minimum.
        lines.append(f'{experiment:7} {bar(len(cells), low)} {len(cells):3}/{target:<7} '
                     f'quality-valid={len(selected):3}; '
                     f'extra completed attempts={max(0, completed_attempts - len(cells))}')
        total_done += len(cells)
        total_low += low
        total_high += high
    lines.append(f'Total: {total_done}/{total_low}-{total_high} configurations '
                 f'({100 * total_done / total_high:.0f}%-'
                 f'{100 * total_done / total_low:.0f}% by count, NOT elapsed time).')
    lines.append(f"Checkpoint: {progress.get('experiment', progress.get('phase', 'unknown'))}; "
                 f"last whole seed={progress.get('last_seed', 'unknown')}")
    log = root / campaign.get('log', 'missing.log')
    try:
        with log.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 8192))
            tail = stream.read().decode(errors='replace').splitlines()
        ready = next((line for line in reversed(tail) if line.startswith('READY ')), '')
        if ready:
            lines.append(('Current: ' if live else 'Last run: ') + ready)
        if tail:
            lines.append('Latest: ' + tail[-1])
        lines.append(f'Log age: {max(0, time.time() - log.stat().st_mtime):.0f}s')
    except OSError:
        lines.append('Log unavailable')
    lines.append('Pending extension decisions can increase totals. Ctrl+C only closes this viewer.')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path,
                        default=Path(__file__).resolve().parents[3] / 'benchmark-results')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=15)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('--interval must be >=1 second')
    try:
        while True:
            if args.watch and sys.stdout.isatty():
                print('\033[2J\033[H', end='')
            print(render(args.results), flush=True)
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
