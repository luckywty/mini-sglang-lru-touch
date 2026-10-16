# RTX 5060 Ti 8GB serving experiments

Authoritative local results: `/home/wangwei/ai-infra/Promotion/benchmark-results`.
Published final evidence: [2026-10-16 results](../../results/rtx5060ti-20261016/README.md).
Read its `PROGRESS.md`, `PROGRESS.json` (when present), run receipts and logs before
resuming. A planning document or old CPU simulation is never a GPU result.

## Sources and common patches

- `mini-sglang-upstream`: official sgl-project repository, detached
  `9a91cfafe754aa85daee49998176275667eb58f2`. This Git checkout stays clean.
- Optimized implementation starts at `7cd28c7bda8f6881558fe810694019d3e46994b5`.
  The local experimental branch contains tooling and documentation, preserving
  the original `main` reference and all earlier history.
- `prepare_sources.py` exports immutable Git versions to `runtime-H`/`runtime-S`.
  The same collector entry hook, host-terminal repair and HTTP edge patch are
  applied to both; exact file manifests and source archives are retained.
- Raw upstream overlap failed length/terminal validation: O32 delivered31 ACKs
  and emitted two server terminals. The shared repair checks host sequence length
  after append, and suppresses already-released results after EOS/Abort.
  A later saturated C64 run exposed a shared Abort/prefill double-cache bug.
  The common repair marks the freed object in the existing recent-release set;
  old formal data/source snapshots are preserved but superseded for uniform pairing.
  **H with this repair must be labeled H*, never pristine upstream.**
- HTTP keepalive3600s supports prewarmed benchmark connections; X-Bench-UID maps
  requests even when they time out before their first token. No sampling/model/
  attention policy is modified by these edge hooks.

## Verified environment

Python3.12.14; torch2.9.1/CUDA Runtime12.8; FlashInfer0.7.0.post1 from uv.lock;
Transformers4.57.3; sgl-kernel0.3.21; TVM-FFI0.1.14.post1. Toolkit13.2.86 and GCC14
are selected by `env.sh`: Toolkit12.9 failed on GCC15 and then on glibc2.43.
FI/BF16/sm120 kernels and graph sizes1/2/4/8/16 ran successfully on the target.
Additional client packages are httpx0.28.1 and uvloop0.22.1.

Model: `/home/wangwei/ai-infra/models/Qwen3-0.6B` (unquantized BF16). Exact files
are identified by `environment/model-files.json`; a historical HF revision is
not asserted from the directory name. Never substitute the W4A16 directory.

## Workflow

Read-only terminal progress (refreshes every15s; Ctrl+C only closes the viewer):

```bash
python3 /home/wangwei/ai-infra/Promotion/mini-sglang-lru-touch/benchmarks/serving/progress.py --watch
```

Bars count completed configurations, including failed outcomes. Quality-valid
counts are shown separately. Pending five-to-ten-repeat decisions make the total
a range; the percentage counts configurations and is not a time estimate.

```bash
cd /home/wangwei/ai-infra/Promotion/mini-sglang-lru-touch
source benchmarks/serving/env.sh
python benchmarks/serving/prepare_sources.py  # only while all run servers are stopped
taskset -c 8,9 python benchmarks/serving/suite.py --phase preflight
taskset -c 8,9 python benchmarks/serving/suite.py --phase pilot
taskset -c 8,9 python benchmarks/serving/calibrate_fixed_arrivals.py
# Do not bypass a failed calibration receipt.
python benchmarks/serving/launch_campaign.py  # detached Linux session, refuses duplicate controller/port
python benchmarks/serving/analyze_results.py --results ../benchmark-results/measure --output-dir ../benchmark-results/report
python benchmarks/serving/finalize_campaign.py  # requires measure complete; --partial for a snapshot
```

The final fixed-arrival calibration completed15 runs and **failed** the strict
tail gate. All ITL effects remain exploratory and are excluded from resume
gains. Capacity/affinity diagnostics and all failed gates are retained; matrix
execution does not turn a failed calibration into acceptance.

Formal runs use G1: FI/fa2, page1, KV16384, seq2304, max running16, prefill256,
graph16, greedy/ignore_eos for fixed lengths. E02/E03/E04/E05 have priority;
E01 is the compatible control and E06 supplies the complete legal combined block.
Five seeds31723–31727 are paired; preregistered instability extends to31732.
No B-without-A or D-without-C configuration is invented.

## Measurement boundaries

`replay.py` starts timing before the HTTP call and records every data ACK,
including empty text, plus separate first/last nonempty text times, finish and
DONE. ACK counts/ordinals/UIDs are reconciled against scheduler-generated IDs
before token throughput or ACK ITL is accepted. The existing benchmark client's
summary and usage=0 responses are not used.

`collector.py` observes only existing host results after the engine's existing
D2H completion. There is no added hot-path GPU synchronization/copy or per-token
printing. Token data stay in bounded binary arrays and are dumped at an explicit
run boundary. CUDA allocator stats are read from the worker at boundaries.
Light mode retains complete token timing and phase/budget events, but samples
Decode request-row details every32 batches; detailed probe/LRU counts are for
full diagnostic mode. Missing counters are not interpreted as zero.

Server emit ITL, client ACK ITL, and per-request TPOT remain distinct. Throughput
uses successful generated tokens / full arrival-to-drain duration. Failures and
timeouts stay in the denominator and in a separate ledger. Unadmitted and censored
waits must not be represented as completed waiting-time quantiles.

## Correctness limitation

Cross-shape greedy bitwise equality and the initial absolute HF-logit tolerance
failed. This failure is retained. On every one of16 fixed differing contexts,
H24/S24 logits were bitwise equal (2,430,976 values); H alone changed7 argmaxes
when only chunk size changed256→24. All51 inference math source files match.
The fixed-length performance study therefore has a stated upstream-equivalence
scope; it does not assert bitwise text invariance or quality losslessness.

## Recovery and evidence

Never overwrite a previous run directory. Completed valid receipts are reused;
interrupted attempts and fixed-rule quality retries have separate directories.
Every server and telemetry process has its own owned process group; shutdown
never uses global `pkill`. All failed, timed-out, OOM and polluted runs remain.

Raw artifacts include immutable trace JSONL/manifests, run.json, client.json,
measurement.pkl, tokens.bin (int64 rows: uid/index/token/host-ns/terminal),
requests.jsonl, GPU500ms telemetry, independent CPU telemetry and both server logs.
Results live outside the repository because its .gitignore excludes JSON/JSONL.
Source/tooling snapshots are archived explicitly, not inferred from git status.

Final reporting is offline and does not rewrite the measurement rule.
`finalize_campaign.py --partial` labels an incomplete snapshot explicitly; the
normal command requires `PROGRESS.json` to say `measure complete`. A detached
`watch_campaign.py --pid PID` can track one identified controller and finalize
only after actual completion. It never starts another GPU suite or kills a PID.
See TOOL_AUDIT.md for the implemented tool inventory and source-path audit.

The detached launcher verifies frozen runtime-file and model SHA256 identities
before starting the suite. Its driver and watcher have independent Linux
sessions and stdin=/dev/null, so they remain after the launching command exits.
It does not rerun accepted preflight/pilot/calibration work. The 2026-10-16 matrix
has completed; failed gates and insufficient quality-valid pairs remain explicit
limitations in its final report.

To export completed evidence to a new, nonexisting Git publication folder:

```bash
python benchmarks/serving/export_results.py --output results/rtx5060ti-20261016
```

The exporter copies statistics, formal receipts/summaries, figures, failed gates,
selected frozen source/tooling archives and checksums. Copied bytes are verified;
existing exports are never overwritten. Full traces, weights, raw token/request
streams and per-run telemetry remain locally. Their original fingerprints are in
`report/artifact-sha256.json`. Historical absolute paths in evidence are retained;
on another machine, adapt ROOT/MODEL/PY in the tools and env.sh before a new run,
then generate new manifests rather than reusing the original frozen identities.
