# Benchmark 执行手册与最小工具需求

本文件中的命令供**未来在目标 RTX 5060 Ti 8GB 上执行**。本阶段只创建规划文档，未执行完整压测或下面的模型命令。已存在命令与拟新增工具分开标识；不把缺失脚本说成可直接运行。

先阅读 [设计与指标](benchmark_plan.md)、[Workload](workload_design.md)、[消融表](ablation_matrix.md)。所有命令在 bash 中运行；目标机须预先有同revision的本地未量化模型，不在本方案中自动下载。

## 1. 现有工具审计

| 文件 | 当前能复用什么 | 正式实验缺口 / 风险 | 最小处理 |
|---|---|---|---|
| `benchmark/offline/bench.py` | LLM.generate、计时与固定随机输入的结构 | 默认num_seqs256、Prefill16384、graph256、page256；与8GB/FI路径不匹配；requested output计数，无请求级指标 | 不直接运行默认main；新薄wrapper复用LLM，显式小配置、真实output IDs计数 |
| `benchmark/offline/bench_wildchat.py` | 自然文本/聊天tokenization例子 | 自动下载较大数据，max output4096；只适合后续缩小后的本地数据验证 | P2，先准备小型本地冻结子集 |
| `python/minisgl/llm/llm.py` | 离线token IDs输入、输出token_ids | offline_receive_msg按prefill_budget逐轮喂输入，策略会影响喂入节奏；generate非流式；最终EOS可能被输出列表隐去 | 用作正确性/离线吞吐，不冒充固定arrival的Serving tail benchmark |
| `benchmark/online/bench_simple.py` | 异步HTTP入口 | 硬编码C64、input8192，生成与压测准备耦合；错误仅日志；现有计数不适合正式结论 | 复用通信结构，改用冻结trace的薄collector |
| `python/minisgl/benchmark/client.py` | AsyncOpenAI、trace时间戳、JSON结构的一部分 | 请求create返回后才初始化tics，漏请求早期时间；role/empty/finish事件计入；把SSE event数用于吞吐；累积event gap命名TPOT；没有P95；input_len可能不含chat模板，output_len为请求值；asyncio.gather失败不形成完整终态账 | 新collector替代计量逻辑；不使用现有process_benchmark_results计算正式指标 |
| `benchmark/online/bench_qwen.py` | trace缩放与回放入口 | 自动下载，dummy=True人为制造共享prefix；hash_ids被忽略；未冻结原始payload | 不作为真实缓存实验；只后续复用arrival结构 |
| `benchmarks/bench_prefill_policy.py` | CPU排序微基准、缓存/队列语义检查 | pin/memory等替身，未加载GPU模型；不能得Serving TTFT/ITL | 用作P2诊断，结果标CPU模拟 |
| `benchmarks/bench_batch_schedule_policy.py` | CPU Scheduler、资源断言、K/D预算轨迹 | fake forward/逻辑step arrivals；无真实GPU或HTTP计时 | 保留为语义验证，不把历史数据折算成GPU提升 |
| `python/minisgl/benchmark/perf.py` | CUDA events的kernel级计时思路 | 主动同步/graph重复，适合microkernel，不能直接放Serving主循环 | 仅独立诊断用，不能用于TTFT |

API流式返回不提供可靠usage；非流式usage的三项数值目前写死为0。detokenizer会处理不可打印文本并隐去最后EOS，SSE可为空；重新tokenize最终文本不保证恢复原生成token序列。源码当前保持DetokenizeMsg→UserReply→数据event一一对应，首个role与首token同事件，额外finish/DONE需排除；经server计数/顺序对账后可复用该映射测client token-ack ITL，无需改SSE协议。为验证生成计数、获得scheduler与后端ITL，仍需最小server侧计数/时间戳。现有客户端数据不能事后修正成完整端到端首token时间。

## 2. 目标机环境与版本锁定（现有工具命令）

```bash
cd /path/to/mini-sglang
BENCH_REPO="$PWD"
BENCH_PY="$BENCH_REPO/.venv/bin/python"
BENCH_MODEL=/path/to/Qwen3-0.6B/snapshot
BENCH_RUN_DIR=/tmp/minisgl-serving-20261014
mkdir -p "$BENCH_RUN_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export MINISGL_DISABLE_OVERLAP_SCHEDULING=0

uname -a > "$BENCH_RUN_DIR/uname.txt"
cat /etc/os-release > "$BENCH_RUN_DIR/os-release.txt"
lscpu > "$BENCH_RUN_DIR/lscpu.txt"
nvidia-smi > "$BENCH_RUN_DIR/nvidia-smi.txt"
nvcc --version > "$BENCH_RUN_DIR/nvcc.txt"
git rev-parse HEAD > "$BENCH_RUN_DIR/head.txt"
git status --porcelain=v1 > "$BENCH_RUN_DIR/status.txt"
git diff --binary HEAD > "$BENCH_RUN_DIR/worktree.patch"
sha256sum "$BENCH_MODEL"/*.safetensors "$BENCH_MODEL/config.json" \
  "$BENCH_MODEL/tokenizer.json" "$BENCH_MODEL/tokenizer_config.json" \
  > "$BENCH_RUN_DIR/model-sha256.txt"
"$BENCH_PY" - <<'PY' > "$BENCH_RUN_DIR/python-environment.txt"
import sys, importlib.metadata as m
import torch
from torch.utils.cpp_extension import CUDA_HOME
print(sys.version)
print('torch', torch.__version__, 'runtime', torch.version.cuda)
print('cuda_available', torch.cuda.is_available(), 'arch_list', torch.cuda.get_arch_list())
print('cuda_home', CUDA_HOME)
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
    print('capability', torch.cuda.get_device_capability(0))
    print('free_total_bytes', torch.cuda.mem_get_info(0))
for name in ['flashinfer-python','transformers','triton','sgl-kernel','apache-tvm-ffi','openai']:
    print(name, m.version(name))
import flashinfer
print('flashinfer_import', flashinfer.__version__)
print('installed_distributions')
for d in sorted(m.distributions(), key=lambda d: d.metadata['Name'].lower()):
    print(d.metadata['Name'], d.version)
PY
```

同时保存`CUDA_HOME`实际nvcc版本、编译器/Ninja、驱动、WSL/宿主版本、运行环境变量白名单、GPU UUID和功耗设置；不导出整个环境，以免泄漏凭据。无GPU、不是目标5060Ti8GB、没有sm_120支持、JIT失败或模型不完整时，停止正式性能阶段并保留环境错误，不自动换模型/后端/精度。

原审计source集合指纹算法（本次新文档不参与此集合）：遍历`python/minisgl`、`benchmark`、`benchmarks`、`tests`，后缀`.py/.cpp/.cu/.h/.cuh`，排除`__pycache__`；按路径排序，向SHA256逐项输入`path.encode()+b'\0'+sha256(file_bytes).digest()`。未来新增collector会改变指纹，应记录新manifest与每文件hash，不要求等于审计值。`git diff`不含untracked文件，必须另外归档这些源码；不要误认为只有patch就能复现S。

## 3. 原版与优化版隔离（命令模板，需下一阶段执行）

基线从固定commit导出到新目录，避免修改现有工作区或`.git`：

```bash
BENCH_BASE_DIR="$(mktemp -d /tmp/minisgl-upstream-9a91cfa-XXXXXX)"
git archive 9a91cfafe754aa85daee49998176275667eb58f2 | tar -x -C "$BENCH_BASE_DIR"
```

S使用冻结的源码快照目录，包含未跟踪benchmarks/tests；只归档所需源码和manifest，不复制`.venv`、大模型、旧results。将相同benchmark采集补丁应用到H/S的隔离副本，单独存patch。H/S共用**同一个绝对Python解释器**，启动前设置各自PYTHONPATH，并检查`importlib.util.find_spec('minisgl').origin`确认加载对应目录；防止editable install将H误导向S。

仓库`.gitignore`忽略所有`*.json`/`*.jsonl`。实验trace与manifest须显式纳入外部归档/校验清单；不能只依赖git status或普通git add证明它们已保存。需要提交少量fixture时，下一阶段再定向force-add指定文件，禁止批量强加模型或大结果目录。

H使用原版CLI公共参数；不向H传新策略flag。A-ISO、V0010、V0011仅在后续通过代码路径/正确性验收后创建，尚无可执行现成命令。

## 4. 模型加载与显存验收

先G0：8192页、seq2304、max running8、graph0、Prefill256，8–16个小请求确认FI/JIT可执行；再启graph8；最后G1 graph16/max running16/16384页。每步记录启动峰值、warmup峰值、最坏输入峰值、最小free。JIT不在测量区间发生。

采用worker内部measurement边界API（待实现）重置peak stats，在结束边界读取；不能在benchmark client的另一个CUDA context读取worker allocator峰值。启动阶段自己也会reset peak，因此外部采样与各阶段峰值须保留，不能仅用最后一次PyTorch peak宣称启动无尖峰。

正式G1资源参数，**这些flag现在已存在**：

```bash
BENCH_COMMON=(
  --model-path "$BENCH_MODEL" --dtype bfloat16 --tp-size 1
  --attention-backend fi --cache-type radix --page-size 1
  --num-pages 16384 --max-running-requests 16
  --max-seq-len-override 2304 --max-prefill-length 256
  --cuda-graph-max-bs 16 --memory-ratio 0.70
  --num-tokenizer 0 --host 127.0.0.1 --port 1919
)
```

memory-ratio在显式num-pages下不控制实际KV预算；保留该字段只为config明确。正式图捕获应是`[1,2,4,8,16]`，如日志不同先解释。不要使用shell-mode，它会把max running和graph改成1。

G1不能保留余量时，整个比较组统一改为G1-small；不能只给优化组8192页、基线16384页。上下文长度、KV容量、client并发逐级改变，每次只改一个验证维度；负载扫描是新实验配置，不与原档位混算。

## 5. Server 启动示例（现有命令；本阶段未运行）

以下为独立启动方式，前一个进程树退出后才能启动下一个。常驻server的生命周期交给未来runner，记录退出码、stderr、启动config、随机种子、ready和graph完成标记；使用拥有的PID/process group退出，避免pkill其他任务。

```bash
# H0000：在独立terminal/runner subprocess启动
PYTHONPATH="$BENCH_BASE_DIR/python" "$BENCH_PY" -m minisgl "${BENCH_COMMON[@]}"
```

```bash
# S1000：当前兼容配置，A仍开启
PYTHONPATH="$BENCH_REPO/python" "$BENCH_PY" -m minisgl "${BENCH_COMMON[@]}" \
  --schedule-policy fcfs --lpm-aging-tokens-per-pass 0 \
  --fairness-max-wait-steps 0 --batch-schedule-policy prefill-first
```

```bash
# S1010：bounded + 固定64预算
PYTHONPATH="$BENCH_REPO/python" "$BENCH_PY" -m minisgl "${BENCH_COMMON[@]}" \
  --schedule-policy fcfs --fairness-max-wait-steps 0 \
  --batch-schedule-policy decode-first-bounded --max-consecutive-decodes 4 \
  --protected-prefill-tokens 64
```

```bash
# S1111：全部优化
PYTHONPATH="$BENCH_REPO/python" "$BENCH_PY" -m minisgl "${BENCH_COMMON[@]}" \
  --schedule-policy lpm-aging --lpm-aging-tokens-per-pass 64 \
  --lpm-queue-threshold 128 --fairness-max-wait-steps 0 \
  --batch-schedule-policy decode-first-bounded --max-consecutive-decodes 4 \
  --protected-prefill-tokens 64 --adaptive-prefill-budget \
  --target-batch-tokens 32 --min-chunk-tokens 8
```

固定24预算对照：S1010仅将protected-prefill-tokens改为24；D的上限仍为64，此差异是静态/动态预算比较的研究变量。比较C时normal budget256等其余参数相同。normal-loop机制验证仅在单独系列设置`MINISGL_DISABLE_OVERLAP_SCHEDULING=1`，不得混进overlap主结果。

## 6. 最小新增文件、接口与职责（全部待实现）

目录建议仅新增`benchmarks/serving/`；4个Python文件加一份矩阵配置足够，避免另建大型框架。

| 拟新增文件 | 接口与输入 | 输出 / 职责 |
|---|---|---|
| `generate_workloads.py` | `--model-path --workload W3 --seed --requests --arrival-mode --rate --output --prime-output`，可选`--input/--output-tokens --burst-size --cycle-s` | 校验真实tokenizer长度/LCP，写trace.jsonl+manifest；不访问GPU、不自动下载 |
| `replay.py` | `--base-url --trace --model --mode open-loop\|closed-loop --concurrency --timeout-s --max-outstanding --output`，可选`--slo-file --disable-progress` | 请求前计时，禁用重试，warm连接，保存token ack及非空文本事件、状态、异常、实际发送与计划偏差；合并server token统计后算精确吞吐 |
| `run_experiments.py` | `--matrix --repo-map --model-path --seeds --results-dir --phase preflight\|pilot\|measure --dry-run` | 启停server/telemetry；按block平衡顺序；执行warmup/prime/replay/drain；校验环境/显存/trace/hash与机制激活；失败也落run manifest；dry-run打印展开表 |
| `analyze_results.py` | `--results --reference --group-by --bootstrap-unit run --bootstrap 10000 --output-dir` | 统一计算指标、配对CI、失败台账、CSV/JSON及静态SVG/PNG图；不根据结果删异常 |
| `matrix.yaml` | variants/公共flags/源码位置、trace hash、SLO、温度稳定规则、重复数与顺序seed、输出规范 | 单一实验配置；schema拒绝D1/C0和未实现A0开关；公共参数由manifest验证 |

server instrumentation建议独立`python/minisgl/benchmark/collector.py`，加少量明确的观测hook；需要接入的函数列于设计文档第10节。本阶段不新增该文件，也不修改消息/核心类。

采集接口建议 `begin_run(run_id, mode)` / `end_run()`，通过benchmark专用本地控制通道或启动参数触发；只控制缓冲区、峰值统计和计时，不改变scheduler行为。主run保存完整token时间戳，详细batch/GPU profiling另跑。H使用相同边界的hook；若上下文不兼容，benchmark adapter只作数据转换，逐项核验其不改变功能。

最少需要的server字段：uid与输入长度；首次准入C/q/h；token index/id/host ready time；batch阶段、real/padded size、n_decode、budget、actual prefill extends、Kstreak；Table/KV拒绝；free pages/tree size/evict计数；结束/abort。token记录不能重复计入overlap终态；如发现重复token/终态，按正确性问题报告，不在聚合时无声去重。

## 7. 拟议自动化命令模板（脚本不存在，不能现在执行）

```bash
# 阶段1：仅展开配置并检查依赖/公共参数
"$BENCH_PY" benchmarks/serving/run_experiments.py \
  --matrix benchmarks/serving/matrix.yaml --repo-map /path/to/repos.json \
  --model-path "$BENCH_MODEL" --results-dir "$BENCH_RUN_DIR" \
  --seeds 31723 31724 31725 31726 31727 --phase preflight --dry-run

# 阶段2：H pilot后选定绝对到达率，再生成冻结负载
"$BENCH_PY" benchmarks/serving/generate_workloads.py \
  --model-path "$BENCH_MODEL" --workload W4 --seed 31723 \
  --requests 1000 --arrival-mode poisson --rate '<H pilot确定的req/s>' \
  --output "$BENCH_RUN_DIR/traces/W4-s31723.jsonl"

# 阶段3：正式矩阵，runner每次重启并重放相同warmup/prime
"$BENCH_PY" benchmarks/serving/run_experiments.py \
  --matrix benchmarks/serving/matrix.yaml --repo-map /path/to/repos.json \
  --model-path "$BENCH_MODEL" --results-dir "$BENCH_RUN_DIR" \
  --seeds 31723 31724 31725 31726 31727 --phase measure

# 阶段4：聚合和图表
"$BENCH_PY" benchmarks/serving/analyze_results.py \
  --results "$BENCH_RUN_DIR" --reference H0000 \
  --group-by workload,trace_family,arrival_rate,resource_profile,cache_state \
  --bootstrap-unit run --bootstrap 10000 --output-dir "$BENCH_RUN_DIR/report"
```

rate占位符必须替换数值；不同seed分别生成trace。所有未指定的工具默认值写进manifest。没有实测λ之前不给一个看似精确却未经验证的req/s。

## 8. 正式run的固定步骤

1. 清点宿主机/WSL进程、GPU任务、Windows桌面负载、后台下载、节电/插电状态；采集idle VRAM、CPU/GPU温度与时钟。目标CPU若只有4线程，不照搬4+核的隔离方案。
2. 按block随机顺序启动指定源码server，检查实际backend=fi/page1、KV页数、实际graph列表、权重与tokenizerhash。等待启动和JIT结束；记录启动失败也作为run。
3. 固定shape warmup：C1/4/8/16，I128/512/2048、O8–32，每shape少量请求；测试prefix与warmup不同。不得在measurement中生成prompt/首次加载tokenizer/JIT。
4. 完成预热后检查温度稳定：例连续60s温度变化≤2°C、无明显throttle；不稳定延长相同规则的settle并记录，不能只为某策略多冷却。prime前无测试队列。
5. 根据cache条件重放独立prime trace或不prime；确认无在途prime，记录实际C。没有安全cache reset时，重启和prefix隔离定义冷条件。
6. collector begin_run，reset measurement peak，启动低频telemetry；记录run起点，按绝对时间发送冻结trace。所有请求禁重试、共享连接池；pool连接上限足以覆盖outstanding，不在隐藏连接池排队。
7. 注入后drain，记录每请求完整终态；deadline触发取消/abort，并等待资源回收或标异常退出。超时/已丢弃请求不能仍算success。measurement结束后再统一flush缓冲和CUDA event诊断结果。
8. 保存allocator peaks、最小free、请求数/token计数对账、resources状态、计数器与终态。发现allocator assertion/OOM/非预期截断/输出差异，保留所有结果并终止该server，不假装正常继续下一cell。
9. 停止本run拥有的server进程树与telemetry。下一个变体重新启动。同一block使用相同trace，5个block的顺序平衡；run目录从不覆盖。

每请求完整性验收：offered=success+failure+timeout+cancelled+unresolved，类别互斥；实际I/O与manifest一致或错误被记录；uid/index序列连续且无重复；first/last/terminal时间合理；无多余truncation；cold删失样本保留。对生成token的计数不能直接从SSE数量或usage=0读出。

正确性小集覆盖唯一prefix、完整命中、partial命中、跨chunk、多请求同prefix、不同batch大小以及自然EOS结束。先比较greedy token IDs与参考；若仅因浮点归约/近同分logit导致分歧，应独立定位并以固定输入的逐步logit误差和已预注册容差验证，而非放宽到“文本看起来差不多”。不为获得吞吐而改精度、权重、最大输出或接受未解释的cache数值错误。

## 9. CPU/client与遥测控制

预生成输入与JSON；长连接，SDK `max_retries=0`、明确timeout，关闭每token进度条和同步磁盘写。HTTP/SSE解析时间和event-loop lag写入内存，结束后flush。AsyncOpenAI以外也可用现有依赖中的异步HTTP客户端，但不得暗改响应解析/计时口径。

目标机按实际物理核安排client与server，避免绑在同一物理核的SMT线程。只有4逻辑CPU时先做默认与有限亲和的校准，保留能让server/tokenizer获得足够CPU的设置；不得为隔离client把server挤到单核后声称GPU受限。可远端client做交叉验证；增加网络后延迟域不同，不能把远端TTFT与本地TTFT混算收益。

GPU遥测命令模板（已存在nvidia-smi；另开进程，本阶段未运行）：

```bash
nvidia-smi --query-gpu=timestamp,name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,utilization.gpu \
  --format=csv -lms 500 > "$BENCH_RUN_DIR/gpu-telemetry.csv"
```

字段在WSL不可用则保存错误、改用受支持字段，并将缺项设NA；不能估造功耗/温度。CPU每秒采样各进程CPU、RSS、context switches；磁盘写入避免measurement热点。client dummy SSE承载2倍预计事件率后，还须在真实run检查lag，dummy校准不代表真实争用已排除。

同一次网络读取可能包含多条完整SSE事件，client时间戳记录实际解析/交付时刻及receive-batch ID；这些接近零的ack间隔反映传输合并与客户端解析，不人为均匀摊开，也不解释为GPU计算时间。TTFT在HTTP调用前起计，连接池等待也属于端到端；连接建立是否预热需固定。聚合JSON解码/缓冲区flush的开销通过collector校准与loop lag暴露。

## 10. 重复、聚合、失败与停止规则

每核心cell5次独立run，trace seed配对；初次tail样本不足则按预注册增加N并对所有策略重做相同设计。pilot与formal目录隔离，不把pilot中最好的数并进formal。每个run独立算P50/P95/P99，再报告5次中位和IQR/MAD；同时给运行级配对bootstrap CI。完整raw data用于审计，不能先pool所有请求再把5次重复解释为单次稳定尾部。

追加到10次的规则：主改善方向不稳定、CI跨0或主要指标run间CV>5%，在正式前预注册；保留前5次。热不稳、后台GPU任务或client lag超标属于污染标记，标明原因、原始结果与替代run，报告包含/排除该类的敏感性；性能差不是排除理由。

每配置包括：attempted_runs、valid_runs、OOM_runs、allocator_assert_runs、timeout_rate、unresolved_rate、cold/warm条件、trace hash、实际显存、机制激活比例。不用成功run的均值掩盖OOM，发生系统性失败的配置判为不适用于8GB正式结论。所有指标的比较组key必须一致，聚合脚本拒绝跨key计算百分比。

目录示意：

```text
results/
  environment/                 # 硬件、OS、版本、源码/模型manifest
  traces/                      # 不变请求、prime、SLO和hash
  pilot/                       # rate/参数选择证据
  W4/rate-X/S1010/seed-31723/
    run.json                   # flags、clock域、code hash、状态与边界
    requests.jsonl             # 每个offered请求的终态、到达/发送/文本时间
    tokens.bin                 # uid/token index/ID/后端ready timestamp
    scheduler.bin              # batch/chunk/预算/拒绝/计数（有界缓冲）
    telemetry.csv
    server.stdout.log
    server.stderr.log
  report/                      # 聚合CSV、失败台账、SVG/PNG与结果说明
```

## 11. 下阶段实施边界

可以授权下一阶段新增薄benchmark工具和观测hook，但仍无需重构调度器。A关闭adapter、A-ISO/V0010等源码变体是独立任务，必须有旧路径语义与正确性证据；当前默认关闭策略不足以构造原版。page>1容量缺口本阶段仅报告，正式主组page1；若未来修复引擎，记录新commit并重新建立可比基线，不把修复前后数据混进同一cell。
