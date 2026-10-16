# RTX 5060 Ti 8GB mini-SGLang 实测归档

实验于北京时间2026-10-16 09:39完成，最终报告于09:43生成。Qwen3-0.6B、BF16、
TP1、FlashInfer/fa2，固定page1、KV16384、context2304、running16、普通Prefill256
和CUDA Graph列表[1,2,4,8,16]。

共7类实验、180个seed×配置单元、260次正式尝试，157次满足记录/资源与运行质量
筛选规则。157不是“所有请求均成功”的运行数：超时是保留的实验结果。

## 阅读入口

- [完整报告，包含11张图及样本数](REPORT.md)
- [HTML版本](report/index.html)：下载整个目录后用浏览器打开，GitHub通常显示HTML源码。
- [逐次CSV](report/runs.csv)、[完整逐次JSON](report/runs.json)
- [运行间中位数/IQR/MAD](report/aggregates.json)、[配对置信区间](report/paired-tradeoffs.json)
- [配置覆盖与入选数](report/coverage.json)、[失败与错误台账](report/error-ledger.json)
- [冻结配置](frozen-config.json)、[模型与源码校验信息](environment/)
- [共同正确性与观测补丁](patches/)、[简历证据边界](RESUME_EVIDENCE.md)
- [E06退化机制与观测证据](E06_DIAGNOSIS.md)
- [实验工具及原实验机复现命令](../../benchmarks/serving/README.md)

## 主要结果

变化率是同trace配对运行变化的中位数，不是两列总体中位数相除；只在同一实验内比较。
Goodput按固定客户端TTFT与E2E门槛计算，未进行多重比较校正。

| 对照 | 指标 | 配对变化中位数及95%区间 |
| --- | --- | --- |
| E04 FCFS→LPM，8对 | Goodput | +35.01% [+2.23%,+134.86%] |
| E04 FCFS→LPM，8对 | 成功输出吞吐 | +1.46% [-0.07%,+4.12%]，方向不稳定 |
| E05 固定24→动态预算，10对 | Goodput | +24.18% [+22.61%,+26.43%] |
| E05 固定24→动态预算，10对 | 长请求TTFT P95 | 降低2.78% [1.95%,3.26%] |
| E05 固定64→动态预算，10对 | Goodput | -49.22% [-49.42%,-48.40%] |
| E06 H*→Full，5对 | 成功输出吞吐 | -79.90% [-80.95%,-79.20%] |
| E06 H*→Full，5对 | Goodput | -99.49% [-99.72%,-98.94%] |

**没有稳定的整体吞吐提升结论。** E06 Full每轮1000请求中561–595个超时。
E01默认实现吞吐基本持平；E03没有稳定的Aging公平性改善。E02及E02-low的C64
没有足够质量合格配对，不发布该主对照收益。

## 指标的实际定义

- **成功输出吞吐（token/s）**：完整成功且通过输入/输出/终态对账请求的输出token总数，
  除以正式回放开始至最后客户端终止的完整时长。失败请求的部分输出不算入分子，
  超时等待仍留在分母。启动、预热、prime和温度等待不在该测量时长内。
- **Goodput（请求/s）**：完整成功且同时满足TTFT和E2E门槛的请求数，除以上述时长。
  TTFT门槛为long类2000ms、其他类500ms；E2E门槛为
  `对应TTFT门槛 + 50ms×(输出token数−1) + 500ms`。
  50ms是总耗时预算系数，不表示每个token间隔都必须小于50ms。
- **TTFT-text**：客户端调用HTTP前记录的发送时刻，到首个非空文本块到达的时间。
  它包含服务端排队、Prefill、传输及文本交付；从计划到达到发送的偏差另计。
- **ITL P99**：同一成功请求相邻token时间间隔汇总后的第99百分位；
  服务端时间是既有D2H后的host emit，客户端时间是SSE ACK到达，分别保存。
- **TPOT**：每个成功请求的首末token跨度除以间隔数，再在请求间计算分位数，
  与逐token ITL的分位数不同。
- **TTFT/ITL的成功子集边界**：这些分位数不把超时请求当作正常完成样本。
  必须同时读取超时、未准入及失败数；仅成功样本的延迟降低不能证明整体改善。

## 验收边界

- H*是固定官方上游加双方共有正确性/观测/API补丁，不能称原样上游。
- 尾延迟采集和亲和性校准未通过；所有ITL变化仅为探索性，不写成已验证收益。
- 严格跨形状greedy一致性及最初HF绝对误差门槛未通过。
  相同chunk下H/S logits逐位一致，不等于跨形状质量无损。
- 自然问题2+2在H*/S和相同输入的HF BF16参考均输出2，语义验收未通过。
- A无独立关闭实现，B依赖A、D依赖C；不填造独立消融或未测GPU边界。

## Git发布范围与原始数据

本目录提供统计、260份正式run记录、可用summary、图表、失败验收、trace manifests、
冻结源码及测量工具快照。原始测量和配置保持不变；机器绝对路径是当时环境的记录。

完整原始数据保存在原实验机：`/home/wangwei/ai-infra/Promotion/benchmark-results`。
其中measure约5.7GB、完整trace约368MB，模型权重另存；这些大文件没有上传Git。
因此本发布包可以审阅统计和来源，但不等同于全部原始token/请求数据集。

- [原始证据SHA256](report/artifact-sha256.json)：对照本地原始文件时使用。
- [导出来源及逐文件校验](export-manifest.json)：区分复制证据与未上传数据。
- [本地实验提交历史bundle](environment/local-benchmark-history.bundle)：保存原始
  实验分支截至50031e6的完整增量历史，前提提交为7cd28c7。通过授权GitHub连接
  发布的主分支提交可能与本地提交SHA不同；bundle保留冻结配置引用的原始提交。
- `SHA256SUMS`：校验本发布目录文件，可在此目录执行`sha256sum --check --quiet SHA256SUMS`。
- 在原实验机使用保留的原始数据，可重新运行分析与最终报告工具。
  在其他机器上需修改工具中的ROOT/MODEL/PY和env.sh，准备相同模型与依赖，
  然后建立新实验目录并重新生成manifest；不沿用旧结果身份或将新数据混入本归档。

需要查看原始本地实验历史时，在克隆的仓库中执行：

```bash
git bundle verify results/rtx5060ti-20261016/environment/local-benchmark-history.bundle
git fetch results/rtx5060ti-20261016/environment/local-benchmark-history.bundle \
  refs/heads/benchmark/rtx5060ti-20261014:refs/remotes/archive/rtx5060ti-20261014
```

## 发布验证

加载`benchmarks/serving/env.sh`后，CPU回归261项通过，耗时8.82秒；日志见
[publication validation](preflight/github-publication-validation.txt)。
首次复验漏载环境，PATH中找不到ninja，119项失败；该次
[完整失败日志](preflight/github-publication-validation-unconfigured-env.txt)保留，
环境修正后重新执行全部261项。这不是正式GPU测量数据，也没有替换正式运行。
此前6组GPU正确性/资源验收和失败门槛见`preflight/`。
11张最终PNG均已逐图检查，F4修复了坐标刻度重叠和长标签；绘图数据未变。
[渲染检查记录](report/render-review.json)及[仅展示调整记录](report/publication-render-manifest.json)
与原始统计分开保存。HTML的本地链接已检查，浏览器版式未另外验收。
