# E06完整组合退化的证据与解释

这是对已归档数据的事后诊断，未改变冻结测量，也未追加参数搜索。
[25次运行的逐次机制数据](report/e06-mechanism-diagnostics.json)来自H*、FCFS、
Aging、C+D和Full各5个正式seed的measurement.pkl及请求记录。

原实验机可用保留的原始数据复算：

```bash
python3 benchmarks/serving/diagnose_e06.py \
  --results /home/wangwei/ai-infra/Promotion/benchmark-results \
  --output /tmp/e06-diagnostics.json
```

从仓库根目录执行；复算仅读取本地记录，不启动GPU。本次已验证25/25运行的
输出与归档机制数据一致。

## 消融定位

| 配置 | 成功输出吞吐中位数token/s | 入选运行数 |
| --- | --- | --- |
| H* | 2025.77 | 5 |
| FCFS默认实现 | 2025.67 | 5 |
| Aging | 2024.89 | 5 |
| C+D，关闭Aging | 393.77 | 5 |
| Full | 402.08 | 5 |

严重退化出现在Decode优先与小Prefill动态预算组合启用后。C64和B+C的运行
全部未通过客户端质量门槛，不使用这些运行发布正式C/D边际效应或交互CI。
现有证据不能把总损失精确拆成C、D各自的独立贡献。

## 超时之前已经发生积压

以下范围来自5次运行的前60秒，此时尚未发生超时Abort；长请求分块数另以
每轮完整成功长请求为条件，分别求中位数。

| 观测 | H* | Full |
| --- | --- | --- |
| 有Decode竞争时Prefill实际token数中位数 | 256 | 28 |
| 前60秒平均Decode batch大小，各run范围 | 13.8–15.4 | 3.5–3.9 |
| 第60秒附近pending队列，各run范围 | 0–20 | 392–450 |
| 成功长请求Prefill分块数中位数，每run | 8 | 65 |

源码中的动态预算为`min(64,max(8,32-n_decode))`，只用于有Decode竞争时的
独立Prefill batch；不是融合Prefill/Decode，也不是KV容量限制。配置K=4，
通常连续提交4个Decode batch后才尝试一次保护性Prefill；无Decode时可用普通256预算。

观测支持如下机制解释：小Prefill块与Decode优先减慢长输入完成速度，新请求
较晚进入Decode，Decode batch变小，批次执行与准备开销更难摊薄，进而持续排队。
仅改变chunk造成的kernel耗时占比没有独立剖析，不能把总退化全部归因于某个kernel。

## 缓存和Aging未抵消该代价

H*和Full的prefix token命中率都约4.9%，无稳定提升。新队列超过128时，
LPM-Aging按源码退回到达顺序；已有chunk优先继续，因此无法修复整体输入推进不足。
实际调度源码位于所附`environment/sources/`中的冻结S快照，配置见frozen-config.json。

## 指标为何一起恶化

H*每轮成功请求中位数1000、完整测量时长104.26秒；Full分别为424、220.38秒。
成功输出token中位数211200→89728，分子变小且分母增大，成功吞吐显著下降。
大量完成请求仍超过固定TTFT/E2E门槛，使Goodput进一步下降。

ITL观测下降反映成功子集中Decode交付更连续，不代表所有请求获得更好服务；
超时请求不进入成功延迟分位数，且全局ITL校准失败。
调度决策累计CPU时间约0.44→2.3秒，单独不足以解释约80%的吞吐下降。
该结论针对本机、固定G1资源、W4-X负载及这组参数，不外推为所有Decode优先策略无效。
