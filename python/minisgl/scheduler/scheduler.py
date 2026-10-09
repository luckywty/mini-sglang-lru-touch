from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        # self.config = config

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""


        #若所有 Worker 同时打印等待日志，
        # 会导致终端刷屏与 I/O 竞争。info_rank0 确保仅在主卡（Rank 0）输出日志，保持日志整洁。
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()#用于校验 KV Cache 显存池与前缀缓存树（Radix Cache）的状态一致性

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        将 CPU 端的调度、IPC 通信与后处理开销隐藏在 GPU 计算的执行时间内
        **重叠调度与执行主循环**
        该机制将当前批次的计算执行与上一批次的结果处理进行重叠，
        能够有效掩盖 CPU 延迟，提升 GPU 利用率。
        """



        """
        仅当系统完全没有工作可做（真正处于空闲状态）时，
        blocking 为 True，当前线程让出 CPU 等待前端新消息传入，
        避免自旋消耗 100% CPU
        """
        blocking = not (
            #last_data:上一轮循环已经提交/完成的那个 Batch 的 forward 数据，现在轮到 Scheduler 处理它的结果了
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )

        #把请求放进 Scheduler 管理的数据结构
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        #self._schedule_next_batch()的策略：Prefill 优先，Prefill 没有可调度 Batch 时才尝试 Decode
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream


                #Engine stream 在执行 forward 之前，要等待 Scheduler stream 上此前提交的操作完成
                self.engine.stream.wait_stream(self.stream)



                ongoing_data = (
                    forward_input, 
                    self._forward(forward_input)
                    )

        self._process_last_data(last_data)
        return ongoing_data

    def normal_loop(self) -> None:  ##经典的停等式（Stop-and-Wait）串行逻辑
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)



    """不仅关闭了梯度计算，还禁用了所有 Tensor 的版本计数器（Version Counter）跟踪与
    视图跟踪（View Tracking），使构造出来的张量具有更低的 C++ 运行时开销和内存占用


    NoReturn：来自 typing 库，
    显式标注此函数正常情况下永远不会退出
    （内部包含无条件的 while True 死循环）。
    """
    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            #通过环境变量 ENV.DISABLE_OVERLAP_SCHEDULING 将执行逻辑严格隔离为两个分支
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        #安全退出与资源级联释放（
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:


        """
            [last_data 解包]
            │
            ▼
        [copy_done.synchronize()]  <--- 等待 GPU 到 CPU 的采样结果异步拷贝完成
            │
            ▼
        [with lazy_free_region()]  <--- 开启批量显存释放上下文
            │
            ├──> 跳过 ChunkedReq (分块未完成的 Prompt 请求不产出 Token)
            ├──> 更新 Token 序列: req.append_host(...)
            ├──> 终止判定: 达到最大长度 (not req.can_decode) 或命中 EOS
            ├──> 打包 DetokenizeMsg
            │
            └──> 状态流转与资源回收:
                    ├── 若 finished: 从 decode_manager 移除并释放槽位/KV 页
                    └── 若 prefill 完成: 写入 Radix 前缀缓存 (cache_req)
            │
            ▼
        [send_result(reply)]       <--- IPC 发送生成结果给下游 Detokenizer
        
        """
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        #负责消费来自前端 API 服务或 IPC 管道的各类控制消息，
        # 完成请求的校验过滤、入队分发、异常中断以及进程退出
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)#释放资源并不等于直接销毁 KV Cache，而是将其转换为可复用的前缀缓存（Radix Cache）

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)               #CUDA Graph 尺寸对齐与补齐
        self.cache_manager.allocate_paged(batch.reqs)           #动态申请 Paged KV Cache 物理显存页
        batch.positions = _make_positions(batch, self.device)   #生成 RoPE / 绝对位置索引张量
        input_mapping = _make_input_tuple(batch, self.device)   #生成读写 token_pool 的二维坐标 (table_idx, pos)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]   #确定当前 token 物理 KV 写入位置 out_loc
        self.engine.attn_backend.prepare_metadata(batch)        #构建 Attention Backend (FlashInfer等) 底层元数据
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    """_make_positions 负责为当前批次中所有需要计算的 Token 
    生成一维展平的绝对位置编号（Position IDs），
    主要提供给模型的旋转位置编码（RoPE）以及注意力机制定位使用"""
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:#构造 Token 级别的“读与寻址”坐标
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:#构造 Request 级别的“回写”坐标
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
