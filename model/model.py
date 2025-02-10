import math
import struct
import inspect
import time

from .LMConfig import LMConfig
from typing import Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn.utils.rnn import pad_sequence, unpad_sequence # 从 torch.nn.utils.rnn 导入用于处理 RNN 序列的工具，包括序列填充和解填充
from .label_smoothing_loss import LabelSmoothingLoss # 从 transformer.label_smoothing_loss 导入 LabelSmoothingLoss 类，用于实现标签平滑损失函数

IGNORE_ID = -1


def th_accuracy(pad_outputs: torch.Tensor, pad_targets: torch.Tensor,
                ignore_label: int) -> torch.Tensor:
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        torch.Tensor: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(pad_targets.size(0), pad_targets.size(1),
                                pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return (numerator / denominator).detach()


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_pos_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis


def apply_rotary_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: LMConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.k_cache, self.v_cache = None, None
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, pos_cis: torch.Tensor, kv_cache=False):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        # 更高效的kv_cache实现
        if kv_cache and self.eval():
            if seqlen == 1 and all(cache is not None for cache in (self.k_cache, self.v_cache)):
                xk = torch.cat((self.k_cache, xk), dim=1)
                xv = torch.cat((self.v_cache, xv), dim=1)
            self.k_cache, self.v_cache = xk, xv

        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.flash and seqlen != 1:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None,
                                                                      dropout_p=self.dropout if self.training else 0.0,
                                                                      is_causal=True)
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores = scores + self.mask[:, :, :seqlen, :seqlen]  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoEGate(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape

        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(
                dim=config.dim,
                hidden_dim=config.hidden_dim,
                multiple_of=config.multiple_of,
                dropout=config.dropout,
            )
            for _ in range(config.n_routed_experts)
        ])

        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(
                dim=config.dim,
                hidden_dim=config.hidden_dim,
                multiple_of=config.multiple_of,
                dropout=config.dropout,
            )

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape

        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)

        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            # 训练模式下，重复输入数据
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理模式下，只选择最优专家
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)

        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 例如当tokens_per_expert=[6, 15, 20, 26, 33, 38, 46, 52]
        # 当token_idxs=[3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...]
        # 意味着当token_idxs[:6] -> [3,  7, 19, 21, 24, 25,  4]位置的token都由专家0处理，token_idxs[6:15]位置的token都由专家1处理......
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 使用 scatter_add_ 进行 sum 操作
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: LMConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        if args.use_moe:
            self.feed_forward = MOEFeedForward(args)
        else:
            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=args.hidden_dim,
                multiple_of=args.multiple_of,
                dropout=args.dropout,
            )

    def forward(self, x, pos_cis, kv_cache=False):
        h = x + self.attention(self.attention_norm(x), pos_cis, kv_cache)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(PreTrainedModel):
    config_class = LMConfig
    last_loss: Optional[torch.Tensor]
    last_acc: Optional[torch.Tensor]

    def __init__(self, params: LMConfig = None):
        super().__init__(params)
        if not params:
            params = LMConfig()
        self.params = params
        self.vocab_size = params.vocab_size
        self.speechcab_size = params.speechcab_size
        self.n_layers = params.n_layers

        self.sos_eos = 0 # 定义起始符 (SOS) 和 结束符 (EOS) token 的索引，通常为 0
        self.task_id = 1 # 定义任务 ID token 的索引，例如用于区分不同任务，通常为 1
        self.llm_embedding = torch.nn.Embedding(3, params.dim) # 创建用于 SOS/EOS 和任务 ID token 的嵌入层 # (词汇表大小：2，嵌入维度：llm_input_size)

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.speech_embeddings = nn.Embedding(params.speechcab_size, params.dim)

        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.speechcab_size + 1, bias=False)  # 这里加的1是为了后续计算loss时的维度对齐

        pos_cis = precompute_pos_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("pos_cis", pos_cis, persistent=False)

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers))

        self.criterion_ce = LabelSmoothingLoss( # 初始化标签平滑交叉熵损失函数
            size = params.speechcab_size + 1, # 损失计算的词汇表大小 (语音 token + EOS + 对齐token)
            padding_idx=IGNORE_ID, # padding token 的索引，在损失计算中会被忽略
            smoothing = 0.0, # 标签平滑权重
            normalize_length = True, # 是否按序列长度标准化损失
        )

        self.last_loss = None
        self.last_acc = None
        self.OUT = CausalLMOutputWithPast()
        self._no_split_modules = [name for name, _ in self.named_modules()]

        self.is_first = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def pad_unpad_sequence(self, sos_emb, text_token, text_token_len, task_id_emb, speech_token, speech_token_len, eos_emb): # 定义函数 pad_unpad_sequence，用于处理序列的填充和解填充
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True) # 解填充文本 token 序列，根据 text_token_len 移除 padding，batch_first=True 表示 batch 维度在第一维
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True) # 解填充语音 token 序列，根据 speech_token_len 移除 padding，batch_first=True 表示 batch 维度在第一维
        lm_input = [torch.concat([sos_emb.squeeze(dim=0), text_token[i], task_id_emb.squeeze(dim=0), speech_token[i], eos_emb.squeeze(dim=0)], dim=0) # 对每个样本，将 SOS/EOS 嵌入、说话人嵌入、文本 token、任务 ID 嵌入和语音 token 拼接起来
                    for i in range(len(text_token))] # 遍历 batch 中的每个样本
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32) # 计算每个拼接后的序列的长度
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID) # 对拼接后的序列进行填充，使其长度一致，padding 值使用 IGNORE_ID
        return lm_input, lm_input_len # 返回填充后的输入序列和对应的长度

    def forward(self, text_tokens: Optional[torch.Tensor] = None, text_token_len: Optional[torch.Tensor] = None,
                speech_tokens: Optional[torch.Tensor] = None, speech_token_len: Optional[torch.Tensor] = None,
                kv_cache=False, **keyargs):
        
        current_idx = 0
        if 'input_ids' in keyargs:
            text_tokens = keyargs['input_ids']
        if 'attention_mask' in keyargs:
            speech_tokens = keyargs['attention_mask']
        if 'current_idx' in keyargs:
            current_idx = int(keyargs['current_idx'])

        # 1. 准备 llm_target (语言模型的目标输出)
        lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_tokens[i, :speech_token_len[i]].tolist() + [self.speechcab_size]) 
                     for i in range(text_tokens.size(0))] # 遍历 batch 中的每个样本# 为每个样本生成目标序列，包含 padding, 文本 token 长度的 padding, 语音 token 和 EOS 、taskid token
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID) # 对目标序列进行填充，使其长度一致，并移动到指定设备        
        lm_target = lm_target.to(text_tokens.device) # 将目标序列移动到与输入相同的设备
       

        # 1. 将输入的文本和语音token分别嵌入
        text_tokens = self.tok_embeddings(text_tokens)
        text_tokens = self.dropout(text_tokens)
        speech_tokens = self.speech_embeddings(speech_tokens)
        speech_tokens = self.dropout(speech_tokens)

        # 2. eos (SOS/EOS) 和 task_id 嵌入
        sos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1) # 获取 SOS token 的嵌入，并reshape 成 (1, 1, D) 形状
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1) # 获取 task_id token 的嵌入，并reshape 成 (1, 1, D) 形状
        eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1) # 获取 EOS token 的嵌入，并reshape 成 (1, 1, D) 形状

        # 3. 解填充和填充序列
        lm_input, lm_input_len = self.pad_unpad_sequence(sos_emb, text_tokens, text_token_len, # 调用 pad_unpad_sequence 函数处理输入序列，拼接各种嵌入并填充
                                                         task_id_emb, speech_tokens, speech_token_len, eos_emb)

        pos_cis = self.pos_cis[current_idx:current_idx + lm_input_len.max()]
        for idx, layer in enumerate(self.layers):
            lm_input = layer(lm_input, pos_cis, kv_cache)

        lm_input = self.norm(lm_input)

        if speech_tokens is not None:
            logits = self.output(lm_input)
            self.last_loss = self.criterion_ce(logits, lm_target) # 计算交叉熵损失
            self.last_acc = th_accuracy(logits.view(-1, self.speechcab_size + 1), lm_target, ignore_label=IGNORE_ID) # 计算准确率

        else:
            logits = self.output(lm_input[:, [-1], :])
            self.last_loss = None
            self.last_acc = None

        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        self.OUT.__setitem__('last_acc', self.last_acc)
        return self.OUT

    @torch.inference_mode()
    def generate(self, idx,  max_new_tokens, temperature=0.7, top_k=8, stream=True, rp=1., kv_cache=True):
        """
        Generate tokens autoregressively using the Transformer model.

        Args:
            idx (torch.Tensor): Input tensor of shape (seq_len), containing token indices.
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature for controlling randomness.
            top_k (int): Top-k sampling parameter, the number of highest-probability logits to consider.
            stream (bool): Whether to yield tokens as they're generated (streamed output).
            rp (float): Repetition penalty to discourage repeating tokens (>= 1.0 means no penalty).
            kv_cache (bool): Whether to use key-value caching for faster generation.

        Returns:
            torch.Tensor: Output tensor of generated tokens.
        """
        generated = idx.unsqueeze(0)  # Add batch dimension

        current_idx = 0
        pos_cis = self.pos_cis[current_idx:current_idx + generated.size(1)]

        for _ in range(max_new_tokens):
            if self.is_first:
                sos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
                task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
                text_tokens = self.tok_embeddings(generated)
                llm_input = torch.cat((sos_emb, text_tokens, task_id_emb), dim=1)
                self.is_first = False
            else:
                llm_input = self.tok_embeddings(generated)

            llm_input = self.norm(llm_input)
            logits = self.output(llm_input[:, [-1], :])

            logits = logits / temperature

            if top_k > 0:
                top_logits, top_indices = torch.topk(logits, k=top_k, dim=-1)
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(dim=-1, index=top_indices, src=top_logits)

            if rp != 1.0:
                logits[0, generated[-1]] /= rp

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(torch.squeeze(probs, dim=0), num_samples=1)

            if next_token.item() == 4096:
                break

            generated = torch.cat((generated, next_token), dim=1)
            current_idx += 1
            if current_idx < self.pos_cis.size(0):
                pos_cis = self.pos_cis[current_idx:current_idx + 1]

        return generated[:,len(idx)+1:]