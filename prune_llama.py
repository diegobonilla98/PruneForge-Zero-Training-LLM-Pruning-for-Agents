import argparse
import gc
import json
import math
import random
import re
import time as time_module
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


SEQ_LEN = 256
GEN_BATCH_MULTIPLIER = 2
EMBED_BATCH_MULTIPLIER = 2
DEBUG_LOG_ENABLED = True
DEBUG_LOG_PATH = "pruning_debug.jsonl"
AUTO_SAVE_RECIPE = True
DEFAULT_RECIPE_PATH = "pruning_recipe.json"
TAU_DECAY = 0.94
TAU_FLOOR_REL = 0.50
STAGE_MAX_ITERS = 30
MLP_DIVISORS = (2, 4, 8)
ADAPT_TARGET_REDUCTION = 0.60
ADAPT_TAU_PUSH = 0.45
ADAPT_EPS_SCALE = 2.0
ADAPT_EPS_MAX = 0.18
ADAPT_POOL_SCALE = 1.5
ADAPT_POOL_MAX = 32
ADAPT_STAGE_TRIALS_SCALE = 0.7
ADAPT_MIN_STAGE_TRIALS = 3
ADAPT_FLOOR_PUSH = 0.25
ADAPT_FLOOR_MIN_REL = 0.35
ADAPT_HEADROOM_WEIGHT = 0.8


def log_event(event: Dict[str, object]) -> None:
    if not DEBUG_LOG_ENABLED:
        return
    payload = dict(event)
    payload["time"] = time_module.time()
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def read_lines(path: str) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def batched(items: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield list(items[i : i + batch_size])


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"```[\w]*\n?", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("Unsupported model structure: cannot find decoder layers.")


def reset_layer_indices(model: torch.nn.Module) -> None:
    layers = get_decoder_layers(model)
    for idx, layer in enumerate(layers):
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = idx
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = idx


def ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def build_prompt(
    tokenizer: AutoTokenizer, system_prompt: str, user_text: str
) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"{system_prompt}\n\n{user_text}\n\nAnswer:"


class LlamaGenerator:
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        device: str,
        dtype: torch.dtype,
        batch_size: int,
        max_new_tokens: int,
    ) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, padding_side="left"
        )
        ensure_pad_token(self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        )
        self.model.to(device)
        self.model.eval()
        self.model.config.use_cache = False
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = False

    def build_prompts(self, questions: Sequence[str]) -> List[str]:
        return [
            build_prompt(self.tokenizer, self.system_prompt, question)
            for question in questions
        ]

    def _tokenize_prompts(self, prompts: Sequence[str]) -> Dict[str, Tensor]:
        max_input_len = SEQ_LEN - self.max_new_tokens
        if max_input_len < 1:
            raise ValueError("max_new_tokens is too large for SEQ_LEN=256.")
        batch = self.tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            max_length=max_input_len,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in batch.items()}

    def _tokenize_texts(
        self, texts: Sequence[str], max_length: int = SEQ_LEN
    ) -> Dict[str, Tensor]:
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in batch.items()}

    @torch.no_grad()
    def generate(self, prompts: Sequence[str], desc: str = "Generating") -> List[str]:
        outputs: List[str] = []
        batches = list(batched(list(prompts), self.batch_size))
        for batch_prompts in tqdm(batches, desc=desc, leave=False):
            inputs = self._tokenize_prompts(batch_prompts)
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                top_p=1.0,
                num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False,
            )
            prompt_len = inputs["input_ids"].shape[1]
            gen_tokens = gen_ids[:, prompt_len:]
            decoded = self.tokenizer.batch_decode(
                gen_tokens, skip_special_tokens=True
            )
            outputs.extend([normalize_text(text) for text in decoded])
        return outputs

    @torch.no_grad()
    def forward_for_stats_texts(
        self, texts: Sequence[str], max_length: int = SEQ_LEN, desc: str = "Forward pass"
    ) -> None:
        batches = list(batched(list(texts), self.batch_size))
        for batch_texts in tqdm(batches, desc=desc, leave=False):
            inputs = self._tokenize_texts(batch_texts, max_length=max_length)
            _ = self.model(**inputs, use_cache=False)


class EmbeddingScorer:
    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
        task_description: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.task_description = task_description

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left"
        )
        ensure_pad_token(self.tokenizer)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def _prepare_texts(self, texts: Sequence[str], as_query: bool) -> List[str]:
        if as_query and self.task_description:
            return [
                get_detailed_instruct(self.task_description, text) for text in texts
            ]
        return list(texts)

    @torch.no_grad()
    def encode(
        self, texts: Sequence[str], as_query: bool = False, desc: str = "Encoding"
    ) -> Tensor:
        prepared = self._prepare_texts(texts, as_query)
        all_embs: List[Tensor] = []
        batches = list(batched(prepared, self.batch_size))
        for batch_texts in tqdm(batches, desc=desc, leave=False):
            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=SEQ_LEN,
                return_tensors="pt",
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embs.append(embeddings.detach().cpu())
        return torch.cat(all_embs, dim=0)

    def similarity(
        self, a_texts: Sequence[str], b_texts: Sequence[str], as_query: bool = False
    ) -> float:
        if len(a_texts) != len(b_texts):
            raise ValueError("a_texts and b_texts must have the same length.")
        a_emb = self.encode(a_texts, as_query=as_query)
        b_emb = self.encode(b_texts, as_query=as_query)
        scores = (a_emb * b_emb).sum(dim=1)
        return scores.mean().item()

class StatsCollector:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.mlp_sum_abs: Dict[int, Tensor] = {}
        self.mlp_token_count: Dict[int, int] = {}
        self.head_sum_norm: Dict[int, Tensor] = {}
        self.head_token_count: Dict[int, int] = {}
        self.layer_sum_norm: Dict[int, float] = {}
        self.layer_token_count: Dict[int, int] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def register(self) -> None:
        layers = get_decoder_layers(self.model)
        for idx, layer in enumerate(layers):
            self._register_mlp_hook(idx, layer)
            self._register_attn_hook(idx, layer)
            self._register_layer_hook(idx, layer)

    def _register_mlp_hook(self, layer_idx: int, layer: torch.nn.Module) -> None:
        def hook(_module, inputs, _output):
            with torch.no_grad():
                z = inputs[0].detach().float()
                if z.dim() == 3:
                    z_sum = z.abs().sum(dim=(0, 1))
                    count = z.shape[0] * z.shape[1]
                else:
                    z_sum = z.abs().sum(dim=0)
                    count = z.shape[0]
                if layer_idx not in self.mlp_sum_abs:
                    self.mlp_sum_abs[layer_idx] = z_sum.cpu()
                    self.mlp_token_count[layer_idx] = count
                else:
                    self.mlp_sum_abs[layer_idx] += z_sum.cpu()
                    self.mlp_token_count[layer_idx] += count

        handle = layer.mlp.down_proj.register_forward_hook(hook)
        self.handles.append(handle)

    def _register_attn_hook(self, layer_idx: int, layer: torch.nn.Module) -> None:
        config = self.model.config
        attn = layer.self_attn
        num_heads_val = getattr(attn, "num_heads", None) or config.num_attention_heads
        head_dim_val = getattr(attn, "head_dim", None) or (config.hidden_size // num_heads_val)

        def hook(_module, inputs, _output, num_heads=num_heads_val, head_dim=head_dim_val):
            with torch.no_grad():
                h = inputs[0].detach().float()
                if h.dim() != 3:
                    return
                bsz, seq_len, total_dim = h.shape
                if head_dim * num_heads != total_dim:
                    return
                o_w = layer.self_attn.o_proj.weight.detach().float()
                h = h.view(bsz, seq_len, num_heads, head_dim)
                head_norms = []
                for j in range(num_heads):
                    o_j = h[:, :, j, :]
                    w_j = o_w[:, j * head_dim : (j + 1) * head_dim]
                    contrib = torch.einsum("bsh,oh->bso", o_j, w_j)
                    contrib_norm = torch.linalg.vector_norm(contrib, dim=-1)
                    head_norms.append(contrib_norm.sum().item())
                h_sum = torch.tensor(head_norms, dtype=torch.float32)
                count = bsz * seq_len
                if layer_idx not in self.head_sum_norm:
                    self.head_sum_norm[layer_idx] = h_sum
                    self.head_token_count[layer_idx] = count
                else:
                    self.head_sum_norm[layer_idx] += h_sum
                    self.head_token_count[layer_idx] += count

        handle = layer.self_attn.o_proj.register_forward_hook(hook)
        self.handles.append(handle)

    def _register_layer_hook(self, layer_idx: int, layer: torch.nn.Module) -> None:
        def hook(_module, inputs, output):
            with torch.no_grad():
                inp = inputs[0].detach().float()
                out = output[0] if isinstance(output, (tuple, list)) else output
                out = out.detach().float()
                if inp.shape != out.shape:
                    return
                delta = out - inp
                delta_norm = torch.linalg.vector_norm(delta, dim=-1)
                sum_norm = delta_norm.sum().item()
                count = delta_norm.numel()
                if layer_idx not in self.layer_sum_norm:
                    self.layer_sum_norm[layer_idx] = sum_norm
                    self.layer_token_count[layer_idx] = count
                else:
                    self.layer_sum_norm[layer_idx] += sum_norm
                    self.layer_token_count[layer_idx] += count

        handle = layer.register_forward_hook(hook)
        self.handles.append(handle)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def compute_importances(self) -> Tuple[Dict[int, Tensor], Dict[int, Tensor], Dict[int, float]]:
        mlp_importance: Dict[int, Tensor] = {}
        head_importance: Dict[int, Tensor] = {}
        layer_importance: Dict[int, float] = {}
        layers = get_decoder_layers(self.model)

        for idx, layer in enumerate(layers):
            if idx in self.mlp_sum_abs:
                mean_abs = self.mlp_sum_abs[idx] / max(self.mlp_token_count[idx], 1)
                w2 = layer.mlp.down_proj.weight.detach().float().cpu()
                col_norm = torch.linalg.vector_norm(w2, dim=0)
                mlp_importance[idx] = mean_abs * col_norm

            if idx in self.head_sum_norm:
                head_importance[idx] = self.head_sum_norm[idx] / max(self.head_token_count[idx], 1)

            if idx in self.layer_sum_norm:
                layer_importance[idx] = self.layer_sum_norm[idx] / max(
                    self.layer_token_count[idx], 1
                )

        return mlp_importance, head_importance, layer_importance


def collect_importances(
    generator: LlamaGenerator, texts: Sequence[str], desc: str = "Collecting stats"
) -> Tuple[Dict[int, Tensor], Dict[int, Tensor], Dict[int, float]]:
    collector = StatsCollector(generator.model)
    collector.register()
    generator.forward_for_stats_texts(texts, max_length=SEQ_LEN, desc=desc)
    collector.remove()
    return collector.compute_importances()


def remap_importances_to_orig(
    mlp_imp: Dict[int, Tensor],
    head_imp: Dict[int, Tensor],
    layer_imp: Dict[int, float],
    layer_orig_indices: List[int],
) -> Tuple[Dict[int, Tensor], Dict[int, Tensor], Dict[int, float]]:
    remapped_mlp: Dict[int, Tensor] = {}
    remapped_head: Dict[int, Tensor] = {}
    remapped_layer: Dict[int, float] = {}
    for pos, orig_idx in enumerate(layer_orig_indices):
        if pos in mlp_imp:
            remapped_mlp[orig_idx] = mlp_imp[pos]
        if pos in head_imp:
            remapped_head[orig_idx] = head_imp[pos]
        if pos in layer_imp:
            remapped_layer[orig_idx] = layer_imp[pos]
    return remapped_mlp, remapped_head, remapped_layer


@dataclass
class HeadMap:
    query_head_orig_indices: List[int]
    kv_head_orig_indices: List[int]
    group_size: int
    head_dim: int

    def clone(self) -> "HeadMap":
        return HeadMap(
            query_head_orig_indices=list(self.query_head_orig_indices),
            kv_head_orig_indices=list(self.kv_head_orig_indices),
            group_size=self.group_size,
            head_dim=self.head_dim,
        )


@dataclass
class ModelState:
    layer_orig_indices: List[int]
    mlp_orig_indices_by_layer: Dict[int, List[int]]
    head_map_by_layer: Dict[int, HeadMap]

    @staticmethod
    def from_model(model: torch.nn.Module) -> "ModelState":
        layers = get_decoder_layers(model)
        layer_orig_indices = list(range(len(layers)))
        mlp_orig_indices_by_layer: Dict[int, List[int]] = {}
        head_map_by_layer: Dict[int, HeadMap] = {}
        config = model.config
        for idx, layer in enumerate(layers):
            d_ff = layer.mlp.down_proj.weight.shape[1]
            mlp_orig_indices_by_layer[idx] = list(range(d_ff))
            attn = layer.self_attn
            num_heads = getattr(attn, "num_heads", None) or config.num_attention_heads
            num_kv_heads = getattr(attn, "num_key_value_heads", None) or getattr(config, "num_key_value_heads", num_heads)
            head_dim = getattr(attn, "head_dim", None) or (config.hidden_size // num_heads)
            group_size = num_heads // max(num_kv_heads, 1)
            head_map_by_layer[idx] = HeadMap(
                query_head_orig_indices=list(range(num_heads)),
                kv_head_orig_indices=list(range(num_kv_heads)),
                group_size=group_size,
                head_dim=head_dim,
            )
        return ModelState(
            layer_orig_indices=layer_orig_indices,
            mlp_orig_indices_by_layer=mlp_orig_indices_by_layer,
            head_map_by_layer=head_map_by_layer,
        )


@dataclass
class Operation:
    op_type: str
    layer_orig_idx: int
    remove_score: float
    target_size: Optional[int] = None
    remove_group_orig_idx: Optional[int] = None
    keep_positions: Optional[List[int]] = None
    keep_orig_indices: Optional[List[int]] = None

    def op_id(self) -> str:
        if self.op_type == "mlp":
            return f"MLP:{self.layer_orig_idx}:{self.target_size}"
        if self.op_type == "head":
            return f"HEAD:{self.layer_orig_idx}:{self.remove_group_orig_idx}"
        return f"LAYER:{self.layer_orig_idx}"


@dataclass
class SurgeryUndo:
    revert_fn: callable

    def revert(self) -> None:
        self.revert_fn()


def replace_linear(
    old_linear: torch.nn.Linear, new_weight: Tensor, new_bias: Optional[Tensor]
) -> torch.nn.Linear:
    new_linear = torch.nn.Linear(
        new_weight.shape[1],
        new_weight.shape[0],
        bias=new_bias is not None,
    )
    new_linear.to(device=old_linear.weight.device, dtype=old_linear.weight.dtype)
    with torch.no_grad():
        new_linear.weight.copy_(new_weight.to(new_linear.weight.dtype))
        if new_bias is not None and new_linear.bias is not None:
            new_linear.bias.copy_(new_bias.to(new_linear.bias.dtype))
    return new_linear


def apply_mlp_prune(
    model: torch.nn.Module,
    state: ModelState,
    layer_orig_idx: int,
    keep_positions: List[int],
    keep_orig_indices: List[int],
) -> SurgeryUndo:
    layers = get_decoder_layers(model)
    current_idx = state.layer_orig_indices.index(layer_orig_idx)
    layer = layers[current_idx]

    old_gate = layer.mlp.gate_proj
    old_up = layer.mlp.up_proj
    old_down = layer.mlp.down_proj
    old_state = list(state.mlp_orig_indices_by_layer[layer_orig_idx])

    with torch.no_grad():
        gate_w = old_gate.weight[keep_positions, :].detach().clone()
        up_w = old_up.weight[keep_positions, :].detach().clone()
        down_w = old_down.weight[:, keep_positions].detach().clone()
        gate_b = old_gate.bias[keep_positions].detach().clone() if old_gate.bias is not None else None
        up_b = old_up.bias[keep_positions].detach().clone() if old_up.bias is not None else None
        down_b = old_down.bias.detach().clone() if old_down.bias is not None else None

    layer.mlp.gate_proj = replace_linear(old_gate, gate_w, gate_b)
    layer.mlp.up_proj = replace_linear(old_up, up_w, up_b)
    layer.mlp.down_proj = replace_linear(old_down, down_w, down_b)
    state.mlp_orig_indices_by_layer[layer_orig_idx] = list(keep_orig_indices)

    def revert() -> None:
        layer.mlp.gate_proj = old_gate
        layer.mlp.up_proj = old_up
        layer.mlp.down_proj = old_down
        state.mlp_orig_indices_by_layer[layer_orig_idx] = old_state

    return SurgeryUndo(revert_fn=revert)

def apply_head_prune(
    model: torch.nn.Module,
    state: ModelState,
    layer_orig_idx: int,
    remove_group_orig_idx: int,
) -> SurgeryUndo:
    layers = get_decoder_layers(model)
    current_idx = state.layer_orig_indices.index(layer_orig_idx)
    layer = layers[current_idx]
    attn = layer.self_attn
    config = model.config

    old_q = attn.q_proj
    old_k = attn.k_proj
    old_v = attn.v_proj
    old_o = attn.o_proj
    old_num_heads = getattr(attn, "num_heads", None) or config.num_attention_heads
    old_num_kv_heads = getattr(attn, "num_key_value_heads", None) or getattr(config, "num_key_value_heads", old_num_heads)
    old_num_kv_groups = getattr(attn, "num_key_value_groups", None) or (old_num_heads // max(old_num_kv_heads, 1))

    old_head_map = state.head_map_by_layer[layer_orig_idx].clone()

    q_orig = old_head_map.query_head_orig_indices
    kv_orig = old_head_map.kv_head_orig_indices
    group_size = old_head_map.group_size
    head_dim = old_head_map.head_dim

    remove_q_orig = set(
        range(remove_group_orig_idx * group_size, (remove_group_orig_idx + 1) * group_size)
    )
    keep_q_positions = [i for i, idx in enumerate(q_orig) if idx not in remove_q_orig]
    keep_kv_positions = [
        i for i, idx in enumerate(kv_orig) if idx != remove_group_orig_idx
    ]

    with torch.no_grad():
        q_w = old_q.weight[expand_positions(keep_q_positions, head_dim), :].detach().clone()
        k_w = old_k.weight[expand_positions(keep_kv_positions, head_dim), :].detach().clone()
        v_w = old_v.weight[expand_positions(keep_kv_positions, head_dim), :].detach().clone()
        o_w = old_o.weight[:, expand_positions(keep_q_positions, head_dim)].detach().clone()
        q_b = old_q.bias[expand_positions(keep_q_positions, head_dim)].detach().clone() if old_q.bias is not None else None
        k_b = old_k.bias[expand_positions(keep_kv_positions, head_dim)].detach().clone() if old_k.bias is not None else None
        v_b = old_v.bias[expand_positions(keep_kv_positions, head_dim)].detach().clone() if old_v.bias is not None else None
        o_b = old_o.bias.detach().clone() if old_o.bias is not None else None

    attn.q_proj = replace_linear(old_q, q_w, q_b)
    attn.k_proj = replace_linear(old_k, k_w, k_b)
    attn.v_proj = replace_linear(old_v, v_w, v_b)
    attn.o_proj = replace_linear(old_o, o_w, o_b)

    new_q_orig = [idx for idx in q_orig if idx not in remove_q_orig]
    new_kv_orig = [idx for idx in kv_orig if idx != remove_group_orig_idx]
    new_num_heads = len(new_q_orig)
    new_num_kv_heads = len(new_kv_orig)
    new_num_kv_groups = new_num_heads // max(new_num_kv_heads, 1)

    if hasattr(attn, "num_heads"):
        attn.num_heads = new_num_heads
    if hasattr(attn, "num_key_value_heads"):
        attn.num_key_value_heads = new_num_kv_heads
    if hasattr(attn, "num_key_value_groups"):
        attn.num_key_value_groups = new_num_kv_groups

    state.head_map_by_layer[layer_orig_idx] = HeadMap(
        query_head_orig_indices=new_q_orig,
        kv_head_orig_indices=new_kv_orig,
        group_size=group_size,
        head_dim=head_dim,
    )

    def revert() -> None:
        attn.q_proj = old_q
        attn.k_proj = old_k
        attn.v_proj = old_v
        attn.o_proj = old_o
        if hasattr(attn, "num_heads"):
            attn.num_heads = old_num_heads
        if hasattr(attn, "num_key_value_heads"):
            attn.num_key_value_heads = old_num_kv_heads
        if hasattr(attn, "num_key_value_groups"):
            attn.num_key_value_groups = old_num_kv_groups
        state.head_map_by_layer[layer_orig_idx] = old_head_map

    return SurgeryUndo(revert_fn=revert)


def apply_layer_drop(
    model: torch.nn.Module,
    state: ModelState,
    layer_orig_idx: int,
) -> SurgeryUndo:
    layers = get_decoder_layers(model)
    current_idx = state.layer_orig_indices.index(layer_orig_idx)
    removed_layer = layers[current_idx]
    new_layers = torch.nn.ModuleList(
        [layer for i, layer in enumerate(layers) if i != current_idx]
    )
    state.layer_orig_indices.pop(current_idx)
    if hasattr(model, "model"):
        model.model.layers = new_layers
    else:
        model.layers = new_layers
    model.config.num_hidden_layers = len(new_layers)
    reset_layer_indices(model)

    def revert() -> None:
        restored_layers = list(new_layers)
        restored_layers.insert(current_idx, removed_layer)
        restored = torch.nn.ModuleList(restored_layers)
        if hasattr(model, "model"):
            model.model.layers = restored
        else:
            model.layers = restored
        state.layer_orig_indices.insert(current_idx, layer_orig_idx)
        model.config.num_hidden_layers = len(restored)
        reset_layer_indices(model)

    return SurgeryUndo(revert_fn=revert)


def expand_positions(positions: List[int], head_dim: int) -> List[int]:
    expanded: List[int] = []
    for pos in positions:
        start = pos * head_dim
        expanded.extend(list(range(start, start + head_dim)))
    return expanded


@dataclass
class PruneConfig:
    tau: float
    epsilon: float
    temperature: float
    candidate_pool: int
    max_iters: int
    small_k: int
    holdout_k: int
    small_slack: float
    bandit_lambda: float
    sample_bandit_weight: float
    refresh_every: int
    mlp_top_keep: float
    mlp_sample_band: float
    min_mlp_size: int
    min_kv_heads: int
    min_layers: int
    max_stage_trials: int
    stage_plan: Tuple[str, ...]
    seed: int


class PruningEngine:
    def __init__(
        self,
        generator: LlamaGenerator,
        embedder: EmbeddingScorer,
        teacher_embeddings: Tensor,
        prompts: List[str],
        teacher_forcing_texts: List[str],
        importances: Tuple[Dict[int, Tensor], Dict[int, Tensor], Dict[int, float]],
        state: ModelState,
        config: PruneConfig,
    ) -> None:
        self.generator = generator
        self.embedder = embedder
        self.teacher_embeddings = teacher_embeddings
        self.prompts = prompts
        self.teacher_forcing_texts = teacher_forcing_texts
        self.mlp_importance, self.head_importance, self.layer_importance = importances
        self.state = state
        self.config = config

        self.current_score = 1.0
        self.baseline_score = 0.0
        self.tau_absolute = self.config.tau
        self.tau_floor_absolute = 0.0
        self.base_tau = self.config.tau
        self.base_epsilon = self.config.epsilon
        self.base_candidate_pool = self.config.candidate_pool
        self.base_max_stage_trials = self.config.max_stage_trials
        self.mu: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self.accepted_ops: List[Operation] = []
        self.stage_idx = 0
        self.stage_failures = 0
        self.stage_iters = 0
        self.rng = random.Random(config.seed)
        self.accepted_since_refresh = 0

        all_indices = list(range(len(self.prompts)))
        holdout_k = min(self.config.holdout_k, max(len(all_indices) - 1, 0))
        if holdout_k > 0:
            self.holdout_indices = sorted(self.rng.sample(all_indices, holdout_k))
        else:
            self.holdout_indices = []
        holdout_set = set(self.holdout_indices)
        self.train_indices = [i for i in all_indices if i not in holdout_set]

    def update_adaptive_controls(self, t: int, initial_params: int) -> None:
        if self.config.max_iters <= 0 or initial_params <= 0:
            return
        params_now = count_parameters(self.generator.model)
        reduction = 1.0 - (params_now / initial_params)
        progress = (t + 1) / self.config.max_iters
        expected = min(ADAPT_TARGET_REDUCTION, progress * ADAPT_TARGET_REDUCTION)
        shortfall = max(0.0, expected - reduction)
        aggression = min(1.0, shortfall / max(ADAPT_TARGET_REDUCTION, 1e-6))
        headroom = max(0.0, self.current_score - self.tau_floor_absolute)
        span = max(self.baseline_score - self.tau_floor_absolute, 1e-6)
        headroom_ratio = min(1.0, headroom / span)
        headroom_drive = headroom_ratio * progress * ADAPT_HEADROOM_WEIGHT
        aggression = max(aggression, headroom_drive)
        if aggression <= 0:
            return

        floor_rel_target = TAU_FLOOR_REL - ADAPT_FLOOR_PUSH * aggression
        floor_rel_target = max(ADAPT_FLOOR_MIN_REL, floor_rel_target)
        floor_abs_target = self.baseline_score * floor_rel_target
        tau_target = self.baseline_score * (self.base_tau - ADAPT_TAU_PUSH * aggression)
        tau_target = max(floor_abs_target, tau_target)
        epsilon_target = self.base_epsilon * (1.0 + ADAPT_EPS_SCALE * aggression)
        pool_target = int(round(self.base_candidate_pool * (1.0 + ADAPT_POOL_SCALE * aggression)))
        trials_target = int(round(self.base_max_stage_trials * (1.0 - ADAPT_STAGE_TRIALS_SCALE * aggression)))

        changed = False
        if floor_abs_target < self.tau_floor_absolute:
            self.tau_floor_absolute = floor_abs_target
            changed = True
        if tau_target < self.tau_absolute:
            self.tau_absolute = max(self.tau_floor_absolute, tau_target)
            changed = True
        if epsilon_target > self.config.epsilon:
            self.config.epsilon = min(ADAPT_EPS_MAX, epsilon_target)
            changed = True
        if pool_target > self.config.candidate_pool:
            self.config.candidate_pool = min(ADAPT_POOL_MAX, max(2, pool_target))
            changed = True
        if trials_target < self.config.max_stage_trials:
            self.config.max_stage_trials = max(ADAPT_MIN_STAGE_TRIALS, trials_target)
            changed = True

        if changed:
            log_event(
                {
                    "event": "adaptive_update",
                    "iter": t,
                    "progress": progress,
                    "reduction": reduction,
                    "expected_reduction": expected,
                    "aggression": aggression,
                    "tau_absolute": self.tau_absolute,
                    "tau_floor_absolute": self.tau_floor_absolute,
                    "epsilon": self.config.epsilon,
                    "candidate_pool": self.config.candidate_pool,
                    "max_stage_trials": self.config.max_stage_trials,
                }
            )

    def run(self) -> None:
        initial_params = count_parameters(self.generator.model)
        num_layers = len(self.state.layer_orig_indices)
        print("PRUNING ENGINE INITIALIZED")
        print(f"  Initial parameters: {initial_params:,}")
        print(f"  Number of layers: {num_layers}")
        print(f"  Calibration prompts: {len(self.prompts)} (train: {len(self.train_indices)}, holdout: {len(self.holdout_indices)})")
        print(f"  Stage plan: {' -> '.join(self.config.stage_plan)}")
        print(f"  Similarity threshold (tau): {self.config.tau}")
        print(f"  Max iterations: {self.config.max_iters}")

        print("Computing baseline score...")
        baseline_indices = self.train_indices or list(range(len(self.prompts)))
        self.current_score = self.score_subset(baseline_indices, desc="Baseline")
        self.baseline_score = self.current_score
        if self.config.tau <= 1.0:
            self.tau_absolute = self.baseline_score * self.config.tau
        else:
            self.tau_absolute = self.config.tau
        self.tau_floor_absolute = self.baseline_score * TAU_FLOOR_REL
        print(f"Baseline score: {self.current_score:.4f}")
        print(
            f"Effective tau: {self.tau_absolute:.4f} (relative={self.config.tau:.3f}, floor={self.tau_floor_absolute:.4f})\n"
        )
        log_event(
            {
                "event": "baseline",
                "baseline_score": self.baseline_score,
                "tau_config": self.config.tau,
                "tau_absolute": self.tau_absolute,
                "tau_floor_absolute": self.tau_floor_absolute,
                "num_prompts": len(self.prompts),
                "num_layers": len(self.state.layer_orig_indices),
            }
        )

        pbar = tqdm(range(self.config.max_iters), desc="Pruning iterations", unit="iter")
        for t in pbar:
            self.update_adaptive_controls(t, initial_params)
            stage = self.config.stage_plan[self.stage_idx]
            pbar.set_postfix({
                "stage": stage,
                "score": f"{self.current_score:.3f}",
                "accepted": len(self.accepted_ops),
                "params": f"{count_parameters(self.generator.model)/1e9:.2f}B",
                "tau": f"{self.tau_absolute:.3f}",
            })
            self.stage_iters += 1
            if self.stage_iters >= STAGE_MAX_ITERS and len(self.config.stage_plan) > 1:
                if self.stage_idx + 1 < len(self.config.stage_plan):
                    tqdm.write(
                        f"\n[iter {t}] Stage '{stage}' reached {STAGE_MAX_ITERS} iterations, advancing."
                    )
                    log_event(
                        {
                            "event": "stage_advance",
                            "reason": "stage_max_iters",
                            "iter": t,
                            "from_stage": stage,
                            "to_stage": self.config.stage_plan[self.stage_idx + 1],
                        }
                    )
                    self.stage_idx += 1
                    self.stage_failures = 0
                    self.stage_iters = 0
                    self.refresh_importances()
                    continue

            candidates = self.build_candidates(stage)
            if not candidates:
                if self.stage_idx + 1 < len(self.config.stage_plan):
                    tqdm.write(f"\n[iter {t}] No candidates in stage '{stage}', advancing to next stage.")
                    log_event(
                        {
                            "event": "stage_advance",
                            "reason": "no_candidates",
                            "iter": t,
                            "from_stage": stage,
                            "to_stage": self.config.stage_plan[self.stage_idx + 1],
                        }
                    )
                    self.stage_idx += 1
                    self.stage_failures = 0
                    self.stage_iters = 0
                    self.refresh_importances()
                    continue
                tqdm.write(f"\n[iter {t}] No candidates left. Stopping.")
                break

            sampled = sample_candidates(
                candidates,
                self.config.candidate_pool,
                self.config.temperature,
                self.mu,
                self.config.sample_bandit_weight,
            )
            op = self.pick_candidate(sampled, t)
            accepted, s_small, s_full = self.evaluate_operation(op, t)

            if accepted:
                self.current_score = s_full
                self.accepted_ops.append(op)
                self.stage_failures = 0
                self.accepted_since_refresh += 1
                params_now = count_parameters(self.generator.model)
                holdout_msg = ""
                if self.holdout_indices:
                    holdout_score = self.score_subset(self.holdout_indices, desc="Holdout")
                    holdout_msg = f" | holdout={holdout_score:.4f}"
                tqdm.write(
                    f"  [iter {t}] ACCEPT {op.op_id()} | small={s_small:.4f} | full={s_full:.4f}{holdout_msg} | params={params_now:,}"
                )
                log_event(
                    {
                        "event": "accept",
                        "iter": t,
                        "stage": stage,
                        "op": op.op_id(),
                        "op_type": op.op_type,
                        "layer": op.layer_orig_idx,
                        "target_size": op.target_size,
                        "remove_group": op.remove_group_orig_idx,
                        "remove_score": op.remove_score,
                        "score_small": s_small,
                        "score_full": s_full,
                        "score_current": self.current_score,
                        "tau_absolute": self.tau_absolute,
                        "params": params_now,
                        "accepted": len(self.accepted_ops),
                    }
                )
                if self.config.refresh_every > 0 and (
                    self.accepted_since_refresh >= self.config.refresh_every
                ):
                    self.refresh_importances()
            else:
                self.stage_failures += 1
                tqdm.write(
                    f"  [iter {t}] REJECT {op.op_id()} | small={s_small:.4f} | full={s_full:.4f} | failures={self.stage_failures}/{self.config.max_stage_trials}"
                )
                log_event(
                    {
                        "event": "reject",
                        "iter": t,
                        "stage": stage,
                        "op": op.op_id(),
                        "op_type": op.op_type,
                        "layer": op.layer_orig_idx,
                        "target_size": op.target_size,
                        "remove_group": op.remove_group_orig_idx,
                        "remove_score": op.remove_score,
                        "score_small": s_small,
                        "score_full": s_full,
                        "score_current": self.current_score,
                        "tau_absolute": self.tau_absolute,
                        "params": count_parameters(self.generator.model),
                        "accepted": len(self.accepted_ops),
                    }
                )
                if self.stage_failures >= self.config.max_stage_trials:
                    if self.tau_absolute > self.tau_floor_absolute:
                        new_tau = max(self.tau_floor_absolute, self.tau_absolute * TAU_DECAY)
                        if new_tau < self.tau_absolute:
                            old_tau = self.tau_absolute
                            self.tau_absolute = new_tau
                            self.stage_failures = 0
                            tqdm.write(
                                f"\n[iter {t}] Stage '{stage}' stalled. Lowering tau {old_tau:.4f} -> {self.tau_absolute:.4f} and retrying."
                            )
                            log_event(
                                {
                                    "event": "tau_decay",
                                    "iter": t,
                                    "stage": stage,
                                    "tau_before": old_tau,
                                    "tau_after": self.tau_absolute,
                                    "tau_floor": self.tau_floor_absolute,
                                }
                            )
                            continue
                    if self.stage_idx + 1 < len(self.config.stage_plan):
                        tqdm.write(
                            f"\n[iter {t}] Stage '{stage}' stalled after {self.stage_failures} trials, advancing to next stage."
                        )
                        log_event(
                            {
                                "event": "stage_advance",
                                "reason": "stage_stalled",
                                "iter": t,
                                "from_stage": stage,
                                "to_stage": self.config.stage_plan[self.stage_idx + 1],
                            }
                        )
                        self.stage_idx += 1
                        self.stage_failures = 0
                        self.stage_iters = 0
                        self.refresh_importances()
                    else:
                        tqdm.write(f"\n[iter {t}] All stages exhausted. Stopping.")
                        break

        pbar.close()
        final_params = count_parameters(self.generator.model)
        compression = 1.0 - (final_params / initial_params)
        print("PRUNING COMPLETE")
        print(f"  Final parameters: {final_params:,} ({compression*100:.1f}% reduction)")
        print(f"  Final score: {self.current_score:.4f}")
        print(f"  Accepted operations: {len(self.accepted_ops)}")
        print(f"  Remaining layers: {len(self.state.layer_orig_indices)}")
        log_event(
            {
                "event": "final",
                "final_score": self.current_score,
                "baseline_score": self.baseline_score,
                "tau_absolute": self.tau_absolute,
                "final_params": final_params,
                "compression": compression,
                "accepted_ops": len(self.accepted_ops),
                "remaining_layers": len(self.state.layer_orig_indices),
            }
        )

    def build_candidates(self, stage: str) -> List[Operation]:
        candidates: List[Operation] = []
        if stage.startswith("mlp"):
            candidates.extend(self._build_mlp_candidates())
        if stage == "head":
            candidates.extend(self._build_head_candidates())
        if stage == "layer":
            candidates.extend(self._build_layer_candidates())
        return candidates

    def _build_mlp_candidates(self) -> List[Operation]:
        candidates: List[Operation] = []
        for layer_orig_idx in self.state.layer_orig_indices:
            if layer_orig_idx not in self.mlp_importance:
                continue
            current_orig = self.state.mlp_orig_indices_by_layer[layer_orig_idx]
            current_size = len(current_orig)
            importance = self.mlp_importance[layer_orig_idx]
            for divisor in MLP_DIVISORS:
                target_size = current_size // divisor
                if target_size < self.config.min_mlp_size:
                    continue
                if target_size >= current_size:
                    continue
                keep_positions, keep_orig, remove_score = select_mlp_keep(
                    current_orig,
                    importance,
                    target_size,
                    self.rng,
                    self.config.mlp_top_keep,
                    self.config.mlp_sample_band,
                )
                candidates.append(
                    Operation(
                        op_type="mlp",
                        layer_orig_idx=layer_orig_idx,
                        remove_score=remove_score,
                        target_size=target_size,
                        keep_positions=keep_positions,
                        keep_orig_indices=keep_orig,
                    )
                )
        return candidates

    def _build_head_candidates(self) -> List[Operation]:
        candidates: List[Operation] = []
        for layer_orig_idx in self.state.layer_orig_indices:
            if layer_orig_idx not in self.head_importance:
                continue
            head_map = self.state.head_map_by_layer[layer_orig_idx]
            if len(head_map.kv_head_orig_indices) <= self.config.min_kv_heads:
                continue
            group_importance = compute_group_importance(
                self.head_importance[layer_orig_idx],
                head_map.group_size,
            )
            for group_pos, kv_orig in enumerate(head_map.kv_head_orig_indices):
                if group_pos >= len(group_importance):
                    continue
                candidates.append(
                    Operation(
                        op_type="head",
                        layer_orig_idx=layer_orig_idx,
                        remove_score=float(group_importance[group_pos]),
                        remove_group_orig_idx=kv_orig,
                    )
                )
        return candidates

    def _build_layer_candidates(self) -> List[Operation]:
        candidates: List[Operation] = []
        if len(self.state.layer_orig_indices) <= self.config.min_layers:
            return candidates
        for layer_orig_idx in self.state.layer_orig_indices:
            score = self.layer_importance.get(layer_orig_idx, 0.0)
            candidates.append(
                Operation(
                    op_type="layer",
                    layer_orig_idx=layer_orig_idx,
                    remove_score=float(score),
                )
            )
        return candidates

    def pick_candidate(self, candidates: List[Operation], t: int) -> Operation:
        best_op = candidates[0]
        best_score = float("inf")
        log_term = math.log(1 + t)
        for op in candidates:
            op_id = op.op_id()
            mu = self.mu.get(op_id, 0.0)
            n = self.counts.get(op_id, 0)
            score = mu - self.config.bandit_lambda * math.sqrt(
                log_term / (1 + n)
            )
            if score < best_score:
                best_score = score
                best_op = op
        return best_op

    def evaluate_operation(self, op: Operation, t: int) -> Tuple[bool, float, float]:
        if op.op_type == "mlp":
            undo = apply_mlp_prune(
                self.generator.model,
                self.state,
                op.layer_orig_idx,
                op.keep_positions or [],
                op.keep_orig_indices or [],
            )
        elif op.op_type == "head":
            undo = apply_head_prune(
                self.generator.model,
                self.state,
                op.layer_orig_idx,
                op.remove_group_orig_idx or 0,
            )
        else:
            undo = apply_layer_drop(
                self.generator.model, self.state, op.layer_orig_idx
            )

        indices_small = self.sample_small_indices()
        s_small = self.score_subset(indices_small, desc="Small eval")

        if s_small < self.tau_absolute - self.config.small_slack:
            delta = self.current_score - s_small
            self.update_memory(op, delta)
            undo.revert()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False, s_small, s_small

        full_indices = self.train_indices or list(range(len(self.prompts)))
        s_full = self.score_subset(full_indices, desc="Full eval")
        delta = self.current_score - s_full
        self.update_memory(op, delta)

        accepted = s_full >= self.tau_absolute
        if self.config.epsilon > 0:
            accepted = accepted and (self.current_score - s_full <= self.config.epsilon)

        if not accepted:
            undo.revert()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return accepted, s_small, s_full

    def score_subset(self, indices: List[int], desc: str) -> float:
        prompts = [self.prompts[i] for i in indices]
        outputs = self.generator.generate(prompts, desc=f"{desc} generation")
        outputs = [normalize_text(text) for text in outputs]
        cand_emb = self.embedder.encode(outputs, as_query=False, desc=f"{desc} encoding")
        teacher_emb = self.teacher_embeddings[indices]
        scores = (cand_emb * teacher_emb).sum(dim=1)
        return scores.mean().item()

    def update_memory(self, op: Operation, delta: float) -> None:
        op_id = op.op_id()
        n = self.counts.get(op_id, 0)
        mu = self.mu.get(op_id, 0.0)
        new_mu = (mu * n + delta) / (n + 1)
        self.mu[op_id] = new_mu
        self.counts[op_id] = n + 1

    def sample_small_indices(self) -> List[int]:
        pool = self.train_indices or list(range(len(self.prompts)))
        k = min(self.config.small_k, len(pool))
        if k <= 0:
            return []
        return list(self.rng.sample(pool, k))

    def refresh_importances(self) -> None:
        tqdm.write("\n  >> Refreshing importance statistics on pruned model...")
        mlp_imp, head_imp, layer_imp = collect_importances(
            self.generator, self.teacher_forcing_texts, desc="Refreshing stats"
        )
        self.mlp_importance, self.head_importance, self.layer_importance = (
            remap_importances_to_orig(
                mlp_imp, head_imp, layer_imp, self.state.layer_orig_indices
            )
        )
        self.accepted_since_refresh = 0
        tqdm.write("  >> Importances refreshed.\n")
        log_event(
            {
                "event": "refresh_importances",
                "layers": len(self.state.layer_orig_indices),
            }
        )


def compute_group_importance(head_importance: Tensor, group_size: int) -> Tensor:
    num_heads = head_importance.shape[0]
    num_groups = num_heads // max(group_size, 1)
    groups = []
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        groups.append(head_importance[start:end].mean())
    return torch.stack(groups, dim=0)


def select_mlp_keep(
    current_orig: List[int],
    importance: Tensor,
    target_size: int,
    rng: random.Random,
    top_keep_frac: float,
    sample_band_frac: float,
) -> Tuple[List[int], List[int], float]:
    num_positions = len(current_orig)
    positions = list(range(num_positions))
    ranked_positions = sorted(positions, key=lambda pos: float(importance[pos]), reverse=True)
    fixed = max(0, min(target_size, int(target_size * top_keep_frac)))
    remaining = max(0, target_size - fixed)
    fixed_keep_positions = ranked_positions[:fixed]

    band_size = max(remaining, int(target_size * sample_band_frac))
    band_end = min(len(ranked_positions), fixed + band_size)
    band_candidates = ranked_positions[fixed:band_end]
    if remaining > 0:
        if len(band_candidates) >= remaining:
            sampled_positions = rng.sample(band_candidates, remaining)
        else:
            sampled_positions = list(band_candidates)
            extra_needed = remaining - len(sampled_positions)
            tail_candidates = ranked_positions[band_end:]
            if extra_needed > 0 and tail_candidates:
                sampled_positions.extend(rng.sample(tail_candidates, min(extra_needed, len(tail_candidates))))
        keep_positions_set = set(fixed_keep_positions + sampled_positions)
    else:
        keep_positions_set = set(fixed_keep_positions)

    keep_positions = sorted(keep_positions_set)
    keep_orig = [current_orig[pos] for pos in keep_positions]
    removed_positions = [pos for pos in positions if pos not in keep_positions_set]
    if removed_positions:
        remove_score = float(importance[removed_positions].mean().item())
    else:
        remove_score = 0.0
    return keep_positions, keep_orig, remove_score


def sample_candidates(
    candidates: List[Operation],
    k: int,
    temperature: float,
    mu: Dict[str, float],
    bandit_weight: float,
) -> List[Operation]:
    if len(candidates) <= k:
        return candidates
    scores_list = []
    for op in candidates:
        score = op.remove_score + bandit_weight * mu.get(op.op_id(), 0.0)
        scores_list.append(score)
    scores = torch.tensor(scores_list, dtype=torch.float32)
    scores = scores - scores.min()
    weights = torch.exp(-scores / max(temperature, 1e-6))
    weights = weights / weights.sum()
    idx = torch.multinomial(weights, k, replacement=False).tolist()
    return [candidates[i] for i in idx]


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_pruned_layer_info(state: ModelState) -> List[Dict[str, object]]:
    layer_info: List[Dict[str, object]] = []
    for layer_orig_idx in state.layer_orig_indices:
        head_map = state.head_map_by_layer[layer_orig_idx]
        layer_info.append(
            {
                "layer_orig_idx": layer_orig_idx,
                "mlp_size": len(state.mlp_orig_indices_by_layer[layer_orig_idx]),
                "num_heads": len(head_map.query_head_orig_indices),
                "num_kv_heads": len(head_map.kv_head_orig_indices),
                "head_dim": head_map.head_dim,
                "kv_group_size": head_map.group_size,
            }
        )
    return layer_info


def save_pruning_recipe(
    recipe_path: str,
    base_model_name: str,
    accepted_ops: List[Operation],
    state: ModelState,
    final_score: float,
) -> None:
    recipe = {
        "base_model": base_model_name,
        "final_score": final_score,
        "accepted_ops": [op.op_id() for op in accepted_ops],
        "keep_layer_indices": state.layer_orig_indices,
        "keep_mlp_indices_by_layer": {
            str(k): v for k, v in state.mlp_orig_indices_by_layer.items()
        },
        "keep_heads_by_layer": {
            str(k): {
                "query_head_indices": v.query_head_orig_indices,
                "kv_head_indices": v.kv_head_orig_indices,
                "group_size": v.group_size,
                "head_dim": v.head_dim,
            }
            for k, v in state.head_map_by_layer.items()
        },
        "summary": build_pruned_layer_info(state),
    }
    with open(recipe_path, "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2)


def load_and_prune_model(
    recipe_path: str, device: str, dtype: torch.dtype
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    with open(recipe_path, "r", encoding="utf-8") as f:
        recipe = json.load(f)

    base_model_name = recipe["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, use_fast=True, padding_side="left"
    )
    ensure_pad_token(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False

    apply_pruning_recipe(model, recipe)
    return model, tokenizer


def apply_pruning_recipe(model: torch.nn.Module, recipe: Dict[str, object]) -> None:
    state = ModelState.from_model(model)
    keep_layer_indices = [int(x) for x in recipe["keep_layer_indices"]]
    mlp_by_layer = {int(k): v for k, v in recipe["keep_mlp_indices_by_layer"].items()}
    head_by_layer = {int(k): v for k, v in recipe["keep_heads_by_layer"].items()}

    for layer_idx in keep_layer_indices:
        keep_mlp = mlp_by_layer.get(layer_idx)
        if keep_mlp is not None:
            current_orig = state.mlp_orig_indices_by_layer[layer_idx]
            if len(keep_mlp) < len(current_orig):
                keep_set = set(keep_mlp)
                keep_positions = [
                    i for i, idx in enumerate(current_orig) if idx in keep_set
                ]
                keep_orig = [current_orig[i] for i in keep_positions]
                apply_mlp_prune(model, state, layer_idx, keep_positions, keep_orig)

        head_meta = head_by_layer.get(layer_idx)
        if head_meta is not None:
            keep_kv = set(head_meta["kv_head_indices"])
            current_kv = list(state.head_map_by_layer[layer_idx].kv_head_orig_indices)
            for kv_idx in current_kv:
                if kv_idx not in keep_kv:
                    apply_head_prune(model, state, layer_idx, kv_idx)

    layers = get_decoder_layers(model)
    new_layers = torch.nn.ModuleList([layers[i] for i in keep_layer_indices])
    if hasattr(model, "model"):
        model.model.layers = new_layers
    else:
        model.layers = new_layers
    model.config.num_hidden_layers = len(new_layers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structured pruning with embedding similarity.")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Llama model name (<10B).",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="Qwen/Qwen3-Embedding-4B",
        help="Embedding model name.",
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        default="questions_deep_learning.txt",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=str,
        default="system_prompt_deep_learning.txt",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embed-device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--embed-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--candidate-pool", type=int, default=12)
    parser.add_argument("--max-iters", type=int, default=150)
    parser.add_argument("--small-k", type=int, default=8)
    parser.add_argument("--holdout-k", type=int, default=8)
    parser.add_argument("--small-slack", type=float, default=0.02)
    parser.add_argument("--bandit-lambda", type=float, default=0.5)
    parser.add_argument("--sample-bandit-weight", type=float, default=1.0)
    parser.add_argument("--refresh-every", type=int, default=5)
    parser.add_argument("--mlp-top-keep", type=float, default=0.6)
    parser.add_argument("--mlp-sample-band", type=float, default=0.4)
    parser.add_argument("--min-mlp-size", type=int, default=256)
    parser.add_argument("--min-kv-heads", type=int, default=1)
    parser.add_argument("--min-layers", type=int, default=1)
    parser.add_argument("--max-stage-trials", type=int, default=12)
    parser.add_argument(
        "--stage-plan",
        type=str,
        default="mlp,head,mlp,layer",
        help="Comma-separated stages.",
    )
    parser.add_argument("--save-recipe", type=str, default="", help="Path to save pruning recipe JSON.")
    parser.add_argument("--load-recipe", type=str, default="", help="Path to load pruning recipe JSON and apply it.")
    parser.add_argument(
        "--embedding-task",
        type=str,
        default="Given a candidate answer, retrieve the reference answer that is semantically equivalent",
        help="Task instruction for query embeddings (candidate outputs).",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    name = name.lower().strip()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if SEQ_LEN != 256:
        raise ValueError("SEQ_LEN must be 256.")

    if args.max_new_tokens >= SEQ_LEN:
        raise ValueError("max_new_tokens must be less than 256.")

    dtype = resolve_dtype(args.dtype)
    if args.load_recipe:
        model, tokenizer = load_and_prune_model(args.load_recipe, args.device, dtype)
        print(
            f"Loaded and pruned model from recipe {args.load_recipe} with {count_parameters(model):,} parameters."
        )
        return

    args.batch_size = max(1, args.batch_size * GEN_BATCH_MULTIPLIER)
    args.embed_batch_size = max(1, args.embed_batch_size * EMBED_BATCH_MULTIPLIER)

    questions = read_lines(args.questions_file)
    system_prompt = read_text(args.system_prompt_file)

    print("SETUP PHASE")
    print(f"  Questions file: {args.questions_file} ({len(questions)} questions)")
    print(f"  System prompt: {args.system_prompt_file}")
    print(f"  Teacher model: {args.model}")
    print(f"  Embedding model: {args.embed_model}")
    print(f"  Device: {args.device} | Embed device: {args.embed_device}")
    print(f"  Batch size: {args.batch_size} | Embed batch size: {args.embed_batch_size}")
    print(f"  Debug log: {DEBUG_LOG_PATH} (enabled={DEBUG_LOG_ENABLED})")
    print(f"  Tau decay: {TAU_DECAY} | Tau floor (rel): {TAU_FLOOR_REL}")
    print(f"  Stage max iters: {STAGE_MAX_ITERS} | MLP divisors: {MLP_DIVISORS}")

    print("[1/4] Loading teacher model...")
    generator = LlamaGenerator(
        model_name=args.model,
        system_prompt=system_prompt,
        device=args.device,
        dtype=dtype,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    prompts = generator.build_prompts(questions)
    print(f"      Model loaded. Parameters: {count_parameters(generator.model):,}\n")

    print("[2/4] Generating teacher outputs...")
    teacher_outputs = generator.generate(prompts, desc="Teacher generation")
    teacher_outputs = [normalize_text(text) for text in teacher_outputs]
    teacher_forcing_texts = []
    for prompt, output in zip(prompts, teacher_outputs):
        sep = "" if prompt.endswith(("\n", " ")) else " "
        teacher_forcing_texts.append(prompt + sep + output)
    print(f"      Generated {len(teacher_outputs)} outputs.\n")

    print("[3/4] Loading embedding model and encoding teacher outputs...")
    embed_task = args.embedding_task.strip() or "Given a candidate answer, retrieve the reference answer that is semantically equivalent"
    embedder = EmbeddingScorer(
        model_name=args.embed_model,
        device=args.embed_device,
        batch_size=args.embed_batch_size,
        task_description=embed_task,
    )
    teacher_embeddings = embedder.encode(teacher_outputs, as_query=False, desc="Encoding teacher")
    print(f"      Encoded {len(teacher_embeddings)} embeddings.\n")

    state = ModelState.from_model(generator.model)

    print("[4/4] Collecting importance statistics...")
    mlp_imp, head_imp, layer_imp = collect_importances(
        generator, teacher_forcing_texts, desc="Initial importance stats"
    )
    importances = remap_importances_to_orig(
        mlp_imp, head_imp, layer_imp, state.layer_orig_indices
    )
    print(f"      Collected stats for {len(state.layer_orig_indices)} layers.\n")
    config = PruneConfig(
        tau=args.tau,
        epsilon=args.epsilon,
        temperature=args.temperature,
        candidate_pool=args.candidate_pool,
        max_iters=args.max_iters,
        small_k=args.small_k,
        holdout_k=args.holdout_k,
        small_slack=args.small_slack,
        bandit_lambda=args.bandit_lambda,
        sample_bandit_weight=args.sample_bandit_weight,
        refresh_every=args.refresh_every,
        mlp_top_keep=args.mlp_top_keep,
        mlp_sample_band=args.mlp_sample_band,
        min_mlp_size=args.min_mlp_size,
        min_kv_heads=args.min_kv_heads,
        min_layers=args.min_layers,
        max_stage_trials=args.max_stage_trials,
        stage_plan=tuple(s.strip() for s in args.stage_plan.split(",") if s.strip()),
        seed=args.seed,
    )
    log_event(
        {
            "event": "config",
            "model": args.model,
            "embed_model": args.embed_model,
            "questions": len(questions),
            "batch_size": args.batch_size,
            "embed_batch_size": args.embed_batch_size,
            "tau": args.tau,
            "tau_decay": TAU_DECAY,
            "tau_floor_rel": TAU_FLOOR_REL,
            "stage_max_iters": STAGE_MAX_ITERS,
            "mlp_divisors": list(MLP_DIVISORS),
            "epsilon": args.epsilon,
            "candidate_pool": args.candidate_pool,
            "max_iters": args.max_iters,
            "small_k": args.small_k,
            "holdout_k": args.holdout_k,
            "small_slack": args.small_slack,
            "bandit_lambda": args.bandit_lambda,
            "sample_bandit_weight": args.sample_bandit_weight,
            "refresh_every": args.refresh_every,
            "min_mlp_size": args.min_mlp_size,
            "min_kv_heads": args.min_kv_heads,
            "min_layers": args.min_layers,
            "max_stage_trials": args.max_stage_trials,
            "stage_plan": list(config.stage_plan),
        }
    )

    pruner = PruningEngine(
        generator=generator,
        embedder=embedder,
        teacher_embeddings=teacher_embeddings,
        prompts=prompts,
        teacher_forcing_texts=teacher_forcing_texts,
        importances=importances,
        state=state,
        config=config,
    )
    pruner.run()

    save_path = args.save_recipe.strip()
    if not save_path and AUTO_SAVE_RECIPE:
        save_path = DEFAULT_RECIPE_PATH
    if save_path:
        save_pruning_recipe(
            save_path,
            args.model,
            pruner.accepted_ops,
            pruner.state,
            pruner.current_score,
        )
        print(f"Saved pruning recipe to {save_path}")


if __name__ == "__main__":
    main()
