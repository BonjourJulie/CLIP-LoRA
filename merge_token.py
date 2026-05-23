"""
merge.py
========
两种 token 聚合策略 + 逐层 Hook 提取图文向量工具。

策略
----
1. cls_eos  : 图像侧取 [CLS] token（位置 0），文本侧取 [EOS] token（argmax 位置）
2. mean_pool: 图像侧对全部 patch token（含 CLS）取平均，文本侧对全部 token 取平均

公开接口
--------
- reduce_image_tokens(layer_out, strategy) -> [B, D]
- reduce_text_tokens(layer_out, text_tokens, strategy) -> [B, D]
- LayerwiseHookManager  : 注册所有 Transformer block 的 forward hook
- extract_layerwise_embeddings_classification(...)
- extract_layerwise_embeddings_flickr(...)
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Literal, Tuple

import clip
import torch
import torch.nn as nn

Strategy = Literal["cls_eos", "mean_pool"]

######### change
def reduce_image_tokens(
    layer_out: torch.Tensor,
    strategy: Strategy = "cls_eos",
) -> torch.Tensor:
    if layer_out.dim() == 2:
        return layer_out.float()

    # --- 关键修改：将 [L, B, D] 转换为 [B, L, D] ---
    if layer_out.shape[0] != 0 and layer_out.dim() == 3:
        # 如果第一维不是 batch_size (通常序列长度是固定的 197 或 77)
        # 稳妥起见，CLIP Transformer Block 输出需要转置
        layer_out = layer_out.permute(1, 0, 2)

    if strategy == "cls_eos":
        # 此时 layer_out 形状为 [B, 197, D]，索引 0 是 CLS
        return layer_out[:, 0, :].float() 
    elif strategy == "mean_pool":
        # 此时在 L 维度 (dim=1) 求平均，保留 B 维度
        return layer_out.mean(dim=1).float()
    else:
        raise ValueError(f"未知策略: {strategy}")

def reduce_text_tokens(
    layer_out: torch.Tensor,
    text_tokens: torch.Tensor,
    strategy: Strategy = "cls_eos",
) -> torch.Tensor:
    if layer_out.dim() == 2:
        return layer_out.float()

    # --- 关键修改：将 [L, B, D] 转换为 [B, L, D] ---
    if layer_out.dim() == 3:
        layer_out = layer_out.permute(1, 0, 2)

    if strategy == "cls_eos":
        eot_idx = text_tokens.argmax(dim=-1)
        # 此时 layer_out 形状为 [B, 77, D]
        return layer_out[torch.arange(layer_out.shape[0]), eot_idx].float()
    elif strategy == "mean_pool":
        return layer_out.mean(dim=1).float()
    else:
        raise ValueError(f"未知策略: {strategy}")

'''
# ---------------------------------------------------------------------------
# Token 聚合函数
# ---------------------------------------------------------------------------

def reduce_image_tokens(
    layer_out: torch.Tensor,
    strategy: Strategy = "cls_eos",
) -> torch.Tensor:
    """
    将图像侧单层输出 [B, N_patch, D] 压缩为 [B, D]。

    ViT 的典型序列布局: [CLS, patch_1, ..., patch_196]，共 197 个位置。

    策略
    ----
    cls_eos  : 取位置 0（CLS token）
    mean_pool: 对全部 197 个 token 取平均
    """
    if layer_out.dim() == 2:
        # 某些 hook 位置已经是 [B, D]（如 ln_post 之后），直接返回
        return layer_out.float()

    if strategy == "cls_eos":
        return layer_out[:, 0, :].float()          # [B, D]
    elif strategy == "mean_pool":
        return layer_out.mean(dim=1).float()        # [B, D]
    else:
        raise ValueError(f"未知策略: {strategy}，可选 'cls_eos' 或 'mean_pool'")


def reduce_text_tokens(
    layer_out: torch.Tensor,
    text_tokens: torch.Tensor,
    strategy: Strategy = "cls_eos",
) -> torch.Tensor:
    """
    将文本侧单层输出 [B, L, D] 压缩为 [B, D]。

    text_tokens: clip.tokenize 的输出 [B, 77]，用于定位 EOS 位置（argmax）。

    策略
    ----
    cls_eos  : 取 EOS token（argmax 位置，与 CLIP 官方做法一致）
    mean_pool: 对全部 77 个 token 取平均
    """
    if layer_out.dim() == 2:
        return layer_out.float()

    if strategy == "cls_eos":
        eot_idx = text_tokens.argmax(dim=-1)        # [B]
        return layer_out[torch.arange(len(text_tokens)), eot_idx].float()   # [B, D]
    elif strategy == "mean_pool":
        return layer_out.mean(dim=1).float()        # [B, D]
    else:
        raise ValueError(f"未知策略: {strategy}，可选 'cls_eos' 或 'mean_pool'")
'''

# ---------------------------------------------------------------------------
# Hook 管理器
# ---------------------------------------------------------------------------

class LayerwiseHookManager:
    """
    一次性注册 CLIP 视觉编码器与文本编码器每一层 Transformer block 的
    forward hook，在 forward 后可从 storage 中读取各层输出。

    存储键格式
    ----------
    "img_layer_{i}" : 视觉编码器第 i 层（0-based）输出，[B, N_patch, D]
    "txt_layer_{i}" : 文本编码器第 i 层（0-based）输出，[B, L, D]
    """

    def __init__(self, clip_model: nn.Module):
        self.storage: Dict[str, torch.Tensor] = {}
        self._handles: List = []
        self._register(clip_model)

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------
    def _register(self, clip_model: nn.Module) -> None:
        # ---- 视觉编码器 ----
        # OpenAI CLIP ViT: clip_model.visual.transformer.resblocks
        img_blocks = clip_model.visual.transformer.resblocks
        for i, block in enumerate(img_blocks):
            key = f"img_layer_{i}"
            handle = block.register_forward_hook(self._make_hook(key))
            self._handles.append(handle)

        # ---- 文本编码器 ----
        # OpenAI CLIP Transformer: clip_model.transformer.resblocks
        txt_blocks = clip_model.transformer.resblocks
        for i, block in enumerate(txt_blocks):
            key = f"txt_layer_{i}"
            handle = block.register_forward_hook(self._make_hook(key))
            self._handles.append(handle)

        self.n_img_layers = len(img_blocks)
        self.n_txt_layers = len(txt_blocks)

    def _make_hook(self, key: str):
        def hook(module, input, output):
            # ResidualAttentionBlock 的输出是 Tensor 或 (Tensor, ...)
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            self.storage[key] = out.detach()
        return hook

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------
    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.storage.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()


# ---------------------------------------------------------------------------
# 逐层嵌入提取：分类数据集
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_layerwise_embeddings_classification(
    clip_model: nn.Module,
    loader,
    num_pairs: int,
    strategy: Strategy = "cls_eos",
    device: str = "cuda",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
    """
    分类数据集逐层提取。

    Returns
    -------
    img_by_layer : {"img_layer_0": [N, D], ..., "img_layer_K": [N, D]}
    txt_by_layer : {"txt_layer_0": [N, D], ..., "txt_layer_K": [N, D]}
    labels       : [N]
    """
    clip_model.eval()
    hook_mgr = LayerwiseHookManager(clip_model)

    n_img = hook_mgr.n_img_layers
    n_txt = hook_mgr.n_txt_layers

    img_buf: Dict[str, List[torch.Tensor]] = {f"img_layer_{i}": [] for i in range(n_img)}
    txt_buf: Dict[str, List[torch.Tensor]] = {f"txt_layer_{i}": [] for i in range(n_txt)}
    lbl_buf: List[torch.Tensor] = []

    collected = 0

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        for images, targets in loader:
            if collected >= num_pairs:
                break

            images  = images.to(device)
            targets = targets.to(device)

            # --- forward（触发所有 hook）---
            _ = clip_model.encode_image(images)

            remaining = num_pairs - collected
            cut = min(images.shape[0], remaining)

            # --- 图像侧各层 ---
            for i in range(n_img):
                key = f"img_layer_{i}"
                raw = hook_mgr.storage[key]             # [B, N_patch, D] 或 [B, D]
                feat = reduce_image_tokens(raw, strategy)[:cut]
                feat = feat / feat.norm(dim=-1, keepdim=True)
                img_buf[key].append(feat.cpu().float())

            # --- 文本侧各层：用目标类别标签的 token 做一次 forward ---
            # 注意：分类任务中文本是类别名称，这里对每张图对应的类别单独编码
            # 先收集本 batch 的类别名称
            # （dataset.classnames 需调用方传入；此处接收 targets 并在外部封装）
            # 实际做法：在函数外部已调用，此处只读 hook storage；
            # 因此需要在同一 forward 中顺带触发文本 encoder
            # -> 见 extract_layerwise_cls_with_text 版本

            lbl_buf.append(targets[:cut].cpu())
            collected += cut

    hook_mgr.remove()

    img_by_layer = {k: torch.cat(v, dim=0) for k, v in img_buf.items()}
    labels       = torch.cat(lbl_buf, dim=0)

    # txt_by_layer 在纯分类模式下不含逐样本文本，返回空字典
    # 推荐使用下方 extract_layerwise_cls_full 获得完整图文对
    return img_by_layer, txt_buf, labels


@torch.no_grad()
def extract_layerwise_cls_full(
    clip_model: nn.Module,
    loader,
    classnames: List[str],
    template: str,
    num_pairs: int,
    strategy: Strategy = "cls_eos",
    device: str = "cuda",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
    """
    分类数据集：同时提取图像侧与文本侧各层嵌入（图文严格配对）。

    文本侧：对每张图对应的 ground-truth 类别名称编码，得到逐样本文本向量。

    Parameters
    ----------
    classnames : 数据集类别名称列表
    template   : 文本模板，如 "a photo of a {}"
    """
    clip_model.eval()
    hook_mgr = LayerwiseHookManager(clip_model)

    n_img = hook_mgr.n_img_layers
    n_txt = hook_mgr.n_txt_layers

    img_buf: Dict[str, List[torch.Tensor]] = {f"img_layer_{i}": [] for i in range(n_img)}
    txt_buf: Dict[str, List[torch.Tensor]] = {f"txt_layer_{i}": [] for i in range(n_txt)}
    lbl_buf: List[torch.Tensor] = []

    # 预先 tokenize 所有类别
    all_texts = [template.format(cn.replace("_", " ")) for cn in classnames]
    all_tokens = clip.tokenize(all_texts, truncate=True).to(device)  # [C, 77]

    collected = 0

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        for images, targets in loader:
            if collected >= num_pairs:
                break

            images  = images.to(device)
            targets = targets.to(device)
            remaining = num_pairs - collected
            cut = min(images.shape[0], remaining)

            # ------ 图像 forward ------
            _ = clip_model.encode_image(images)

            for i in range(n_img):
                key  = f"img_layer_{i}"
                raw  = hook_mgr.storage[key]
                feat = reduce_image_tokens(raw, strategy)[:cut]
                feat = feat / feat.norm(dim=-1, keepdim=True)
                img_buf[key].append(feat.cpu().float())

            # ------ 文本 forward（仅本 batch 涉及的类别，保持与图像对应） ------
            # 逐样本单独 forward 开销大；batch 中同一类别出现多次时合并
            batch_targets = targets[:cut]
            unique_cls, inverse = torch.unique(batch_targets, return_inverse=True)
            tok_batch = all_tokens[unique_cls]              # [U, 77]

            _ = clip_model.encode_text(tok_batch)

            for i in range(n_txt):
                key  = f"txt_layer_{i}"
                raw  = hook_mgr.storage[key]               # [U, 77, D] 或 [U, D]
                # 先聚合到 [U, D]
                feat_u = reduce_text_tokens(raw, tok_batch, strategy)  # [U, D]
                feat_u = feat_u / feat_u.norm(dim=-1, keepdim=True)
                # 再按 inverse 展开为 [cut, D]（对齐图像顺序）
                feat_b = feat_u[inverse]
                txt_buf[key].append(feat_b.cpu().float())

            lbl_buf.append(batch_targets.cpu())
            collected += cut

    hook_mgr.remove()

    img_by_layer = {k: torch.cat(v, dim=0) for k, v in img_buf.items()}
    txt_by_layer = {k: torch.cat(v, dim=0) for k, v in txt_buf.items()}
    labels       = torch.cat(lbl_buf, dim=0)

    _print_layer_stats(img_by_layer, txt_by_layer, labels, strategy, "cls")
    return img_by_layer, txt_by_layer, labels


# ---------------------------------------------------------------------------
# 逐层嵌入提取：Flickr30k（真实图文对）
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_layerwise_embeddings_flickr(
    clip_model: nn.Module,
    hf_dataset,
    collate_fn,
    preprocess,
    num_pairs: int,
    strategy: Strategy = "cls_eos",
    device: str = "cuda",
    batch_size: int = 32,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
    """
    Flickr30k 真实图文对逐层提取。

    Returns
    -------
    img_by_layer, txt_by_layer, labels
      labels = torch.arange(num_pairs)（每对自身为正样本）
    """
    clip_model.eval()
    hook_mgr = LayerwiseHookManager(clip_model)

    n_img = hook_mgr.n_img_layers
    n_txt = hook_mgr.n_txt_layers

    img_buf: Dict[str, List[torch.Tensor]] = {f"img_layer_{i}": [] for i in range(n_img)}
    txt_buf: Dict[str, List[torch.Tensor]] = {f"txt_layer_{i}": [] for i in range(n_txt)}

    collected = 0
    stream    = iter(hf_dataset)

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        while collected < num_pairs:
            remaining = num_pairs - collected
            take      = min(batch_size, remaining)

            batch_samples = list(itertools.islice(stream, take))
            if not batch_samples:
                print(f"  [flickr|layerwise] 数据集已耗尽，共收集 {collected} 对")
                break

            # ------ 图像 ------
            pil_images  = collate_fn("image", batch_samples)
            img_tensors = torch.stack(
                [preprocess(img) for img in pil_images]
            ).to(device)

            _ = clip_model.encode_image(img_tensors)

            for i in range(n_img):
                key  = f"img_layer_{i}"
                raw  = hook_mgr.storage[key]
                feat = reduce_image_tokens(raw, strategy)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                img_buf[key].append(feat.cpu().float())

            # ------ 文本 ------
            captions = collate_fn("text", batch_samples)
            tok      = clip.tokenize(captions, truncate=True).to(device)  # [b, 77]

            _ = clip_model.encode_text(tok)

            for i in range(n_txt):
                key  = f"txt_layer_{i}"
                raw  = hook_mgr.storage[key]
                feat = reduce_text_tokens(raw, tok, strategy)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                txt_buf[key].append(feat.cpu().float())

            collected += img_tensors.shape[0]

    hook_mgr.remove()

    img_by_layer = {k: torch.cat(v, dim=0) for k, v in img_buf.items()}
    txt_by_layer = {k: torch.cat(v, dim=0) for k, v in txt_buf.items()}
    labels       = torch.arange(sum(v[0].shape[0] for v in img_buf.values()) // n_img
                                if n_img > 0 else 0)
    labels       = torch.arange(img_by_layer[f"img_layer_0"].shape[0])

    _print_layer_stats(img_by_layer, txt_by_layer, labels, strategy, "flickr")
    return img_by_layer, txt_by_layer, labels


# ---------------------------------------------------------------------------
# 打印摘要
# ---------------------------------------------------------------------------

def _print_layer_stats(
    img_by_layer: Dict[str, torch.Tensor],
    txt_by_layer: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    strategy: str,
    tag: str,
) -> None:
    n_img = len(img_by_layer)
    n_txt = len(txt_by_layer)
    sample_img = next(iter(img_by_layer.values()))
    sample_txt = next(iter(txt_by_layer.values()))
    print(f"  [layerwise|{tag}|{strategy}] 总样本数   : {len(labels)}")
    print(f"  [layerwise|{tag}|{strategy}] 图像层数   : {n_img}")
    print(f"  [layerwise|{tag}|{strategy}] 文本层数   : {n_txt}")
    print(f"  [layerwise|{tag}|{strategy}] 图像特征维 : {sample_img.shape[1]}")
    print(f"  [layerwise|{tag}|{strategy}] 文本特征维 : {sample_txt.shape[1]}")