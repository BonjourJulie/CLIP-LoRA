import os
import torch
import itertools

import clip
from datasets import build_dataset
from datasets.utils import build_data_loader
from utils import clip_classifier
from run_utils import set_random_seed
from loralib.utils import apply_lora, load_lora
from idc import id_correlation
from merge import (
    extract_layerwise_embeddings_flickr,
    extract_layerwise_cls_full,
)


# ---------------------------------------------------------------------------
# 本地 Flickr30k 接口
# ---------------------------------------------------------------------------

def get_flickr(dataset_name, flickr_id=0,
               data_root='/d/hjy/CLIP-LoRA',
               ann_file='annotations/flickr30k_train.json'):

    if dataset_name == 'flickr':
        import json
        from PIL import Image as PILImage

        ann_path = os.path.join(data_root, ann_file)
        with open(ann_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        dataset = []
        for item in raw:
            img_rel = item['image']
            img_path = os.path.join(data_root, img_rel)
            caption  = item['caption']
            if isinstance(caption, list):
                caption = caption[flickr_id]
            dataset.append({'image_path': img_path, 'caption': caption})

        def collate(modality, samples):
            if modality == 'image':
                from PIL import Image as PILImage
                return [PILImage.open(s['image_path']).convert("RGB") for s in samples]
            else:
                return [s['caption'] for s in samples]

        return dataset, collate


# ---------------------------------------------------------------------------
# 辅助：forward hook 截取投影层之前的向量
# ---------------------------------------------------------------------------
def _register_pre_proj_hooks(clip_model):
    storage = {}

    def hook_img(module, input, output):
        storage['img_pre_proj'] = output.detach()

    def hook_txt(module, input, output):
        storage['txt_pre_proj_all'] = output.detach()

    h_img = clip_model.visual.ln_post.register_forward_hook(hook_img)
    h_txt = clip_model.ln_final.register_forward_hook(hook_txt)
    return [h_img, h_txt], storage


# ---------------------------------------------------------------------------
# Flickr30k 专用
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_flickr_embeddings(clip_model, hf_dataset, collate_fn, preprocess,
                               num_pairs, device="cuda", use_pre_proj=False,
                               batch_size=32):
    clip_model.eval()

    if use_pre_proj:
        handles, storage = _register_pre_proj_hooks(clip_model)

    all_img_feats, all_txt_feats = [], []
    collected = 0
    stream = iter(hf_dataset)

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        while collected < num_pairs:
            remaining = num_pairs - collected
            take = min(batch_size, remaining)

            batch_samples = list(itertools.islice(stream, take))
            if len(batch_samples) == 0:
                print(f"  [flickr] 数据集已耗尽，共收集 {collected} 对")
                break

            pil_images  = collate_fn('image', batch_samples)
            img_tensors = torch.stack(
                [preprocess(img) for img in pil_images]
            ).to(device)

            if use_pre_proj:
                _ = clip_model.encode_image(img_tensors)
                img_feats = storage['img_pre_proj'].float()
            else:
                img_feats = clip_model.encode_image(img_tensors).float()
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            captions = collate_fn('text', batch_samples)
            tok = clip.tokenize(captions, truncate=True).to(device)

            if use_pre_proj:
                _ = clip_model.encode_text(tok)
                seq     = storage['txt_pre_proj_all']
                eot_idx = tok.argmax(dim=-1)
                txt_feats = seq[torch.arange(len(tok)), eot_idx].float()
            else:
                txt_feats = clip_model.encode_text(tok).float()
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

            all_img_feats.append(img_feats.cpu())
            all_txt_feats.append(txt_feats.cpu())
            collected += img_feats.shape[0]

    if use_pre_proj:
        for h in handles:
            h.remove()

    image_features = torch.cat(all_img_feats, dim=0)
    text_features  = torch.cat(all_txt_feats, dim=0)
    labels         = torch.arange(len(image_features))

    mode_str = "pre-proj" if use_pre_proj else "post-proj"
    print(f'  [diagnose|flickr|{mode_str}] total pairs        : {len(labels)}')
    print(f'  [diagnose|flickr|{mode_str}] unique text vectors: {len(torch.unique(text_features, dim=0))}')
    print(f'  [diagnose|flickr|{mode_str}] image feat dim     : {image_features.shape[1]}')
    print(f'  [diagnose|flickr|{mode_str}] text  feat dim     : {text_features.shape[1]}')

    return image_features, text_features, labels


# ---------------------------------------------------------------------------
# 分类数据集
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_paired_embeddings(clip_model, loader, dataset, num_pairs, device="cuda",
                               use_pre_proj=False):
    clip_model.eval()

    if use_pre_proj:
        handles, storage = _register_pre_proj_hooks(clip_model)

    all_img_feats, all_txt_feats, all_labels = [], [], []
    collected = 0

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):

        if use_pre_proj:
            template   = dataset.template[0]
            classnames = dataset.classnames
            texts_tok  = clip.tokenize(
                [template.format(cn.replace("_", " ")) for cn in classnames]
            ).to(device)

            cls_feats_list = []
            for i in range(0, len(classnames), 256):
                tok_b = texts_tok[i:i + 256]
                _     = clip_model.encode_text(tok_b)
                seq   = storage['txt_pre_proj_all']
                eot   = tok_b.argmax(dim=-1)
                feat  = seq[torch.arange(len(tok_b)), eot]
                cls_feats_list.append(feat.cpu().float())

            txt_matrix = torch.cat(cls_feats_list, dim=0)
            txt_matrix = txt_matrix / txt_matrix.norm(dim=-1, keepdim=True)
            txt_matrix = txt_matrix.t().to(device)
        else:
            text_weight_matrix = clip_classifier(
                dataset.classnames, dataset.template, clip_model
            )

        for images, targets in loader:
            if collected >= num_pairs:
                break

            images  = images.to(device)
            targets = targets.to(device)

            if use_pre_proj:
                _         = clip_model.encode_image(images)
                img_feats = storage['img_pre_proj'].float()
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                txt_feats = txt_matrix[:, targets].t().float()
            else:
                img_feats = clip_model.encode_image(images).float()
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                txt_feats = text_weight_matrix[:, targets].t()

            remaining = num_pairs - collected
            img_feats = img_feats[:remaining]
            txt_feats = txt_feats[:remaining]
            targets   = targets[:remaining]

            all_img_feats.append(img_feats.cpu().float())
            all_txt_feats.append(txt_feats.cpu().float())
            all_labels.append(targets.cpu())
            collected += img_feats.shape[0]

    if use_pre_proj:
        for h in handles:
            h.remove()

    image_features = torch.cat(all_img_feats, dim=0)
    text_features  = torch.cat(all_txt_feats, dim=0)
    labels         = torch.cat(all_labels,    dim=0)

    mode_str = "pre-proj" if use_pre_proj else "post-proj"
    n_unique_txt = len(torch.unique(text_features, dim=0))
    print(f'  [diagnose|{mode_str}] total samples       : {len(labels)}')
    print(f'  [diagnose|{mode_str}] unique labels       : {len(torch.unique(labels))}')
    print(f'  [diagnose|{mode_str}] image feat dim      : {image_features.shape[1]}')
    print(f'  [diagnose|{mode_str}] text  feat dim      : {text_features.shape[1]}')
    print(f'  [diagnose|{mode_str}] unique text vectors : {n_unique_txt}')
    if n_unique_txt < len(labels):
        print(f'  [diagnose|{mode_str}] WARNING: text point cloud has only {n_unique_txt} '
              f'distinct points => ID(text) may be underestimated')

    return image_features, text_features, labels


# ---------------------------------------------------------------------------
# 保存工具
# ---------------------------------------------------------------------------
def save_embeddings(output_dir, prefix, image_features, text_features, labels):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(image_features, os.path.join(output_dir, f"{prefix}_image_features.pt"))
    torch.save(text_features,  os.path.join(output_dir, f"{prefix}_text_features.pt"))
    torch.save(labels,         os.path.join(output_dir, f"{prefix}_labels.pt"))
    print(f"[{prefix}] 已保存 {image_features.shape[0]} 对嵌入至 {output_dir}")
    print(f"  image_features : {image_features.shape}  dtype={image_features.dtype}")
    print(f"  text_features  : {text_features.shape}   dtype={text_features.dtype}")
    print(f"  labels         : {labels.shape}")


# ---------------------------------------------------------------------------
# ★ 核心新增：计算 N_img_layers × N_txt_layers 的完整 IDC 矩阵
# ---------------------------------------------------------------------------
def compute_cross_layer_idc_matrix(
    img_by_layer: dict,
    txt_by_layer: dict,
    idc_n_permutations: int,
    tag: str,
    output_dir: str,
    strategy: str,
):
    """
    对视觉编码器每一层与文本编码器每一层做全量 IDC 配对，
    生成形状为 (n_img_layers, n_txt_layers) 的 IDC 矩阵。

    img_by_layer: {"img_layer_0": Tensor[N, D], ..., "img_layer_K": Tensor[N, D]}
    txt_by_layer: {"txt_layer_0": Tensor[N, D], ..., "txt_layer_M": Tensor[N, D]}

    返回:
        idc_matrix  : Tensor[n_img_layers, n_txt_layers]  IDC 相关系数
        id_img_vec  : Tensor[n_img_layers]                各视觉层的 ID
        id_txt_vec  : Tensor[n_txt_layers]                各文本层的 ID
        full_results: list[dict]  每个 (i,j) 配对的完整指标，用于保存 CSV
    """
    n_img_layers = len(img_by_layer)
    n_txt_layers = len(txt_by_layer)
    total_pairs  = n_img_layers * n_txt_layers

    print(f"\n=== Cross-Layer IDC 矩阵 [{tag}|{strategy}] ===")
    print(f"  视觉层数={n_img_layers}, 文本层数={n_txt_layers}, "
          f"共 {total_pairs} 个 (img_layer, txt_layer) 配对")

    idc_matrix  = torch.full((n_img_layers, n_txt_layers), float('nan'))
    id_img_vec  = torch.full((n_img_layers,),              float('nan'))
    id_txt_vec  = torch.full((n_txt_layers,),              float('nan'))
    full_results = []

    # ---- 打印表头（列=文本层，行=视觉层） ----
    col_header = "img\\txt |" + "".join(f" {j:>7}" for j in range(n_txt_layers))
    print("\n" + col_header)
    print("-" * len(col_header))

    for img_idx in range(n_img_layers):
        img_feats = img_by_layer[f"img_layer_{img_idx}"]   # [N, D_img]
        row_vals  = []

        for txt_idx in range(n_txt_layers):
            txt_feats = txt_by_layer[f"txt_layer_{txt_idx}"]   # [N, D_txt]

            assert img_feats.shape[0] == txt_feats.shape[0], (
                f"样本数不匹配: img_layer_{img_idx}={img_feats.shape[0]}, "
                f"txt_layer_{txt_idx}={txt_feats.shape[0]}"
            )

            res = id_correlation(
                img_feats, txt_feats,
                N=idc_n_permutations,
                algorithm='twoNN',
                return_pvalue=True,
            )

            id1   = res.get('id1',  None)
            id2   = res.get('id2',  None)
            id_jt = res.get('id',   None)
            corr  = res.get('corr', None)
            pval  = res.get('p',    None)

            corr_val = corr if corr is not None else float('nan')
            idc_matrix[img_idx, txt_idx] = corr_val
            row_vals.append(corr_val)

            # 对角线上顺带记录各自 ID（id1=img 的 ID，id2=txt 的 ID）
            if img_idx == txt_idx:
                if id1 is not None:
                    id_img_vec[img_idx] = id1
                if id2 is not None:
                    id_txt_vec[txt_idx] = id2

            full_results.append({
                'img_layer': img_idx,
                'txt_layer': txt_idx,
                'id_image':  id1,
                'id_text':   id2,
                'id_joint':  id_jt,
                'idc':       corr,
                'p_value':   pval,
                'img_dim':   img_feats.shape[1],
                'txt_dim':   txt_feats.shape[1],
            })

        # 打印当前视觉层的一整行
        row_str = f"  img_{img_idx:>2}  |" + "".join(
            f" {v:>7.4f}" if not torch.isnan(torch.tensor(v)) else "    N/A"
            for v in row_vals
        )
        print(row_str)

    # ---- 保存 CSV ----
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir,
                            f"cross_layer_idc_matrix_{tag}_{strategy}.csv")
    _save_csv(full_results, csv_path)
    print(f"\n  [Cross-Layer IDC] 完整结果已保存至 {csv_path}")

    # ---- 保存矩阵本身为 .pt ----
    pt_path = os.path.join(output_dir,
                           f"cross_layer_idc_matrix_{tag}_{strategy}.pt")
    torch.save({
        'idc_matrix':  idc_matrix,
        'id_img_vec':  id_img_vec,
        'id_txt_vec':  id_txt_vec,
    }, pt_path)
    print(f"  [Cross-Layer IDC] 矩阵张量已保存至 {pt_path}")

    return idc_matrix, id_img_vec, id_txt_vec, full_results


# ---------------------------------------------------------------------------
# ★ 保留原有逐层（对角线）IDC（可选，向下兼容）
# ---------------------------------------------------------------------------
def compute_layerwise_idc(
    img_by_layer,
    txt_by_layer,
    idc_n_permutations: int,
    tag: str,
    output_dir: str,
    strategy: str,
):
    """同层配对 IDC（原有逻辑，保持向下兼容）"""
    n_layers = min(len(img_by_layer), len(txt_by_layer))
    print(f"\n  [layerwise IDC|{tag}|{strategy}] 共 {n_layers} 层配对（对角线）")

    results = []
    header = f"  {'层':>8}  {'ID(img)':>10}  {'ID(txt)':>10}  {'ID(img+txt)':>12}  {'IDC':>10}  {'p-value':>10}"
    print(f"\n=== 逐层 IDC [{tag}|{strategy}] ===")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for layer_idx in range(n_layers):
        img_feats = img_by_layer[f"img_layer_{layer_idx}"]
        txt_feats = txt_by_layer[f"txt_layer_{layer_idx}"]

        idc_res = id_correlation(
            img_feats, txt_feats,
            N=idc_n_permutations,
            algorithm='twoNN',
            return_pvalue=True,
        )

        id1   = idc_res.get('id1',  None)
        id2   = idc_res.get('id2',  None)
        id_jt = idc_res.get('id',   None)
        corr  = idc_res.get('corr', None)
        pval  = idc_res.get('p',    None)

        def fmt(v):
            return f"{v:.4f}" if v is not None else "  N/A "

        print(f"  {layer_idx:>8}  {fmt(id1):>10}  {fmt(id2):>10}  "
              f"{fmt(id_jt):>12}  {fmt(corr):>10}  {fmt(pval):>10}")

        results.append({
            'layer':    layer_idx,
            'id_image': id1,
            'id_text':  id2,
            'id_joint': id_jt,
            'idc':      corr,
            'p_value':  pval,
            'img_dim':  img_feats.shape[1],
            'txt_dim':  txt_feats.shape[1],
        })

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"layerwise_idc_{tag}_{strategy}.csv")
    _save_csv(results, csv_path)
    print(f"\n  [layerwise IDC] 结果已保存至 {csv_path}")
    return results


def _save_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', encoding='utf-8') as f:
        f.write(','.join(keys) + '\n')
        for row in rows:
            f.write(','.join(str(row[k]) for k in keys) + '\n')


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',         default=1,    type=int)
    parser.add_argument('--root_path',    type=str,     default='')
    parser.add_argument('--dataset',      type=str,     default='imagenet')
    parser.add_argument('--shots',        default=16,   type=int)
    parser.add_argument('--backbone',     default='ViT-B/16', type=str)
    parser.add_argument('--lr',           default=2e-4, type=float)
    parser.add_argument('--n_iters',      default=500,  type=int)
    parser.add_argument('--batch_size',   default=32,   type=int)
    parser.add_argument('--position',     type=str,     default='all',
                        choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'])
    parser.add_argument('--encoder',      type=str,     default='both',
                        choices=['text', 'vision', 'both'])
    parser.add_argument('--params',       metavar='N',  type=str, nargs='+',
                        default=['q', 'k', 'v'])
    parser.add_argument('--r',            default=2,    type=int)
    parser.add_argument('--alpha',        default=1,    type=int)
    parser.add_argument('--dropout_rate', default=0.25, type=float)
    parser.add_argument('--save_path',    default=None)
    parser.add_argument('--filename',     default='lora_weights')
    parser.add_argument('--eval_only',    default=False, action='store_true')

    parser.add_argument('--num_pairs',  default=500,  type=int)
    parser.add_argument('--output_dir', default='./embeddings', type=str)
    parser.add_argument('--split',      default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--idc_n_permutations', default=100, type=int)
    parser.add_argument('--use_pre_proj', default=False, action='store_true')
    parser.add_argument('--flickr_caption_id', default=0, type=int, choices=[0,1,2,3,4],
                        help='Flickr30k 每张图有5条caption，选第几条(0-4)，默认0')
    parser.add_argument('--flickr_root', default='/d/zjw/flickr30k', type=str,
                        help='本地 Flickr30k 数据集根目录')

    parser.add_argument('--layerwise',  default=False, action='store_true',
                        help='是否对每一层提取嵌入并计算 IDC（包含对角线+完整矩阵）')
    parser.add_argument('--diagonal_only', default=False, action='store_true',
                        help='仅计算对角线（同层）IDC，不计算完整 12×12 矩阵')
    parser.add_argument('--strategy',   default='cls_eos',
                        choices=['cls_eos', 'mean_pool'],
                        help='token 聚合策略：cls_eos（CLS/EOS）或 mean_pool（平均池化）')

    args = parser.parse_args()
    set_random_seed(args.seed)

    print(f"\n加载 CLIP 模型: {args.backbone}")
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    clip_model = clip_model.cuda()

    # ---- 准备数据 ----
    is_flickr = (args.dataset == 'flickr')

    if is_flickr:
        print(f"准备数据集: flickr30k（caption_id={args.flickr_caption_id}）")
        hf_dataset, collate_fn = get_flickr(
            'flickr', flickr_id=args.flickr_caption_id, data_root=args.flickr_root
        )
        print(f"使用 Flickr30k，提取 {args.num_pairs} 对真实图文对")
    else:
        print(f"准备数据集: {args.dataset}")
        dataset    = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
        split_map  = {'train': dataset.train_x, 'val': dataset.val, 'test': dataset.test}
        split_data = split_map[args.split]
        print(f"使用 split='{args.split}', 提取 {args.num_pairs} 对")

        if args.dataset == 'imagenet':
            loader = torch.utils.data.DataLoader(
                split_data, batch_size=args.batch_size,
                num_workers=4, shuffle=True, pin_memory=True
            )
        else:
            loader = build_data_loader(
                data_source=split_data, batch_size=args.batch_size,
                is_train=False, tfm=preprocess, shuffle=True, num_workers=4
            )

    # ==================================================================
    # 辅助：提取逐层嵌入（共用逻辑，避免重复）
    # ==================================================================
    def _extract_layerwise(clip_mdl, tag_str):
        """返回 (img_by_layer, txt_by_layer)"""
        print(f"\n=== [{tag_str}] 逐层嵌入提取（策略={args.strategy}）===")
        if is_flickr:
            hf_ds_lw, collate_lw = get_flickr(
                'flickr', flickr_id=args.flickr_caption_id, data_root=args.flickr_root
            )
            img_lw, txt_lw, _ = extract_layerwise_embeddings_flickr(
                clip_mdl, hf_ds_lw, collate_lw, preprocess,
                num_pairs=args.num_pairs,
                strategy=args.strategy,
                batch_size=args.batch_size,
            )
        else:
            img_lw, txt_lw, _ = extract_layerwise_cls_full(
                clip_mdl, loader,
                classnames=dataset.classnames,
                template=dataset.template[0],
                num_pairs=args.num_pairs,
                strategy=args.strategy,
            )
        return img_lw, txt_lw

    # ==================================================================
    # 阶段 1：微调前（原始 CLIP）
    # ==================================================================
    print("\n=== [阶段 1] 微调前（原始 CLIP）提取嵌入 ===")
    if is_flickr:
        img_feats_before, txt_feats_before, labels_before = extract_flickr_embeddings(
            clip_model, hf_dataset, collate_fn, preprocess,
            num_pairs=args.num_pairs, use_pre_proj=args.use_pre_proj,
            batch_size=args.batch_size
        )
    else:
        img_feats_before, txt_feats_before, labels_before = extract_paired_embeddings(
            clip_model, loader, dataset, args.num_pairs,
            use_pre_proj=args.use_pre_proj
        )
    save_embeddings(args.output_dir, "before_lora",
                    img_feats_before, txt_feats_before, labels_before)

    # ★ 阶段 1：逐层提取 & IDC（before LoRA）
    if args.layerwise:
        img_lw_before, txt_lw_before = _extract_layerwise(clip_model, "阶段 1")

        if args.diagonal_only:
            # 仅计算同层对角线 IDC（原有行为）
            print("\n  [阶段 1] 计算逐层 IDC（对角线，before LoRA）...")
            compute_layerwise_idc(
                img_lw_before, txt_lw_before,
                idc_n_permutations=args.idc_n_permutations,
                tag="before_lora",
                output_dir=args.output_dir,
                strategy=args.strategy,
            )
        else:
            # ★ 计算完整 N_img × N_txt IDC 矩阵
            print("\n  [阶段 1] 计算完整 Cross-Layer IDC 矩阵（before LoRA）...")
            compute_cross_layer_idc_matrix(
                img_lw_before, txt_lw_before,
                idc_n_permutations=args.idc_n_permutations,
                tag="before_lora",
                output_dir=args.output_dir,
                strategy=args.strategy,
            )

    # ==================================================================
    # 阶段 2：加载 LoRA 权重
    # ==================================================================
    print("\n=== [阶段 2] 加载 LoRA 权重 ===")
    if args.save_path is None:
        raise ValueError("请通过 --save_path 指定 LoRA 权重目录")

    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda()
    load_lora(args, list_lora_layers)
    clip_model.half()
    clip_model.eval()
    print("LoRA 权重加载完成。")

    print("\n=== [阶段 2] 微调后（LoRA CLIP）提取嵌入 ===")
    if is_flickr:
        hf_dataset2, collate_fn2 = get_flickr(
            'flickr', flickr_id=args.flickr_caption_id, data_root=args.flickr_root
        )
        img_feats_after, txt_feats_after, labels_after = extract_flickr_embeddings(
            clip_model, hf_dataset2, collate_fn2, preprocess,
            num_pairs=args.num_pairs, use_pre_proj=args.use_pre_proj,
            batch_size=args.batch_size
        )
    else:
        img_feats_after, txt_feats_after, labels_after = extract_paired_embeddings(
            clip_model, loader, dataset, args.num_pairs,
            use_pre_proj=args.use_pre_proj
        )
    save_embeddings(args.output_dir, "after_lora",
                    img_feats_after, txt_feats_after, labels_after)

    # ★ 阶段 2：逐层提取 & IDC（after LoRA）
    if args.layerwise:
        img_lw_after, txt_lw_after = _extract_layerwise(clip_model, "阶段 2")

        if args.diagonal_only:
            print("\n  [阶段 2] 计算逐层 IDC（对角线，after LoRA）...")
            compute_layerwise_idc(
                img_lw_after, txt_lw_after,
                idc_n_permutations=args.idc_n_permutations,
                tag="after_lora",
                output_dir=args.output_dir,
                strategy=args.strategy,
            )
        else:
            # ★ 计算完整 N_img × N_txt IDC 矩阵
            print("\n  [阶段 2] 计算完整 Cross-Layer IDC 矩阵（after LoRA）...")
            compute_cross_layer_idc_matrix(
                img_lw_after, txt_lw_after,
                idc_n_permutations=args.idc_n_permutations,
                tag="after_lora",
                output_dir=args.output_dir,
                strategy=args.strategy,
            )

    # ==================================================================
    # 余弦相似度统计
    # ==================================================================
    print("\n=== 嵌入统计对比 ===")
    if img_feats_before.shape[1] == txt_feats_before.shape[1]:
        cos_before = (img_feats_before * txt_feats_before).sum(dim=-1)
        cos_after  = (img_feats_after  * txt_feats_after ).sum(dim=-1)
        print(f"  微调前 图文余弦相似度: mean={cos_before.mean():.4f}  std={cos_before.std():.4f}")
        print(f"  微调后 图文余弦相似度: mean={cos_after.mean():.4f}   std={cos_after.std():.4f}")
    else:
        print(f"  投影前模式：图像{img_feats_before.shape[1]}维，"
              f"文本{txt_feats_before.shape[1]}维，维度不同，跳过余弦相似度")

    # ==================================================================
    # 阶段 3：整体 IDC
    # ==================================================================
    print("\n=== [阶段 3] 计算整体 IDC ===")
    print(f"  permutation 次数: {args.idc_n_permutations}")

    print("\n  [before LoRA] 计算图文 IDC ...")
    idc_before = id_correlation(
        img_feats_before, txt_feats_before,
        N=args.idc_n_permutations, algorithm='twoNN', return_pvalue=True,
    )

    print("\n  [after  LoRA] 计算图文 IDC ...")
    idc_after = id_correlation(
        img_feats_after, txt_feats_after,
        N=args.idc_n_permutations, algorithm='twoNN', return_pvalue=True,
    )

    print("\n=== 整体 IDC 结果汇总 ===")
    print(f"  {'指标':<20} {'before LoRA':>15} {'after LoRA':>15}")
    print(f"  {'-'*52}")
    for key, label in [('id1','ID(image)'), ('id2','ID(text)'),
                        ('id','ID(image+text)'), ('corr','IDC'), ('p','p-value')]:
        v_b = idc_before[key]
        v_a = idc_after[key]
        sb  = f"{v_b:.4f}" if v_b is not None else "N/A"
        sa  = f"{v_a:.4f}" if v_a is not None else "N/A"
        print(f"  {label:<20} {sb:>15} {sa:>15}")


if __name__ == '__main__':
    main()