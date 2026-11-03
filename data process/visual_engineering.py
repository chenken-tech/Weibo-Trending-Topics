# -*- coding: utf-8 -*-
r"""
图片 embedding —— 与 VITtest.py（videoembedding）同一套写法，只把"视频抽帧"换成"多图"。

规格（与 videoembedding/<id>.npy 完全一致，已核对）
----------------------------------------------------
    模型        google/vit-base-patch16-224-in21k
    取哪一层    outputs.last_hidden_state[:, 0, :]   → CLS token，768 维
    多图融合    np.mean(..., axis=0)  —— 平均融合（和视频的 8 帧平均同理）
    落盘        np.save(..., float32)   768 维 = 3072 字节数据 + 128 字节头 = 3200 字节
    输出路径    <话题>/imageembedding/<帖子id>.npy   （一个帖子一个向量）

输入：<话题>/image/ 下的图片，文件名形如 <帖子id>_<序号>.<后缀>
      （由 fetch_images.py 下载，帖子的多张图靠前缀 id 归组）

输入/输出示例
-------------
    E:\dataset\weibodataset\%23某某话题%23\
        ├── image\            5288511973622096_0.jpg
        │                     5288511973622096_1.jpg
        │                     5288802646231686_0.jpg
        └── imageembedding\   5288511973622096.npy     ← 这个帖子的 2 张图平均融合
                              5288802646231686.npy

说明
----
* 已经存在的 <id>.npy 默认跳过，所以随时可以中断、重跑接着做（OVERWRITE=True 可强制重算）。
* 读不出来的单张图会被跳过并用剩下的图求均值；整个帖子一张都读不到才算失败。
* imageembedding/ 里只写 .npy，不放图片。
"""

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import re
import glob
import numpy as np
from PIL import Image, ImageOps
import torch
from transformers import ViTImageProcessor, ViTModel

# =========================
# 配置区（按需修改）
# =========================
ROOT_DIR = r"E:\dataset\weibodataset"      # 数据集根目录，下面每个文件夹是一个话题
IMAGE_DIR_NAME = "image"                   # 话题里图片所在目录
EMBED_DIR_NAME = "imageembedding"          # 输出目录（只放 .npy）

MODEL_NAME = "google/vit-base-patch16-224-in21k"
BATCH_SIZE = 32                            # 一次送进模型的图片数（显存不够就调小）
OVERWRITE = False                          # True = 重算已存在的 .npy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 支持的图片后缀
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

# 文件名 <帖子id>_<序号>.<后缀>；也兼容没有序号的 <帖子id>.<后缀>
RE_NAME = re.compile(r"^(?P<post_id>\d+)(?:_(?P<seq>\d+))?$")


# =========================
# 模型
# =========================
def load_vit_model(model_name: str, device: str):
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


def load_image(path: str) -> Image.Image:
    """统一成 RGB：处理 EXIF 旋转、CMYK/P 模式、带透明通道的 PNG。"""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                im = bg
            else:
                im = im.convert("RGB")
        return im.copy()


def get_image_embeddings(images, processor, model, device: str):
    """对一批图片取 CLS token embedding，返回 [N, 768]。"""
    outputs = []
    for i in range(0, len(images), BATCH_SIZE):
        chunk = images[i:i + BATCH_SIZE]
        inputs = processor(images=chunk, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            emb = out.last_hidden_state[:, 0, :]        # [B, 768]
        outputs.append(emb.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def get_post_embedding(image_embeddings: np.ndarray):
    """多图平均融合，得到帖子级向量（对应 VITtest.py 里的 get_video_embedding）。"""
    return np.mean(image_embeddings, axis=0)


# =========================
# 单个帖子的多张图 → 一个向量
# =========================
def process_single_post(post_id: str, image_paths, output_npy_path: str,
                        processor, model, device: str):
    images = []
    used = 0
    for p in image_paths:
        try:
            images.append(load_image(p))
            used += 1
        except Exception as e:
            print(f"  [跳过坏图] {os.path.basename(p)}: {e}")

    if not images:
        raise RuntimeError("该帖子的图片全部读取失败")

    image_embeddings = get_image_embeddings(images, processor, model, device)
    post_embedding = get_post_embedding(image_embeddings)

    os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
    np.save(output_npy_path, post_embedding)
    return used, post_embedding.shape


# =========================
# 整个数据集
# =========================
def process_dataset(root_dir: str):
    print(f"Using device: {DEVICE}")
    print(f"开始加载模型: {MODEL_NAME}")
    processor, model = load_vit_model(MODEL_NAME, DEVICE)
    print("模型加载完成。")

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"根目录不存在: {root_dir}")

    topic_folders = [
        os.path.join(root_dir, name)
        for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    ]

    print(f"\n共发现 {len(topic_folders)} 个话题文件夹")

    total_posts = 0        # 有图的帖子数
    total_images = 0       # 图片总数
    success_count = 0
    skip_count = 0
    fail_count = 0

    for topic_dir in topic_folders:
        print(f"\n{'=' * 80}")
        print(f"当前话题文件夹: {topic_dir}")

        image_dir = os.path.join(topic_dir, IMAGE_DIR_NAME)
        if not os.path.exists(image_dir) or not os.path.isdir(image_dir):
            print(f"未找到 {IMAGE_DIR_NAME} 文件夹，跳过: {topic_dir}")
            continue

        embedding_dir = os.path.join(topic_dir, EMBED_DIR_NAME)
        os.makedirs(embedding_dir, exist_ok=True)

        # 把图片按帖子 id 归组
        groups = {}
        bad_name = 0
        for path in sorted(glob.glob(os.path.join(image_dir, "*"))):
            if not os.path.isfile(path):
                continue
            if os.path.splitext(path)[1].lower() not in IMAGE_EXTS:
                continue
            stem = os.path.splitext(os.path.basename(path))[0]
            m = RE_NAME.match(stem)
            if not m:
                bad_name += 1
                continue
            groups.setdefault(m.group("post_id"), []).append(path)

        print(f"发现 {len(groups)} 个有图的帖子，图片共 "
              f"{sum(len(v) for v in groups.values())} 张"
              + (f"（另有 {bad_name} 个文件名不符合 <id>_<序号>，已跳过）" if bad_name else ""))
        total_posts += len(groups)
        total_images += sum(len(v) for v in groups.values())

        for post_id, paths in groups.items():
            output_npy_path = os.path.join(embedding_dir, f"{post_id}.npy")

            if os.path.exists(output_npy_path) and not OVERWRITE:
                skip_count += 1
                continue

            try:
                used, shape = process_single_post(post_id, paths, output_npy_path,
                                                  processor, model, DEVICE)
                success_count += 1
                if success_count % 50 == 0:
                    print(f"  已生成 {success_count} 个 .npy（最近一个：{post_id} "
                          f"用 {used} 张图，shape={shape}）")
            except Exception as e:
                fail_count += 1
                print(f"处理失败: {post_id}  图片数={len(paths)}")
                print(f"错误原因: {e}")

    print(f"\n{'=' * 80}")
    print("全部处理完成")
    print(f"有图帖子数: {total_posts}")
    print(f"图片总数:   {total_images}")
    print(f"新生成 .npy: {success_count}")
    print(f"跳过(已有):  {skip_count}")
    print(f"失败:        {fail_count}")
    print(f"输出目录:   <话题>/{EMBED_DIR_NAME}/<帖子id>.npy   （768 维 float32）")


if __name__ == "__main__":
    process_dataset(ROOT_DIR)
