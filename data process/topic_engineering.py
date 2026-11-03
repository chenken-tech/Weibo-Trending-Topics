# -*- coding: utf-8 -*-
"""
generate_topic_embeddings.py

功能：
为每个热搜话题文件夹生成 topic_embedding.npy。
默认使用 sentence-transformers 加载 BAAI/bge-base-zh-v1.5。
每个话题的文本默认来自文件夹名，例如：
%23黄金涨跌逻辑变了%23 -> #黄金涨跌逻辑变了#

运行示例：
python generate_topic_embeddings.py ^
  --dataset_root "E:/dataset/entropy_top20" ^
  --model_name "BAAI/bge-base-zh-v1.5" ^
  --device cuda ^
  --overwrite

如果显存不够：
python generate_topic_embeddings.py --dataset_root "E:/dataset/entropy_top20" --device cpu
"""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm


def clean_topic_name(name: str) -> str:
    s = name.replace("%23", "#")
    s = s.replace("_", " ")
    return s.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="BAAI/bge-base-zh-v1.5")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    topic_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "缺少 sentence-transformers，请先安装：\n"
            "pip install sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple"
        )

    model = SentenceTransformer(args.model_name, device=args.device)

    todo_dirs = []
    texts = []

    for td in topic_dirs:
        out_path = td / "topic_embedding.npy"
        if out_path.exists() and not args.overwrite:
            continue
        todo_dirs.append(td)
        texts.append(clean_topic_name(td.name))

    print(f"需要生成 topic embedding 的话题数：{len(todo_dirs)}")

    if not todo_dirs:
        print("没有需要处理的文件。")
        return

    for start in tqdm(range(0, len(texts), args.batch_size), desc="Encoding topic names"):
        batch_texts = texts[start:start + args.batch_size]
        batch_dirs = todo_dirs[start:start + args.batch_size]

        emb = model.encode(
            batch_texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        emb = np.asarray(emb, dtype=np.float32)

        for td, vec in zip(batch_dirs, emb):
            np.save(td / "topic_embedding.npy", vec)

    print("完成！")
    print(f"数据集目录：{dataset_root}")
    print("每个话题文件夹下已生成 topic_embedding.npy")


if __name__ == "__main__":
    main()
