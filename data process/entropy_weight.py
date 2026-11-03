# -*- coding: utf-8 -*-
"""
filter_dataset_by_entropy_weight.py

功能：
对微博热搜数据集中的每个话题 CSV：
1. 以该 CSV 中最早的有效“发布时间”为起点；
2. 仅保留起点之后前 N 小时内的帖子；
3. 基于以下六个正向指标使用熵值法计算指标权重：
   - 转发数
   - 评论数
   - 点赞数
   - 主页关注数
   - 主页粉丝数
   - 主页转评赞数
4. 计算每条帖子的综合熵权得分；
5. 按综合得分从高到低保留指定比例的帖子。

输出：
- 保留原始话题文件夹结构；
- 复制 popularity.txt；
- 复制各类 embedding 文件夹中被保留 post_id 对应的文件；
- 保存筛选后的 CSV；
- 保存 entropy_select_info.json；
- 保存数据集级 entropy_filter_summary.json。

注意：
1. 不会修改原始数据集。
2. videos 文件夹默认不复制，避免输出目录过大。
3. --hours 控制时间窗口，例如 24 表示仅使用话题最早帖子发布后 24 小时内的数据。
4. --keep_ratio 控制窗口内帖子最终保留比例，例如 0.3 表示保留熵权得分最高的 30%。

Windows 运行示例：
python entropyweighttime.py --src "E:/dataset/结果文件" --dst "E:/dataset/结果文件_entropy_top30_24h" --hours 24 --keep_ratio 0.3 --overwrite
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_KEEP_RATIO = 0.3
DEFAULT_HOURS = 24.0
DEFAULT_TIME_COL = "发布时间"
MIN_KEEP = 5
EPS = 1e-12

SKIP_DIR_NAMES = {"videos", "video", "Videos", "Video"}

EMBEDDING_DIR_NAMES = {
    "videoembedding",
    "imageembedding",
    "audioembedding",
    "textembedding",
    "text_embeddings",
    "bertembedding",
    "contextembedding",
    "context_text_for_check",
}

# (输出指标名称, 可能出现的列名, 输出列后缀)
FEATURE_SPECS: Sequence[Tuple[str, Sequence[str], str]] = (
    ("转发数", ("转发数", "reposts", "repost_count"), "repost"),
    ("评论数", ("评论数", "comments", "comment_count"), "comment"),
    ("点赞数", ("点赞数", "likes", "like_count"), "like"),
    ("主页关注数", ("主页关注数", "关注数", "following_count"), "home_follow"),
    ("主页粉丝数", ("主页粉丝数", "粉丝数", "followers_count"), "home_fan"),
    ("主页转评赞数", ("主页转评赞数", "转评赞数", "total_interaction_count"), "home_interaction"),
)


def safe_read_csv(csv_path: Path) -> pd.DataFrame:
    """尝试使用常见中文编码读取 CSV。"""
    last_error: Optional[Exception] = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError(f"CSV 读取失败：{csv_path}") from last_error


def find_topic_csv(topic_dir: Path) -> Optional[Path]:
    csvs = sorted(topic_dir.glob("*.csv"))
    if not csvs:
        return None

    raw = [
        p
        for p in csvs
        if "_scored" not in p.name
        and "_selected" not in p.name
        and "entropy_selected" not in p.name
    ]
    return raw[0] if raw else csvs[0]


def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def get_numeric_col(df: pd.DataFrame, candidates: Sequence[str]) -> Tuple[pd.Series, Optional[str]]:
    """读取非负数值列；缺失列按 0 处理，并返回实际匹配到的列名。"""
    col = find_col(df, candidates)
    if col is None:
        values = pd.Series(np.zeros(len(df)), index=df.index, dtype=np.float64)
        return values, None

    values = (
        pd.to_numeric(df[col], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .astype(np.float64)
    )
    return values, col


def filter_first_n_hours(
    df: pd.DataFrame,
    hours: float,
    time_col: str = DEFAULT_TIME_COL,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    以最早有效发布时间为起点，截取其后前 N 小时内的数据。

    时间窗口为：[最早发布时间, 最早发布时间 + N 小时]。
    无法解析发布时间的记录不参与后续熵权计算。
    """
    if hours <= 0:
        raise ValueError(f"hours 必须大于 0，当前值为：{hours}")
    if time_col not in df.columns:
        raise KeyError(f"CSV 缺少时间列：{time_col}")

    parsed_time = pd.to_datetime(df[time_col], errors="coerce")
    valid_mask = parsed_time.notna()
    valid_count = int(valid_mask.sum())

    if valid_count == 0:
        raise ValueError(f"时间列 {time_col} 中没有可解析的有效时间")

    start_time = parsed_time.loc[valid_mask].min()
    end_time = start_time + pd.Timedelta(hours=float(hours))
    window_mask = valid_mask & parsed_time.ge(start_time) & parsed_time.le(end_time)

    filtered_df = df.loc[window_mask].copy().reset_index(drop=True)

    info: Dict[str, object] = {
        "time_column": time_col,
        "hours": float(hours),
        "window_start": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "window_end": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_count": int(len(df)),
        "valid_time_count": valid_count,
        "invalid_time_count": int((~valid_mask).sum()),
        "window_count": int(len(filtered_df)),
    }
    return filtered_df, info


def entropy_weight_method(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    使用六个正向指标计算熵权及每条记录的综合得分。

    处理逻辑：
    1. 六个指标分别做 Min-Max 归一化；
    2. 计算各指标信息熵；
    3. 计算差异系数及熵权；
    4. 使用归一化指标与熵权加权求和。

    对窗口内为常数的指标，将其信息熵视为 1、差异系数视为 0，
    避免没有区分能力的指标获得虚高权重。
    """
    out = df.copy()
    feature_values: List[np.ndarray] = []
    actual_columns: Dict[str, Optional[str]] = {}

    for output_name, candidates, _suffix in FEATURE_SPECS:
        values, actual_col = get_numeric_col(out, candidates)
        feature_values.append(values.to_numpy(dtype=np.float64))
        actual_columns[output_name] = actual_col

    X_raw = np.column_stack(feature_values).astype(np.float64)
    n, m = X_raw.shape

    if n == 0:
        raise ValueError("用于熵权计算的数据为空")

    # Min-Max 归一化。常数列保持为 0。
    X_norm = np.zeros_like(X_raw, dtype=np.float64)
    varying_mask = np.zeros(m, dtype=bool)

    for j in range(m):
        col = X_raw[:, j]
        cmin = float(np.min(col))
        cmax = float(np.max(col))
        span = cmax - cmin
        if span > EPS:
            X_norm[:, j] = (col - cmin) / span
            varying_mask[j] = True

    # 只有 1 条样本时无法根据分布计算有效熵差，使用均匀权重。
    if n <= 1:
        entropy = np.ones(m, dtype=np.float64)
        diversity = np.zeros(m, dtype=np.float64)
        weights = np.ones(m, dtype=np.float64) / m
    else:
        P = np.zeros_like(X_norm, dtype=np.float64)

        for j in range(m):
            col_sum = float(X_norm[:, j].sum())
            if varying_mask[j] and col_sum > EPS:
                P[:, j] = X_norm[:, j] / col_sum
            else:
                # 常数列没有区分能力，令其熵为最大值 1。
                P[:, j] = 1.0 / n

        k = 1.0 / np.log(n)
        entropy = -k * np.sum(P * np.log(P + EPS), axis=0)
        entropy = np.clip(entropy, 0.0, 1.0)
        diversity = 1.0 - entropy
        diversity[~varying_mask] = 0.0

        diversity_sum = float(diversity.sum())
        if diversity_sum <= EPS:
            weights = np.ones(m, dtype=np.float64) / m
        else:
            weights = diversity / diversity_sum

    scores = X_norm @ weights

    weight_info: Dict[str, object] = {
        "weights": {},
        "entropy": {},
        "diversity": {},
        "actual_source_columns": actual_columns,
        "missing_source_columns": [
            name for name, actual_col in actual_columns.items() if actual_col is None
        ],
    }

    for j, (output_name, _candidates, suffix) in enumerate(FEATURE_SPECS):
        out[f"entropy_{suffix}_norm"] = X_norm[:, j].astype(np.float32)
        out[f"entropy_weight_{suffix}"] = float(weights[j])

        weight_info["weights"][output_name] = float(weights[j])
        weight_info["entropy"][output_name] = float(entropy[j])
        weight_info["diversity"][output_name] = float(diversity[j])

    out["entropy_score"] = scores.astype(np.float32)
    return out, weight_info


def select_by_entropy(
    df: pd.DataFrame,
    keep_ratio: float,
    min_keep: int = MIN_KEEP,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if not 0 < keep_ratio <= 1:
        raise ValueError(f"keep_ratio 必须位于 (0, 1]，当前值为：{keep_ratio}")
    if min_keep < 1:
        raise ValueError(f"min_keep 必须至少为 1，当前值为：{min_keep}")

    scored_df, weight_info = entropy_weight_method(df)
    scored_df = scored_df.sort_values(
        "entropy_score", ascending=False, kind="mergesort"
    ).reset_index(drop=True)

    window_count = len(scored_df)
    keep_n = max(min_keep, int(np.ceil(window_count * keep_ratio)))
    keep_n = min(keep_n, window_count)

    selected_df = scored_df.iloc[:keep_n].copy().reset_index(drop=True)

    info: Dict[str, object] = {
        "selected_count": int(keep_n),
        "keep_ratio": float(keep_ratio),
        "actual_keep_ratio_in_window": float(keep_n / max(window_count, 1)),
        "entropy_weight": weight_info,
    }
    return selected_df, info


def copy_file_if_exists(src: Path, dst: Path) -> None:
    if src.exists() and src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_selected_embeddings(
    src_topic_dir: Path,
    dst_topic_dir: Path,
    selected_ids: List[str],
) -> None:
    selected_set = set(selected_ids)
    selected_int_set = set()

    for pid in selected_ids:
        try:
            selected_int_set.add(str(int(float(pid))))
        except (TypeError, ValueError, OverflowError):
            pass

    for child in src_topic_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name in SKIP_DIR_NAMES:
            continue

        dst_child = dst_topic_dir / child.name

        if child.name in EMBEDDING_DIR_NAMES:
            dst_child.mkdir(parents=True, exist_ok=True)
            for file_path in child.iterdir():
                if not file_path.is_file():
                    continue
                stem = file_path.stem
                if stem in selected_set or stem in selected_int_set:
                    shutil.copy2(file_path, dst_child / file_path.name)
        else:
            shutil.copytree(child, dst_child, dirs_exist_ok=True)


def process_one_topic(
    src_topic_dir: Path,
    dst_topic_dir: Path,
    keep_ratio: float,
    min_keep: int,
    hours: float,
    time_col: str,
) -> Dict[str, object]:
    csv_path = find_topic_csv(src_topic_dir)
    if csv_path is None:
        return {
            "status": "skipped",
            "reason": "未找到 CSV",
        }

    try:
        df = safe_read_csv(csv_path)

        if "id" not in df.columns:
            return {
                "status": "skipped",
                "reason": "CSV 缺少 id 列",
                "csv": str(csv_path),
            }

        window_df, time_info = filter_first_n_hours(
            df=df,
            hours=hours,
            time_col=time_col,
        )

        if window_df.empty:
            return {
                "status": "skipped",
                "reason": "指定时间窗口内没有数据",
                "csv": str(csv_path),
                **time_info,
            }

        selected_df, select_info = select_by_entropy(
            window_df,
            keep_ratio=keep_ratio,
            min_keep=min_keep,
        )

        dst_topic_dir.mkdir(parents=True, exist_ok=True)

        selected_csv_path = dst_topic_dir / csv_path.name
        selected_df.to_csv(selected_csv_path, index=False, encoding="utf-8-sig")
        selected_df.to_csv(
            dst_topic_dir / "entropy_selected.csv",
            index=False,
            encoding="utf-8-sig",
        )

        copy_file_if_exists(
            src_topic_dir / "popularity.txt",
            dst_topic_dir / "popularity.txt",
        )

        selected_ids = selected_df["id"].astype(str).tolist()
        copy_selected_embeddings(src_topic_dir, dst_topic_dir, selected_ids)

        info: Dict[str, object] = {
            "status": "success",
            "csv": str(csv_path),
            **time_info,
            **select_info,
        }

        with open(
            dst_topic_dir / "entropy_select_info.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(info, file, ensure_ascii=False, indent=2)

        return info

    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "reason": str(exc),
            "csv": str(csv_path),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按话题前 N 小时数据计算六指标熵权并筛选帖子"
    )
    parser.add_argument("--src", type=str, required=True, help="原始数据集目录")
    parser.add_argument("--dst", type=str, required=True, help="筛选后数据集输出目录")
    parser.add_argument(
        "--hours",
        type=float,
        default=DEFAULT_HOURS,
        help="从每个话题最早发布时间起保留的小时数，默认 24",
    )
    parser.add_argument(
        "--time_col",
        type=str,
        default=DEFAULT_TIME_COL,
        help="发布时间列名，默认：发布时间",
    )
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=DEFAULT_KEEP_RATIO,
        help="时间窗口内按熵权得分保留的比例，例如 0.3 表示保留前 30%",
    )
    parser.add_argument(
        "--min_keep",
        type=int,
        default=MIN_KEEP,
        help="每个话题最少保留帖子数",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="目标目录存在时删除并重建",
    )
    args = parser.parse_args()

    if args.hours <= 0:
        parser.error("--hours 必须大于 0")
    if not 0 < args.keep_ratio <= 1:
        parser.error("--keep_ratio 必须位于 (0, 1]")
    if args.min_keep < 1:
        parser.error("--min_keep 必须至少为 1")

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        raise FileNotFoundError(f"源目录不存在：{src_root}")
    if not src_root.is_dir():
        raise NotADirectoryError(f"源路径不是目录：{src_root}")

    if dst_root.exists():
        if args.overwrite:
            print(f"删除已有目录：{dst_root}")
            shutil.rmtree(dst_root)
        else:
            raise FileExistsError(
                f"目标目录已存在：{dst_root}。如需覆盖请加 --overwrite"
            )

    dst_root.mkdir(parents=True, exist_ok=True)
    topic_dirs = sorted(path for path in src_root.iterdir() if path.is_dir())

    summary: Dict[str, object] = {
        "src": str(src_root),
        "dst": str(dst_root),
        "hours": float(args.hours),
        "time_column": args.time_col,
        "keep_ratio": float(args.keep_ratio),
        "min_keep": int(args.min_keep),
        "features": [spec[0] for spec in FEATURE_SPECS],
        "topics": [],
    }

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for topic_dir in tqdm(topic_dirs, desc="Entropy filtering topics"):
        dst_topic_dir = dst_root / topic_dir.name
        info = process_one_topic(
            src_topic_dir=topic_dir,
            dst_topic_dir=dst_topic_dir,
            keep_ratio=args.keep_ratio,
            min_keep=args.min_keep,
            hours=args.hours,
            time_col=args.time_col,
        )

        status = info.get("status")
        if status == "success":
            success_count += 1
        elif status == "skipped":
            skipped_count += 1
        else:
            failed_count += 1

        summary["topics"].append({
            "topic": topic_dir.name,
            **info,
        })

    summary["success_count"] = success_count
    summary["skipped_count"] = skipped_count
    summary["failed_count"] = failed_count

    with open(
        dst_root / "entropy_filter_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("完成！")
    print(f"源数据集：{src_root}")
    print(f"筛选后数据集：{dst_root}")
    print(f"时间窗口：每个话题最早发布时间后的前 {args.hours:g} 小时")
    print(f"处理成功：{success_count}")
    print(f"跳过：{skipped_count}")
    print(f"失败：{failed_count}")
    print(f"窗口内保留比例：{args.keep_ratio}")


if __name__ == "__main__":
    main()
