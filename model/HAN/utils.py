import re
import random
import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import dgl


def clean_path(s: str) -> Path:
    s = str(s).strip().replace("\ue000", "").replace("\ufeff", "")
    return Path(s.strip('"').strip("'"))

def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dgl.random.seed(seed)

def safe_read_csv(csv_path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(csv_path)

def parse_popularity(pop_path: Path) -> Dict[str, float]:
    result = {}
    if not pop_path.exists():
        return result
    text = pop_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "：" in line:
            k, v = line.split("：", 1)
        elif ":" in line:
            k, v = line.split(":", 1)
        else:
            continue
        v = re.sub(r"[^\d.]", "", v)
        result[k.strip()] = float(v) if v else 0.0
    return result

def normalize_topic_name(name: str) -> str:
    return name.replace("%23", "#")

def find_topic_csv(topic_dir: Path) -> Optional[Path]:
    entropy_csv = topic_dir / "entropy_selected.csv"
    if entropy_csv.exists():
        return entropy_csv
    csvs = list(topic_dir.glob("*.csv"))
    if not csvs:
        return None
    raw = [p for p in csvs if "_scored" not in p.name and "_selected" not in p.name and "entropy_selected" not in p.name]
    return raw[0] if raw else csvs[0]

def to_numeric_series(df: pd.DataFrame, col_candidates: List[str], default=0.0) -> pd.Series:
    for col in col_candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)
    return pd.Series(np.full(len(df), default), index=df.index, dtype=float)

def log1p_norm(x: pd.Series) -> np.ndarray:
    x = pd.to_numeric(x, errors="coerce").fillna(0).clip(lower=0).astype(float)
    y = np.log1p(x)
    mn, mx = y.min(), y.max()
    if mx - mn < 1e-12:
        return np.zeros(len(y), dtype=np.float32)
    return ((y - mn) / (mx - mn)).astype(np.float32).values

def minmax_norm_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)

def minmax_norm(x: pd.Series) -> np.ndarray:
    x = pd.to_numeric(x, errors="coerce").fillna(0).astype(float)
    return minmax_norm_array(x.values)

def auth_to_id(x) -> int:
    if pd.isna(x):
        return 0
    s = str(x).strip()
    mapping = {"": 0, "nan": 0, "None": 0, "普通用户": 0, "黄V": 1, "红V": 2, "蓝V": 3, "金V": 4,
               "微博个人认证": 1, "个人认证": 1, "微博机构认证": 3, "机构认证": 3, "微博官方认证": 3, "官方认证": 3,
               "蓝V认证": 3}
    return mapping.get(s, 5)

def member_type_to_id(x) -> int:
    if pd.isna(x):
        return 0
    s = str(x).strip()
    mapping = {"": 0, "nan": 0, "None": 0, "非会员": 0, "会员": 1, "超级会员": 2}
    return mapping.get(s, 3)

def one_hot(ids: np.ndarray, num_classes: int) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    ids = np.clip(ids, 0, num_classes - 1)
    out = np.zeros((len(ids), num_classes), dtype=np.float32)
    out[np.arange(len(ids)), ids] = 1.0
    return out

def make_fixed_dim(vec: Optional[np.ndarray], dim: int) -> np.ndarray:
    if vec is None:
        return np.zeros(dim, dtype=np.float32)
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    if len(vec) == dim:
        return vec
    if len(vec) > dim:
        return vec[:dim]
    out = np.zeros(dim, dtype=np.float32)
    out[:len(vec)] = vec
    return out

def load_npy_by_post_id(folder: Path, post_id: str) -> Optional[np.ndarray]:
    if not folder.exists():
        return None
    candidates = [folder / f"{post_id}.npy"]
    try:
        candidates.append(folder / f"{int(float(post_id))}.npy")
    except Exception:
        pass
    for p in candidates:
        if p.exists():
            try:
                return np.asarray(np.load(p), dtype=np.float32).reshape(-1)
            except Exception:
                return None
    return None

def infer_embedding_dim(topic_dirs: List[Path], folder_names: List[str], fallback_dim: int) -> int:
    for td in topic_dirs:
        for fn in folder_names:
            folder = td / fn
            if folder.exists():
                files = list(folder.glob("*.npy"))
                if files:
                    try:
                        return int(np.load(files[0]).reshape(-1).shape[0])
                    except Exception:
                        pass
    return fallback_dim

def infer_topic_embedding_dim(topic_dirs: List[Path], fallback_dim: int = 768) -> int:
    for td in topic_dirs:
        for name in ["topic_embedding.npy", "topicembedding.npy", "topic_semantic.npy"]:
            f = td / name
            if f.exists():
                try:
                    return int(np.load(f).reshape(-1).shape[0])
                except Exception:
                    pass
    return fallback_dim

def load_topic_embedding(topic_dir: Path, topic_sem_dim: int) -> np.ndarray:
    for name in ["topic_embedding.npy", "topicembedding.npy", "topic_semantic.npy"]:
        f = topic_dir / name
        if f.exists():
            try:
                return make_fixed_dim(np.load(f), topic_sem_dim)
            except Exception:
                pass
    return make_fixed_dim(None, topic_sem_dim)

def get_user_ids(df: pd.DataFrame) -> List[str]:
    for col in ["user_id", "用户id", "用户ID", "用户昵称", "昵称"]:
        if col in df.columns:
            return df[col].fillna("unknown").astype(str).tolist()
    return [f"user_{i}" for i in range(len(df))]