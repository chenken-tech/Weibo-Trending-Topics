#这段代码没有切分音频，直接整段训练
# import os
# import subprocess
# import tempfile
# from pathlib import Path
#
# import librosa
# import numpy as np
# import torch
# from transformers import ClapModel, ClapProcessor
#
# # =========================
# # 配置区
# # =========================
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#
# # 模式:
# # "single"  -> 处理单个 mp4
# # "dataset" -> 批量处理 ROOT_DIR 下每个话题文件夹里的 videos/*.mp4
# MODE = "dataset"
#
# # -------- 单文件模式配置 --------
# VIDEO_PATH = r"E:\dataset\example.mp4"
# OUTPUT_NPY_PATH = r"E:\dataset\example_audio_embedding.npy"
#
# # -------- 批量模式配置 --------
# ROOT_DIR = r"E:\dataset\微博热搜数据集"
#
# # -------- 模型配置 --------
# MODEL_NAME = "laion/clap-htsat-unfused"
# TARGET_SR = 48000
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#
# # 支持的视频后缀
# VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm", ".mpeg", ".mpg"}
#
#
# def check_ffmpeg():
#     """检查 ffmpeg 是否可用"""
#     try:
#         result = subprocess.run(
#             ["ffmpeg", "-version"],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             check=True,
#             text=True
#         )
#         print("ffmpeg 可用：", result.stdout.splitlines()[0])
#     except Exception as e:
#         raise EnvironmentError(
#             "未检测到 ffmpeg，请先安装 ffmpeg 并加入系统 PATH。\n"
#             "Windows 可用 conda install -c conda-forge ffmpeg 或手动安装。"
#         ) from e
#
#
# def extract_audio_from_video(video_path: str, output_wav_path: str, target_sr: int = 48000):
#     """
#     从视频中提取单声道 wav 音频
#     """
#     if not os.path.exists(video_path):
#         raise FileNotFoundError(f"视频不存在: {video_path}")
#
#     cmd = [
#         "ffmpeg",
#         "-y",                # 覆盖输出
#         "-i", video_path,    # 输入视频
#         "-vn",               # 不要视频
#         "-ac", "1",          # 单声道
#         "-ar", str(target_sr),  # 采样率
#         "-f", "wav",
#         output_wav_path
#     ]
#
#     # result = subprocess.run(
#     #     cmd,
#     #     stdout=subprocess.PIPE,
#     #     stderr=subprocess.PIPE,
#     #     text=True
#     # )
#     #
#     # if result.returncode != 0:
#     #     raise RuntimeError(
#     #         f"ffmpeg 提取音频失败: {video_path}\n"
#     #         f"stderr:\n{result.stderr}"
#     #     )
#
#     result = subprocess.run(
#         cmd,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE
#     )
#
#     if result.returncode != 0:
#         raise RuntimeError(
#             f"ffmpeg 提取音频失败: {video_path}\n"
#             f"stderr:\n{result.stderr.decode('utf-8', errors='ignore')}"
#         )
#
#
# def load_audio(audio_path: str, target_sr: int = 48000):
#     """
#     读取音频
#     """
#     if not os.path.exists(audio_path):
#         raise FileNotFoundError(f"音频文件不存在: {audio_path}")
#
#     audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
#     if audio is None or len(audio) == 0:
#         raise ValueError(f"音频为空或读取失败: {audio_path}")
#     return audio, sr
#
#
# # def load_clap_model(model_name: str, device: str):
# #     """
# #     加载 CLAP 模型
# #     """
# #     processor = ClapProcessor.from_pretrained(model_name)
# #     model = ClapModel.from_pretrained(model_name).to(device)
# #     model.eval()
# #     return processor, model
# def load_clap_model(model_name: str, device: str):
#     processor = ClapProcessor.from_pretrained(model_name)
#     model = ClapModel.from_pretrained(
#         model_name,
#         use_safetensors=True
#     ).to(device)
#     model.eval()
#     return processor, model
#
# def get_audio_embedding(audio: np.ndarray, sr: int, processor, model, device: str):
#     """
#     提取单个音频的 embedding
#     """
#     inputs = processor(
#         audios=audio,
#         sampling_rate=sr,
#         return_tensors="pt"
#     )
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#
#     with torch.no_grad():
#         audio_features = model.get_audio_features(**inputs)
#
#     return audio_features.squeeze(0).cpu().numpy()
#
#
# def process_single_video(video_path: str, output_npy_path: str, processor, model, device: str):
#     """
#     处理单个视频:
#     mp4 -> wav -> embedding -> npy
#     """
#     print(f"\n开始处理视频: {video_path}")
#
#     with tempfile.TemporaryDirectory() as tmpdir:
#         wav_path = os.path.join(tmpdir, "temp_audio.wav")
#
#         print("1) 提取音轨...")
#         extract_audio_from_video(video_path, wav_path, target_sr=TARGET_SR)
#
#         print("2) 读取音频...")
#         audio, sr = load_audio(wav_path, target_sr=TARGET_SR)
#
#         print("3) 提取 audio embedding...")
#         audio_embedding = get_audio_embedding(audio, sr, processor, model, device)
#
#         os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
#         np.save(output_npy_path, audio_embedding)
#
#         print(f"已保存 embedding 到: {output_npy_path}")
#         print(f"audio_embedding.shape = {audio_embedding.shape}")
#
#
# def process_dataset(root_dir: str):
#     """
#     批量处理数据集目录
#     假设结构类似：
#     ROOT_DIR/
#       话题A/
#         videos/
#           123.mp4
#           456.mp4
#       话题B/
#         videos/
#           789.mp4
#     输出到：
#       话题A/audioembedding/123.npy
#       话题A/audioembedding/456.npy
#     """
#     print(f"Using device: {DEVICE}")
#     print(f"开始加载模型: {MODEL_NAME}")
#     processor, model = load_clap_model(MODEL_NAME, DEVICE)
#     print("模型加载完成。")
#
#     if not os.path.exists(root_dir):
#         raise FileNotFoundError(f"根目录不存在: {root_dir}")
#
#     topic_folders = [
#         os.path.join(root_dir, name)
#         for name in os.listdir(root_dir)
#         if os.path.isdir(os.path.join(root_dir, name))
#     ]
#
#     print(f"\n共发现 {len(topic_folders)} 个话题文件夹")
#
#     total_videos = 0
#     success_count = 0
#     fail_count = 0
#
#     for topic_dir in topic_folders:
#         print(f"\n{'=' * 80}")
#         print(f"当前话题文件夹: {topic_dir}")
#
#         videos_dir = os.path.join(topic_dir, "videos")
#         if not os.path.exists(videos_dir) or not os.path.isdir(videos_dir):
#             print(f"未找到 videos 文件夹，跳过: {topic_dir}")
#             continue
#
#         embedding_dir = os.path.join(topic_dir, "audioembedding")
#         os.makedirs(embedding_dir, exist_ok=True)
#         print(f"audioembedding 文件夹已就绪: {embedding_dir}")
#
#         video_files = [
#             f for f in os.listdir(videos_dir)
#             if os.path.isfile(os.path.join(videos_dir, f))
#             and os.path.splitext(f)[1].lower() in VIDEO_EXTS
#         ]
#
#         print(f"发现 {len(video_files)} 个视频文件")
#         total_videos += len(video_files)
#
#         for video_file in video_files:
#             video_path = os.path.join(videos_dir, video_file)
#             video_id = os.path.splitext(video_file)[0]
#             output_npy_path = os.path.join(embedding_dir, f"{video_id}.npy")
#
#             try:
#                 process_single_video(video_path, output_npy_path, processor, model, DEVICE)
#                 success_count += 1
#             except Exception as e:
#                 fail_count += 1
#                 print(f"处理失败: {video_path}")
#                 print(f"错误原因: {e}")
#
#     print(f"\n{'=' * 80}")
#     print("全部处理完成")
#     print(f"视频总数: {total_videos}")
#     print(f"成功数: {success_count}")
#     print(f"失败数: {fail_count}")
#
#
# def process_single():
#     """
#     单文件模式
#     """
#     print(f"Using device: {DEVICE}")
#     print(f"开始加载模型: {MODEL_NAME}")
#     processor, model = load_clap_model(MODEL_NAME, DEVICE)
#     print("模型加载完成。")
#
#     process_single_video(VIDEO_PATH, OUTPUT_NPY_PATH, processor, model, DEVICE)
#
#
# if __name__ == "__main__":
#     check_ffmpeg()
#
#     if MODE == "single":
#         process_single()
#     elif MODE == "dataset":
#         process_dataset(ROOT_DIR)
#     else:
#         raise ValueError("MODE 只能是 'single' 或 'dataset'")


import os
import subprocess
import tempfile

import librosa
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

# =========================
# 配置区
# =========================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 模式:
# "single"  -> 处理单个 mp4
# "dataset" -> 批量处理 ROOT_DIR 下每个话题文件夹里的 videos/*.mp4
MODE = "dataset"

# -------- 单文件模式配置 --------
VIDEO_PATH = r"E:\dataset\example.mp4"
OUTPUT_NPY_PATH = r"E:\dataset\example_audio_embedding.npy"

# -------- 批量模式配置 --------
ROOT_DIR = r"F:/微博热搜数据集"

# -------- 模型配置 --------
MODEL_NAME = "laion/clap-htsat-unfused"
TARGET_SR = 48000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- 音频切片配置 --------
SEGMENT_SECONDS = 10          # 每段固定 10 秒
HOP_SECONDS = 10              # 步长 10 秒（无重叠）
POOLING = "mean"              # 视频级聚合方式: "mean" 或 "max"
SAVE_SEGMENT_EMBEDDINGS = False   # 是否额外保存每段 embedding

# 支持的视频后缀
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm", ".mpeg", ".mpg"}


def check_ffmpeg():
    """检查 ffmpeg 是否可用"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        first_line = result.stdout.decode("utf-8", errors="ignore").splitlines()[0]
        print("ffmpeg 可用：", first_line)
    except Exception as e:
        raise EnvironmentError(
            "未检测到 ffmpeg，请先安装 ffmpeg 并加入系统 PATH。\n"
            "Windows 可用 conda install -c conda-forge ffmpeg 或手动安装。"
        ) from e


def extract_audio_from_video(video_path: str, output_wav_path: str, target_sr: int = 48000):
    """
    从视频中提取单声道 wav 音频
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频不存在: {video_path}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", str(target_sr),
        "-f", "wav",
        output_wav_path
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 提取音频失败: {video_path}\n"
            f"stderr:\n{result.stderr.decode('utf-8', errors='ignore')}"
        )


def load_audio(audio_path: str, target_sr: int = 48000):
    """
    读取音频
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    if audio is None or len(audio) == 0:
        raise ValueError(f"音频为空或读取失败: {audio_path}")
    return audio, sr


def load_clap_model(model_name: str, device: str):
    """
    加载 CLAP 模型
    """
    processor = ClapProcessor.from_pretrained(model_name)
    model = ClapModel.from_pretrained(
        model_name,
        use_safetensors=True
    ).to(device)
    model.eval()
    return processor, model


def split_audio_into_segments(
    audio: np.ndarray,
    sr: int,
    segment_seconds: int = 10,
    hop_seconds: int = 10
):
    """
    将整段音频切成固定长度片段
    - 不足一段时补零
    - 最后一段不足也补零
    """
    if audio.ndim != 1:
        audio = audio.flatten()

    segment_samples = int(segment_seconds * sr)
    hop_samples = int(hop_seconds * sr)

    if segment_samples <= 0 or hop_samples <= 0:
        raise ValueError("segment_seconds 和 hop_seconds 必须大于 0")

    total_samples = len(audio)

    # 如果整段音频不足一个片段，直接补零成一个片段
    if total_samples <= segment_samples:
        padded = np.zeros(segment_samples, dtype=np.float32)
        padded[:total_samples] = audio.astype(np.float32)
        return [padded]

    segments = []
    start = 0

    while start < total_samples:
        end = start + segment_samples
        segment = audio[start:end]

        # 最后一段不足 segment_samples，补零
        if len(segment) < segment_samples:
            padded = np.zeros(segment_samples, dtype=np.float32)
            padded[:len(segment)] = segment.astype(np.float32)
            segment = padded
        else:
            segment = segment.astype(np.float32)

        segments.append(segment)

        if end >= total_samples:
            break

        start += hop_samples

    return segments


def get_audio_embedding(audio: np.ndarray, sr: int, processor, model, device: str):
    """
    提取单个音频片段的 embedding
    """
    inputs = processor(
        audios=audio,
        sampling_rate=sr,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        audio_features = model.get_audio_features(**inputs)

    return audio_features.squeeze(0).cpu().numpy()


def aggregate_segment_embeddings(segment_embeddings: list, pooling: str = "mean"):
    """
    将多个片段 embedding 聚合为视频级 embedding
    """
    if len(segment_embeddings) == 0:
        raise ValueError("segment_embeddings 为空，无法聚合")

    segment_embeddings = np.stack(segment_embeddings, axis=0)  # [num_segments, dim]

    if pooling == "mean":
        return np.mean(segment_embeddings, axis=0)
    elif pooling == "max":
        return np.max(segment_embeddings, axis=0)
    else:
        raise ValueError("pooling 只能是 'mean' 或 'max'")


def process_single_video(video_path: str, output_npy_path: str, processor, model, device: str):
    """
    处理单个视频:
    mp4 -> wav -> 分段 -> 每段 embedding -> 池化 -> npy
    """
    print(f"\n开始处理视频: {video_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "temp_audio.wav")

        print("1) 提取音轨...")
        extract_audio_from_video(video_path, wav_path, target_sr=TARGET_SR)

        print("2) 读取音频...")
        audio, sr = load_audio(wav_path, target_sr=TARGET_SR)

        print("3) 切分音频片段...")
        segments = split_audio_into_segments(
            audio,
            sr,
            segment_seconds=SEGMENT_SECONDS,
            hop_seconds=HOP_SECONDS
        )
        print(f"共切分得到 {len(segments)} 个片段")

        print("4) 提取每个片段的 audio embedding...")
        segment_embeddings = []
        for idx, segment in enumerate(segments):
            emb = get_audio_embedding(segment, sr, processor, model, device)
            segment_embeddings.append(emb)
            print(f"   片段 {idx + 1}/{len(segments)} 提取完成，shape = {emb.shape}")

        print("5) 聚合为视频级 embedding...")
        video_embedding = aggregate_segment_embeddings(
            segment_embeddings,
            pooling=POOLING
        )

        os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
        np.save(output_npy_path, video_embedding)

        print(f"已保存视频级 embedding 到: {output_npy_path}")
        print(f"video_embedding.shape = {video_embedding.shape}")

        # 可选：保存每段 embedding
        if SAVE_SEGMENT_EMBEDDINGS:
            segment_dir = os.path.splitext(output_npy_path)[0] + "_segments"
            os.makedirs(segment_dir, exist_ok=True)

            for idx, emb in enumerate(segment_embeddings):
                seg_path = os.path.join(segment_dir, f"segment_{idx:03d}.npy")
                np.save(seg_path, emb)

            print(f"每段 embedding 已保存到: {segment_dir}")


def process_dataset(root_dir: str):
    """
    批量处理数据集目录
    假设结构类似：
    ROOT_DIR/
      话题A/
        videos/
          123.mp4
          456.mp4
      话题B/
        videos/
          789.mp4
    输出到：
      话题A/audioembedding/123.npy
      话题A/audioembedding/456.npy
    """
    print(f"Using device: {DEVICE}")
    print(f"开始加载模型: {MODEL_NAME}")
    processor, model = load_clap_model(MODEL_NAME, DEVICE)
    print("模型加载完成。")

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"根目录不存在: {root_dir}")

    topic_folders = [
        os.path.join(root_dir, name)
        for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    ]

    print(f"\n共发现 {len(topic_folders)} 个话题文件夹")

    total_videos = 0
    success_count = 0
    fail_count = 0

    for topic_dir in topic_folders:
        print(f"\n{'=' * 80}")
        print(f"当前话题文件夹: {topic_dir}")

        videos_dir = os.path.join(topic_dir, "videos")
        if not os.path.exists(videos_dir) or not os.path.isdir(videos_dir):
            print(f"未找到 videos 文件夹，跳过: {topic_dir}")
            continue

        embedding_dir = os.path.join(topic_dir, "audioembedding")
        os.makedirs(embedding_dir, exist_ok=True)
        print(f"audioembedding 文件夹已就绪: {embedding_dir}")

        video_files = [
            f for f in os.listdir(videos_dir)
            if os.path.isfile(os.path.join(videos_dir, f))
            and os.path.splitext(f)[1].lower() in VIDEO_EXTS
        ]

        print(f"发现 {len(video_files)} 个视频文件")
        total_videos += len(video_files)

        for video_file in video_files:
            video_path = os.path.join(videos_dir, video_file)
            video_id = os.path.splitext(video_file)[0]
            output_npy_path = os.path.join(embedding_dir, f"{video_id}.npy")

            try:
                process_single_video(video_path, output_npy_path, processor, model, DEVICE)
                success_count += 1
            except Exception as e:
                fail_count += 1
                print(f"处理失败: {video_path}")
                print(f"错误原因: {e}")

    print(f"\n{'=' * 80}")
    print("全部处理完成")
    print(f"视频总数: {total_videos}")
    print(f"成功数: {success_count}")
    print(f"失败数: {fail_count}")


def process_single():
    """
    单文件模式
    """
    print(f"Using device: {DEVICE}")
    print(f"开始加载模型: {MODEL_NAME}")
    processor, model = load_clap_model(MODEL_NAME, DEVICE)
    print("模型加载完成。")

    process_single_video(VIDEO_PATH, OUTPUT_NPY_PATH, processor, model, DEVICE)


if __name__ == "__main__":
    check_ffmpeg()

    if MODE == "single":
        process_single()
    elif MODE == "dataset":
        process_dataset(ROOT_DIR)
    else:
        raise ValueError("MODE 只能是 'single' 或 'dataset'")