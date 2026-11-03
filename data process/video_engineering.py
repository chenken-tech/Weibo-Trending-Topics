# import os
# import numpy as np
# import cv2
# from PIL import Image
# import torch
# from transformers import ViTImageProcessor, ViTModel
#
# VIDEO_PATH = "Ed7LeNr4lx08wiezBukE010412007drK0E010.mp4"   # 你已经下载好的本地视频
# FRAME_DIR = "sampled_frames"
# EMBED_PATH = "video_embedding.npy"
#
# MODEL_NAME = "google/vit-base-patch16-224-in21k"
# NUM_FRAMES = 4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#
#
# def sample_frame_indices(total_frames: int, num_frames: int = 4):
#     if total_frames <= 0:
#         raise ValueError("视频总帧数无效。")
#
#     indices = np.linspace(0, total_frames - 1, num_frames)
#     return [int(round(x)) for x in indices]
#
#
# def extract_frames(video_path: str, num_frames: int = 4, save_dir: str = None):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise RuntimeError(f"无法打开视频: {video_path}")
#
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f"视频总帧数: {total_frames}")
#
#     frame_indices = sample_frame_indices(total_frames, num_frames)
#     print(f"采样帧索引: {frame_indices}")
#
#     frames = []
#
#     if save_dir is not None:
#         os.makedirs(save_dir, exist_ok=True)
#
#     for i, idx in enumerate(frame_indices):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         success, frame = cap.read()
#         if not success:
#             cap.release()
#             raise RuntimeError(f"读取第 {idx} 帧失败")
#
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         pil_img = Image.fromarray(frame_rgb)
#         frames.append(pil_img)
#
#         if save_dir is not None:
#             pil_img.save(os.path.join(save_dir, f"frame_{i}_idx_{idx}.jpg"))
#
#     cap.release()
#     return frames
#
#
# def get_frame_embeddings(frames, model_name: str, device: str):
#     processor = ViTImageProcessor.from_pretrained(model_name)
#     model = ViTModel.from_pretrained(model_name).to(device)
#     model.eval()
#
#     inputs = processor(images=frames, return_tensors="pt")
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#
#     with torch.no_grad():
#         outputs = model(**inputs)
#
#         # 推荐用 CLS token
#         frame_embeddings = outputs.last_hidden_state[:, 0, :]   # [4, 768]
#
#         # 如果你想改成 pooler_output，也可以：
#         # frame_embeddings = outputs.pooler_output
#
#     return frame_embeddings.cpu().numpy()
#
#
# def get_video_embedding(frame_embeddings: np.ndarray):
#     return np.mean(frame_embeddings, axis=0)
#
#
# def main():
#     print(f"Using device: {DEVICE}")
#
#     if not os.path.exists(VIDEO_PATH):
#         raise FileNotFoundError(
#             f"未找到本地视频文件: {VIDEO_PATH}\n"
#             f"请先把视频下载到当前目录，并命名为 {VIDEO_PATH}"
#         )
#
#     print("开始抽帧...")
#     frames = extract_frames(VIDEO_PATH, num_frames=NUM_FRAMES, save_dir=FRAME_DIR)
#
#     print("开始提取帧 embedding...")
#     frame_embeddings = get_frame_embeddings(frames, MODEL_NAME, DEVICE)
#     print("frame_embeddings.shape =", frame_embeddings.shape)
#
#     print("开始求平均得到视频向量...")
#     video_embedding = get_video_embedding(frame_embeddings)
#     print("video_embedding.shape =", video_embedding.shape)
#
#     np.save(EMBED_PATH, video_embedding)
#     print(f"视频最终向量已保存到: {EMBED_PATH}")
#
#     print("前10维向量值:")
#     print(video_embedding[:10])
#
#
# if __name__ == "__main__":
#     main()








import os
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTModel
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# =========================
# 配置区
# =========================
ROOT_DIR = r"E:\dataset\微博热搜数据集"
MODEL_NAME = "google/vit-base-patch16-224-in21k"
NUM_FRAMES = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 支持的视频后缀
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm", ".mpeg", ".mpg"}


def sample_frame_indices(total_frames: int, num_frames: int = 4):
    if total_frames <= 0:
        raise ValueError("视频总帧数无效。")
    indices = np.linspace(0, total_frames - 1, num_frames)
    return [int(round(x)) for x in indices]


def extract_frames(video_path: str, num_frames: int = 4):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"视频总帧数无效: {video_path}")

    frame_indices = sample_frame_indices(total_frames, num_frames)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            cap.release()
            raise RuntimeError(f"读取第 {idx} 帧失败: {video_path}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        frames.append(pil_img)

    cap.release()
    return frames


def load_vit_model(model_name: str, device: str):
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


def get_frame_embeddings(frames, processor, model, device: str):
    inputs = processor(images=frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # 取 CLS token，shape: [num_frames, 768]
        frame_embeddings = outputs.last_hidden_state[:, 0, :]

    return frame_embeddings.cpu().numpy()


def get_video_embedding(frame_embeddings: np.ndarray):
    # 对多帧 embedding 求平均，得到视频级向量
    return np.mean(frame_embeddings, axis=0)


def process_single_video(video_path: str, output_npy_path: str, processor, model, device: str):
    print(f"\n开始处理视频: {video_path}")

    frames = extract_frames(video_path, num_frames=NUM_FRAMES)
    frame_embeddings = get_frame_embeddings(frames, processor, model, device)
    video_embedding = get_video_embedding(frame_embeddings)

    np.save(output_npy_path, video_embedding)
    print(f"已保存 embedding 到: {output_npy_path}")
    print(f"video_embedding.shape = {video_embedding.shape}")


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

        embedding_dir = os.path.join(topic_dir, "videoembedding")
        os.makedirs(embedding_dir, exist_ok=True)
        print(f"videoembedding 文件夹已就绪: {embedding_dir}")

        video_files = [
            f for f in os.listdir(videos_dir)
            if os.path.isfile(os.path.join(videos_dir, f))
            and os.path.splitext(f)[1].lower() in VIDEO_EXTS
        ]

        print(f"发现 {len(video_files)} 个视频文件")
        total_videos += len(video_files)

        for video_file in video_files:
            video_path = os.path.join(videos_dir, video_file)
            video_id = os.path.splitext(video_file)[0]   # 文件名即 id
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


if __name__ == "__main__":
    process_dataset(ROOT_DIR)