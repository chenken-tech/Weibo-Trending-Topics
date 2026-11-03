# # import os
# # import re
# # import json
# # import subprocess
# # import tempfile
# # import numpy as np
# # import pandas as pd
# # import cv2
# # import torch
# # import torch.nn.functional as F
# # import whisper
# #
# # from paddleocr import PaddleOCR
# # from transformers import AutoTokenizer, AutoModel
# #
# # # =========================
# # # 配置区
# # # =========================
# # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# #
# # ROOT_DIR = r"E:\dataset\结果文件1"
# #
# # VIDEO_DIR_NAME = "videos"
# # OUTPUT_EMBED_DIR_NAME = "contextembedding"
# # CHECK_DIR_NAME = "context_text_for_check"
# #
# # WEIBO_TEXT_COL = "微博正文"
# # ID_COL = None
# #
# # BGE_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
# # WHISPER_MODEL_NAME = "base"
# #
# # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # MAX_LENGTH = 512
# #
# # NUM_OCR_FRAMES = 8
# # OCR_SCORE_THRESHOLD = 0.5
# #
# # VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"}
# #
# #
# # # =========================
# # # 通用函数
# # # =========================
# # def read_csv_safely(csv_path):
# #     for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
# #         try:
# #             return pd.read_csv(csv_path, encoding=enc)
# #         except Exception:
# #             pass
# #     raise RuntimeError(f"CSV读取失败: {csv_path}")
# #
# #
# # def clean_text(text):
# #     if text is None or pd.isna(text):
# #         return ""
# #     text = str(text)
# #     text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
# #     text = re.sub(r"http\S+", "", text)
# #     text = re.sub(r"\s+", " ", text)
# #     return text.strip()
# #
# #
# # def safe_filename(name):
# #     name = str(name).strip()
# #     if name == "" or name.lower() == "nan":
# #         name = "empty_id"
# #     return re.sub(r'[\\/:*?"<>|]', "_", name)
# #
# #
# # def find_csv_file(topic_dir):
# #     csv_files = [f for f in os.listdir(topic_dir) if f.lower().endswith(".csv")]
# #     if not csv_files:
# #         return None
# #     return os.path.join(topic_dir, csv_files[0])
# #
# #
# # def infer_id_column(df):
# #     candidates = [
# #         "id", "ID",
# #         "视频id", "视频ID",
# #         "作品id", "作品ID",
# #         "微博id", "微博ID",
# #         "mid", "mblogid",
# #         "item_id", "video_id", "aweme_id"
# #     ]
# #     for col in candidates:
# #         if col in df.columns:
# #             return col
# #     return df.columns[0]
# #
# #
# # def find_video_file(video_dir, video_id):
# #     sid = safe_filename(video_id)
# #
# #     for ext in VIDEO_EXTS:
# #         path = os.path.join(video_dir, sid + ext)
# #         if os.path.exists(path):
# #             return path
# #
# #     for file in os.listdir(video_dir):
# #         name, ext = os.path.splitext(file)
# #         if ext.lower() in VIDEO_EXTS and name == sid:
# #             return os.path.join(video_dir, file)
# #
# #     return None
# #
# #
# # # =========================
# # # OCR：视频画面文字识别
# # # =========================
# # def sample_frame_indices(total_frames, num_frames):
# #     if total_frames <= 0:
# #         return []
# #     return [int(x) for x in np.linspace(0, total_frames - 1, num_frames)]
# #
# #
# # def extract_ocr_text(video_path, ocr_model):
# #     cap = cv2.VideoCapture(video_path)
# #
# #     if not cap.isOpened():
# #         raise RuntimeError(f"无法打开视频: {video_path}")
# #
# #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# #     indices = sample_frame_indices(total_frames, NUM_OCR_FRAMES)
# #
# #     all_texts = []
# #
# #     for idx in indices:
# #         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
# #         ok, frame = cap.read()
# #
# #         if not ok:
# #             continue
# #
# #         result = ocr_model.ocr(frame, cls=True)
# #
# #         if not result:
# #             continue
# #
# #         for group in result:
# #             if not group:
# #                 continue
# #             for line in group:
# #                 try:
# #                     text = line[1][0]
# #                     score = line[1][1]
# #                     if score >= OCR_SCORE_THRESHOLD and text.strip():
# #                         all_texts.append(text.strip())
# #                 except Exception:
# #                     continue
# #
# #     cap.release()
# #
# #     seen = set()
# #     dedup = []
# #
# #     for t in all_texts:
# #         t = clean_text(t)
# #         if t and t not in seen:
# #             seen.add(t)
# #             dedup.append(t)
# #
# #     return " ".join(dedup)
# #
# #
# # # =========================
# # # ASR：视频语音转文字
# # # =========================
# # def extract_audio(video_path, wav_path):
# #     cmd = [
# #         "ffmpeg",
# #         "-y",
# #         "-i", video_path,
# #         "-vn",
# #         "-ac", "1",
# #         "-ar", "16000",
# #         "-f", "wav",
# #         wav_path
# #     ]
# #
# #     result = subprocess.run(
# #         cmd,
# #         stdout=subprocess.PIPE,
# #         stderr=subprocess.PIPE
# #     )
# #
# #     if result.returncode != 0:
# #         raise RuntimeError(result.stderr.decode("utf-8", errors="ignore"))
# #
# #
# # def extract_asr_text(video_path, whisper_model):
# #     with tempfile.TemporaryDirectory() as tmpdir:
# #         wav_path = os.path.join(tmpdir, "audio.wav")
# #         extract_audio(video_path, wav_path)
# #
# #         result = whisper_model.transcribe(
# #             wav_path,
# #             language="zh",
# #             fp16=(DEVICE == "cuda")
# #         )
# #
# #         return clean_text(result.get("text", ""))
# #
# #
# # # =========================
# # # BGE embedding
# # # =========================
# # def load_bge_model():
# #     tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_NAME)
# #
# #     model = AutoModel.from_pretrained(
# #         BGE_MODEL_NAME,
# #         use_safetensors=True
# #     ).to(DEVICE)
# #
# #     model.eval()
# #     return tokenizer, model
# #
# #
# # def encode_bge(text, tokenizer, model):
# #     if not text.strip():
# #         text = "空文本"
# #
# #     inputs = tokenizer(
# #         text,
# #         max_length=MAX_LENGTH,
# #         padding=True,
# #         truncation=True,
# #         return_tensors="pt"
# #     )
# #
# #     inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
# #
# #     with torch.no_grad():
# #         outputs = model(**inputs)
# #         emb = outputs.last_hidden_state[:, 0]
# #         emb = F.normalize(emb, p=2, dim=1)
# #
# #     return emb.squeeze(0).cpu().numpy()
# #
# #
# # def build_context_text(weibo_text, ocr_text, asr_text):
# #     parts = []
# #
# #     if weibo_text.strip():
# #         parts.append(weibo_text.strip())
# #
# #     if ocr_text.strip():
# #         parts.append(ocr_text.strip())
# #
# #     if asr_text.strip():
# #         parts.append(asr_text.strip())
# #
# #     if not parts:
# #         return "空文本"
# #
# #     return " [SEP] ".join(parts)
# #
# #
# # # =========================
# # # 主处理逻辑
# # # =========================
# # def process_topic(topic_dir, tokenizer, bge_model, ocr_model, whisper_model):
# #     csv_path = find_csv_file(topic_dir)
# #
# #     if csv_path is None:
# #         print(f"未找到CSV，跳过: {topic_dir}")
# #         return 0, 0
# #
# #     video_dir = os.path.join(topic_dir, VIDEO_DIR_NAME)
# #
# #     if not os.path.exists(video_dir):
# #         print(f"未找到videos文件夹，跳过: {topic_dir}")
# #         return 0, 0
# #
# #     df = read_csv_safely(csv_path)
# #
# #     id_col = ID_COL if ID_COL is not None else infer_id_column(df)
# #
# #     embed_dir = os.path.join(topic_dir, OUTPUT_EMBED_DIR_NAME)
# #     check_dir = os.path.join(topic_dir, CHECK_DIR_NAME)
# #
# #     os.makedirs(embed_dir, exist_ok=True)
# #     os.makedirs(check_dir, exist_ok=True)
# #
# #     print(f"\n处理话题文件夹: {topic_dir}")
# #     print(f"CSV: {csv_path}")
# #     print(f"ID列: {id_col}")
# #     print(f"输出embedding: {embed_dir}")
# #     print(f"输出check: {check_dir}")
# #
# #     success = 0
# #     fail = 0
# #
# #     for idx, row in df.iterrows():
# #         try:
# #             video_id = str(row[id_col]).strip()
# #
# #             if video_id == "" or video_id.lower() == "nan":
# #                 video_id = f"row_{idx}"
# #
# #             sid = safe_filename(video_id)
# #
# #             video_path = find_video_file(video_dir, video_id)
# #
# #             weibo_text = clean_text(row.get(WEIBO_TEXT_COL, ""))
# #
# #             ocr_text = ""
# #             asr_text = ""
# #
# #             if video_path is not None:
# #                 try:
# #                     ocr_text = extract_ocr_text(video_path, ocr_model)
# #                 except Exception as e:
# #                     print(f"OCR失败: {video_id}, {e}")
# #
# #                 try:
# #                     asr_text = extract_asr_text(video_path, whisper_model)
# #                 except Exception as e:
# #                     print(f"ASR失败: {video_id}, {e}")
# #             else:
# #                 print(f"未找到视频文件: {video_id}")
# #
# #             context_text = build_context_text(
# #                 weibo_text=weibo_text,
# #                 ocr_text=ocr_text,
# #                 asr_text=asr_text
# #             )
# #
# #             embedding = encode_bge(context_text, tokenizer, bge_model)
# #
# #             npy_path = os.path.join(embed_dir, sid + ".npy")
# #             np.save(npy_path, embedding)
# #
# #             check_info = {
# #                 "video_id": video_id,
# #                 "safe_video_id": sid,
# #                 "video_path": video_path,
# #
# #                 "weibo_text": weibo_text,
# #                 "ocr_text": ocr_text,
# #                 "asr_text": asr_text,
# #
# #                 "used_weibo": bool(weibo_text.strip()),
# #                 "used_ocr": bool(ocr_text.strip()),
# #                 "used_asr": bool(asr_text.strip()),
# #
# #                 "weibo_length": len(weibo_text),
# #                 "ocr_length": len(ocr_text),
# #                 "asr_length": len(asr_text),
# #                 "context_length": len(context_text),
# #
# #                 "context_text": context_text,
# #                 "embedding_path": npy_path,
# #                 "embedding_shape": list(embedding.shape)
# #             }
# #
# #             check_path = os.path.join(check_dir, sid + ".json")
# #
# #             with open(check_path, "w", encoding="utf-8") as f:
# #                 json.dump(check_info, f, ensure_ascii=False, indent=2)
# #
# #             success += 1
# #
# #             print(f"[成功] {video_id}")
# #             print(f"  used_weibo={check_info['used_weibo']}, used_ocr={check_info['used_ocr']}, used_asr={check_info['used_asr']}")
# #             print(f"  context_length={check_info['context_length']}, embedding_shape={embedding.shape}")
# #
# #         except Exception as e:
# #             fail += 1
# #             print(f"[失败] 第{idx}行: {e}")
# #
# #     print(f"话题完成: 成功 {success}, 失败 {fail}")
# #     return success, fail
# #
# #
# # def process_dataset():
# #     print(f"Using device: {DEVICE}")
# #
# #     print(f"加载BGE模型: {BGE_MODEL_NAME}")
# #     tokenizer, bge_model = load_bge_model()
# #
# #     print("加载PaddleOCR模型")
# #     ocr_model = PaddleOCR(use_angle_cls=True, lang="ch")
# #
# #     print(f"加载Whisper模型: {WHISPER_MODEL_NAME}")
# #     whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=DEVICE)
# #
# #     topic_dirs = [
# #         os.path.join(ROOT_DIR, name)
# #         for name in os.listdir(ROOT_DIR)
# #         if os.path.isdir(os.path.join(ROOT_DIR, name))
# #     ]
# #
# #     print(f"共发现 {len(topic_dirs)} 个话题文件夹")
# #
# #     total_success = 0
# #     total_fail = 0
# #
# #     for topic_dir in topic_dirs:
# #         s, f = process_topic(
# #             topic_dir=topic_dir,
# #             tokenizer=tokenizer,
# #             bge_model=bge_model,
# #             ocr_model=ocr_model,
# #             whisper_model=whisper_model
# #         )
# #
# #         total_success += s
# #         total_fail += f
# #
# #     print("\n全部处理完成")
# #     print(f"总成功: {total_success}")
# #     print(f"总失败: {total_fail}")
# #
# #
# # if __name__ == "__main__":
# #     process_dataset()
#
#
#
#
#
#
#
# import os
# import re
# import json
# import subprocess
# import tempfile
# import multiprocessing as mp
# from concurrent.futures import ProcessPoolExecutor, as_completed
# # from concurrent.futures import ThreadPoolExecutor, as_completed   # 不是 ProcessPoolExecutor
#
# import numpy as np
# import pandas as pd
# import cv2
# import torch
# import torch.nn.functional as F
# import whisper
#
# from paddleocr import PaddleOCR
# from transformers import AutoTokenizer, AutoModel
# from tqdm import tqdm
#
# # =========================
# # 配置区
# # =========================
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#
# ROOT_DIR = r"E:/dataset/微博热搜数据集"
#
# VIDEO_DIR_NAME = "videos"
# OUTPUT_EMBED_DIR_NAME = "contextembedding"
# CHECK_DIR_NAME = "context_text_for_check"
#
# WEIBO_TEXT_COL = "微博正文"
# ID_COL = None
#
# BGE_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
#
# # RTX 4060 推荐 small；如果显存不够，改 tiny
# WHISPER_MODEL_NAME = "tiny"
#
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#
# MAX_LENGTH = 512
#
#
# # RTX 4060 推荐 2；爆显存就改 1
# NUM_WORKERS = 1
# # OCR 抽帧数，推荐 4；追求更快可改 2
# NUM_OCR_FRAMES = 4
# OCR_SCORE_THRESHOLD = 0.5
#
# VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"}
#
#
# # =========================
# # 通用函数
# # =========================
# def read_csv_safely(csv_path):
#     for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
#         try:
#             return pd.read_csv(csv_path, encoding=enc)
#         except Exception:
#             pass
#     raise RuntimeError(f"CSV读取失败: {csv_path}")
#
#
# def clean_text(text):
#     if text is None or pd.isna(text):
#         return ""
#     text = str(text)
#     text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
#     text = re.sub(r"http\S+", "", text)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()
#
#
# def safe_filename(name):
#     name = str(name).strip()
#     if name == "" or name.lower() == "nan":
#         name = "empty_id"
#     if name.endswith(".0"):
#         name = name[:-2]
#     return re.sub(r'[\\/:*?"<>|]', "_", name)
#
#
# def find_csv_file(topic_dir):
#     csv_files = [
#         f for f in os.listdir(topic_dir)
#         if f.lower().endswith(".csv")
#         and not f.lower().endswith("_scored.csv")
#         and not f.lower().endswith("_selected.csv")
#         and not f.lower().endswith("_userinfo.csv")
#     ]
#
#     if not csv_files:
#         return None
#
#     return os.path.join(topic_dir, csv_files[0])
#
#
# def infer_id_column(df):
#     candidates = [
#         "id", "ID",
#         "视频id", "视频ID",
#         "作品id", "作品ID",
#         "微博id", "微博ID",
#         "mid", "mblogid",
#         "item_id", "video_id", "aweme_id"
#     ]
#
#     for col in candidates:
#         if col in df.columns:
#             return col
#
#     return df.columns[0]
#
#
# def find_video_file(video_dir, video_id):
#     sid = safe_filename(video_id)
#
#     for ext in VIDEO_EXTS:
#         path = os.path.join(video_dir, sid + ext)
#         if os.path.exists(path):
#             return path
#
#     for file in os.listdir(video_dir):
#         name, ext = os.path.splitext(file)
#         if ext.lower() in VIDEO_EXTS and name == sid:
#             return os.path.join(video_dir, file)
#
#     return None
#
#
# # =========================
# # OCR：视频画面文字识别
# # =========================
# def sample_frame_indices(total_frames, num_frames):
#     if total_frames <= 0:
#         return []
#     return [int(x) for x in np.linspace(0, total_frames - 1, num_frames)]
#
#
# def extract_ocr_text(video_path, ocr_model):
#     cap = cv2.VideoCapture(video_path)
#
#     if not cap.isOpened():
#         raise RuntimeError(f"无法打开视频: {video_path}")
#
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     indices = sample_frame_indices(total_frames, NUM_OCR_FRAMES)
#
#     all_texts = []
#
#     for idx in indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         ok, frame = cap.read()
#
#         if not ok:
#             continue
#
#         result = ocr_model.ocr(frame, cls=True)
#
#         if not result:
#             continue
#
#         for group in result:
#             if not group:
#                 continue
#
#             for line in group:
#                 try:
#                     text = line[1][0]
#                     score = line[1][1]
#
#                     if score >= OCR_SCORE_THRESHOLD and text.strip():
#                         all_texts.append(text.strip())
#
#                 except Exception:
#                     continue
#
#     cap.release()
#
#     seen = set()
#     dedup = []
#
#     for t in all_texts:
#         t = clean_text(t)
#         if t and t not in seen:
#             seen.add(t)
#             dedup.append(t)
#
#     return " ".join(dedup)
#
#
# # =========================
# # ASR：视频语音转文字
# # =========================
# def extract_audio(video_path, wav_path):
#     cmd = [
#         "ffmpeg",
#         "-y",
#         "-i", video_path,
#         "-vn",
#         "-ac", "1",
#         "-ar", "16000",
#         "-f", "wav",
#         wav_path
#     ]
#
#     result = subprocess.run(
#         cmd,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE
#     )
#
#     if result.returncode != 0:
#         raise RuntimeError(result.stderr.decode("utf-8", errors="ignore"))
#
#
# def extract_asr_text(video_path, whisper_model):
#     with tempfile.TemporaryDirectory() as tmpdir:
#         wav_path = os.path.join(tmpdir, "audio.wav")
#         extract_audio(video_path, wav_path)
#
#         result = whisper_model.transcribe(
#             wav_path,
#             language="zh",
#             fp16=(DEVICE == "cuda")
#         )
#
#         return clean_text(result.get("text", ""))
#
#
# # =========================
# # BGE embedding
# # =========================
# def load_bge_model():
#     tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_NAME)
#
#     model = AutoModel.from_pretrained(
#         BGE_MODEL_NAME,
#         use_safetensors=True
#     ).to(DEVICE)
#
#     model.eval()
#     return tokenizer, model
#
#
# def encode_bge(text, tokenizer, model):
#     if not text.strip():
#         text = "空文本"
#
#     inputs = tokenizer(
#         text,
#         max_length=MAX_LENGTH,
#         padding=True,
#         truncation=True,
#         return_tensors="pt"
#     )
#
#     inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
#
#     with torch.no_grad():
#         outputs = model(**inputs)
#         emb = outputs.last_hidden_state[:, 0]
#         emb = F.normalize(emb, p=2, dim=1)
#
#     return emb.squeeze(0).cpu().numpy()
#
#
# def build_context_text(weibo_text, ocr_text, asr_text):
#     parts = []
#
#     if weibo_text.strip():
#         parts.append(weibo_text.strip())
#
#     if ocr_text.strip():
#         parts.append(ocr_text.strip())
#
#     if asr_text.strip():
#         parts.append(asr_text.strip())
#
#     if not parts:
#         return "空文本"
#
#     return " [SEP] ".join(parts)
#
#
# # =========================
# # 单个话题处理
# # =========================
# def process_topic(topic_dir, tokenizer, bge_model, ocr_model, whisper_model):
#     csv_path = find_csv_file(topic_dir)
#
#     if csv_path is None:
#         print(f"未找到CSV，跳过: {topic_dir}")
#         return 0, 0
#
#     video_dir = os.path.join(topic_dir, VIDEO_DIR_NAME)
#
#     if not os.path.exists(video_dir):
#         print(f"未找到videos文件夹，跳过: {topic_dir}")
#         return 0, 0
#
#     df = read_csv_safely(csv_path)
#     id_col = ID_COL if ID_COL is not None else infer_id_column(df)
#
#     embed_dir = os.path.join(topic_dir, OUTPUT_EMBED_DIR_NAME)
#     check_dir = os.path.join(topic_dir, CHECK_DIR_NAME)
#
#     os.makedirs(embed_dir, exist_ok=True)
#     os.makedirs(check_dir, exist_ok=True)
#
#     print(f"\n处理话题文件夹: {topic_dir}")
#     print(f"CSV: {csv_path}")
#     print(f"ID列: {id_col}")
#     print(f"输出embedding: {embed_dir}")
#     print(f"输出check: {check_dir}")
#
#     success = 0
#     fail = 0
#
#     # for idx, row in df.iterrows():
#     for idx, row in tqdm(
#             df.iterrows(),
#             total=len(df),
#             desc=f"处理 {os.path.basename(topic_dir)}",
#             ncols=100
#     ):
#         try:
#             video_id = str(row[id_col]).strip()
#
#             if video_id == "" or video_id.lower() == "nan":
#                 video_id = f"row_{idx}"
#
#             # sid = safe_filename(video_id)
#             # video_path = find_video_file(video_dir, video_id)
#             #
#             # weibo_text = clean_text(row.get(WEIBO_TEXT_COL, ""))
#             # ocr_text = ""
#             # asr_text = ""
#             sid = safe_filename(video_id)
#
#             npy_path = os.path.join(embed_dir, sid + ".npy")
#             check_path = os.path.join(check_dir, sid + ".json")
#
#             # =========================
#             # 缓存机制：已生成则跳过
#             # =========================
#             if os.path.exists(npy_path) and os.path.exists(check_path):
#                 try:
#                     arr = np.load(npy_path)
#                     if arr.shape[0] == 768:
#                         success += 1
#                         print(f"[跳过-已存在] {video_id} -> {npy_path}")
#                         continue
#                     else:
#                         print(f"[重新处理] {video_id}，embedding维度异常: {arr.shape}")
#                 except Exception:
#                     print(f"[重新处理] {video_id}，已有npy无法读取")
#
#             video_path = find_video_file(video_dir, video_id)
#
#             weibo_text = clean_text(row.get(WEIBO_TEXT_COL, ""))
#             ocr_text = ""
#             asr_text = ""
#
#             if video_path is not None:
#                 try:
#                     ocr_text = extract_ocr_text(video_path, ocr_model)
#                 except Exception as e:
#                     print(f"OCR失败: {video_id}, {e}")
#
#                 try:
#                     asr_text = extract_asr_text(video_path, whisper_model)
#                 except Exception as e:
#                     print(f"ASR失败: {video_id}, {e}")
#             else:
#                 print(f"未找到视频文件: {video_id}")
#
#             context_text = build_context_text(
#                 weibo_text=weibo_text,
#                 ocr_text=ocr_text,
#                 asr_text=asr_text
#             )
#
#             embedding = encode_bge(context_text, tokenizer, bge_model)
#
#             # npy_path = os.path.join(embed_dir, sid + ".npy")
#             np.save(npy_path, embedding)
#
#             check_info = {
#                 "video_id": video_id,
#                 "safe_video_id": sid,
#                 "video_path": video_path,
#
#                 "weibo_text": weibo_text,
#                 "ocr_text": ocr_text,
#                 "asr_text": asr_text,
#
#                 "used_weibo": bool(weibo_text.strip()),
#                 "used_ocr": bool(ocr_text.strip()),
#                 "used_asr": bool(asr_text.strip()),
#
#                 "weibo_length": len(weibo_text),
#                 "ocr_length": len(ocr_text),
#                 "asr_length": len(asr_text),
#                 "context_length": len(context_text),
#
#                 "context_text": context_text,
#                 "embedding_path": npy_path,
#                 "embedding_shape": list(embedding.shape)
#             }
#
#             # check_path = os.path.join(check_dir, sid + ".json")
#             #
#             # with open(check_path, "w", encoding="utf-8") as f:
#             #     json.dump(check_info, f, ensure_ascii=False, indent=2)
#             with open(check_path, "w", encoding="utf-8") as f:
#                 json.dump(check_info, f, ensure_ascii=False, indent=2)
#
#             success += 1
#
#             print(f"[成功] {video_id}")
#             print(
#                 f"  used_weibo={check_info['used_weibo']}, "
#                 f"used_ocr={check_info['used_ocr']}, "
#                 f"used_asr={check_info['used_asr']}"
#             )
#             print(
#                 f"  context_length={check_info['context_length']}, "
#                 f"embedding_shape={embedding.shape}"
#             )
#
#         except Exception as e:
#             fail += 1
#             print(f"[失败] 第{idx}行: {e}")
#
#     print(f"话题完成: 成功 {success}, 失败 {fail}")
#     return success, fail
#
#
# # =========================
# # 多进程 Worker
# # =========================
# def process_topic_worker(topic_dir):
#     """
#     每个进程单独加载模型。
#     RTX 4060 推荐 NUM_WORKERS=2。
#     如果显存不足，改 NUM_WORKERS=1。
#     """
#     print(f"\n[Worker] 开始处理: {topic_dir}")
#     print(f"[Worker] DEVICE = {DEVICE}")
#
#     tokenizer, bge_model = load_bge_model()
#
#     # OCR 默认走 CPU，避免占用 GPU 显存
#     ocr_model = PaddleOCR(
#         use_angle_cls=True,
#         lang="ch",
#         show_log=False,
#         use_gpu=False
#     )
#
#     whisper_model = whisper.load_model(
#         WHISPER_MODEL_NAME,
#         device=DEVICE
#     )
#
#     s, f = process_topic(
#         topic_dir=topic_dir,
#         tokenizer=tokenizer,
#         bge_model=bge_model,
#         ocr_model=ocr_model,
#         whisper_model=whisper_model
#     )
#
#     return topic_dir, s, f
#
#
# def process_dataset_parallel():
#     topic_dirs = [
#         os.path.join(ROOT_DIR, name)
#         for name in os.listdir(ROOT_DIR)
#         if os.path.isdir(os.path.join(ROOT_DIR, name))
#     ]
#
#     print(f"Using device: {DEVICE}")
#     print(f"共发现 {len(topic_dirs)} 个话题文件夹")
#     print(f"并行进程数: {NUM_WORKERS}")
#     print(f"Whisper模型: {WHISPER_MODEL_NAME}")
#     print(f"OCR抽帧数: {NUM_OCR_FRAMES}")
#
#     total_success = 0
#     total_fail = 0
#     # with ThreadPoolExecutor(max_workers=4) as executor:  # 不是 ProcessPoolExecutor
#     with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
#         futures = {
#             executor.submit(process_topic_worker, topic_dir): topic_dir
#             for topic_dir in topic_dirs
#         }
#
#         # for future in as_completed(futures):
#         for future in tqdm(
#                 as_completed(futures),
#                 total=len(futures),
#                 desc="整体话题进度",
#                 ncols=100
#         ):
#             topic_dir = futures[future]
#
#             try:
#                 _, s, f = future.result()
#                 total_success += s
#                 total_fail += f
#
#                 print(f"\n[完成] {topic_dir}")
#                 print(f"成功: {s}, 失败: {f}")
#
#             except Exception as e:
#                 print(f"\n[失败] {topic_dir}")
#                 print(f"错误: {e}")
#
#     print("\n全部处理完成")
#     print(f"总成功: {total_success}")
#     print(f"总失败: {total_fail}")
#
#
# if __name__ == "__main__":
#     mp.freeze_support()
#     process_dataset_parallel()


import os
import re
import json
import subprocess
import tempfile
import gc

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
import whisper

from paddleocr import PaddleOCR
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# =========================
# 配置区
# =========================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

ROOT_DIR = r"E:/dataset/微博热搜数据集"

VIDEO_DIR_NAME = "videos"
OUTPUT_EMBED_DIR_NAME = "contextembedding"
CHECK_DIR_NAME = "context_text_for_check"

WEIBO_TEXT_COL = "微博正文"
ID_COL = None

BGE_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
WHISPER_MODEL_NAME = "tiny"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512

NUM_OCR_FRAMES = 4
OCR_SCORE_THRESHOLD = 0.5

# 如果后续某些视频 ASR 仍然容易卡住/崩溃，可以先把 ENABLE_ASR 改成 False
ENABLE_OCR = False
ENABLE_ASR = False

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"}

# 总失败日志，记录每个话题/每个视频失败原因
FAILED_LOG_NAME = "_single_process_failed_log.json"


# =========================
# 通用函数
# =========================
def read_csv_safely(csv_path):
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"CSV读取失败: {csv_path}")


def clean_text(text):
    if text is None or pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_filename(name):
    name = str(name).strip()
    if name == "" or name.lower() == "nan":
        name = "empty_id"
    if name.endswith(".0"):
        name = name[:-2]
    return re.sub(r'[\\/:*?"<>|]', "_", name)


def find_csv_file(topic_dir):
    csv_files = [
        f for f in os.listdir(topic_dir)
        if f.lower().endswith(".csv")
        and not f.lower().endswith("_scored.csv")
        and not f.lower().endswith("_selected.csv")
        and not f.lower().endswith("_userinfo.csv")
    ]

    if not csv_files:
        return None

    return os.path.join(topic_dir, csv_files[0])


def infer_id_column(df):
    candidates = [
        "id", "ID",
        "视频id", "视频ID",
        "作品id", "作品ID",
        "微博id", "微博ID",
        "mid", "mblogid",
        "item_id", "video_id", "aweme_id"
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return df.columns[0]


def find_video_file(video_dir, video_id):
    sid = safe_filename(video_id)

    for ext in VIDEO_EXTS:
        path = os.path.join(video_dir, sid + ext)
        if os.path.exists(path):
            return path

    try:
        files = os.listdir(video_dir)
    except Exception:
        return None

    for file in files:
        name, ext = os.path.splitext(file)
        if ext.lower() in VIDEO_EXTS and name == sid:
            return os.path.join(video_dir, file)

    return None


def is_valid_embedding(npy_path, expected_dim=768):
    """
    判断某个视频的 embedding 是否已经正确生成。
    同时要求 npy 能读取，并且 shape 为 (768,) 或最后一维为 768。
    """
    if not os.path.exists(npy_path):
        return False

    try:
        arr = np.load(npy_path)
        if arr.shape == (expected_dim,):
            return True
        if len(arr.shape) >= 1 and arr.shape[-1] == expected_dim:
            return True
        return False
    except Exception:
        return False


def is_video_done(npy_path, check_path):
    """
    单视频断点续跑判断：
    只有 embedding 文件和 check json 都存在，并且 embedding 维度正确，才认为已完成。
    """
    return os.path.exists(check_path) and is_valid_embedding(npy_path)


def append_failed_log(record):
    """
    追加失败日志。采用整体 JSON 列表形式，方便后续查看。
    """
    log_path = os.path.join(ROOT_DIR, FAILED_LOG_NAME)

    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    else:
        data = []

    data.append(record)

    tmp_path = log_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    os.replace(tmp_path, log_path)


# =========================
# OCR：视频画面文字识别
# =========================
def sample_frame_indices(total_frames, num_frames):
    if total_frames <= 0:
        return []
    return [int(x) for x in np.linspace(0, total_frames - 1, num_frames)]


def extract_ocr_text(video_path, ocr_model):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sample_frame_indices(total_frames, NUM_OCR_FRAMES)

        all_texts = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()

            if not ok:
                continue

            result = ocr_model.ocr(frame, cls=True)

            if not result:
                continue

            for group in result:
                if not group:
                    continue

                for line in group:
                    try:
                        text = line[1][0]
                        score = line[1][1]

                        if score >= OCR_SCORE_THRESHOLD and text.strip():
                            all_texts.append(text.strip())

                    except Exception:
                        continue

        seen = set()
        dedup = []

        for t in all_texts:
            t = clean_text(t)
            if t and t not in seen:
                seen.add(t)
                dedup.append(t)

        return " ".join(dedup)

    finally:
        cap.release()


# =========================
# ASR：视频语音转文字
# =========================
def extract_audio(video_path, wav_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        wav_path
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="ignore"))


def extract_asr_text(video_path, whisper_model):
    if whisper_model is None:
        return ""

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "audio.wav")
        extract_audio(video_path, wav_path)

        result = whisper_model.transcribe(
            wav_path,
            language="zh",
            fp16=(DEVICE == "cuda")
        )

        return clean_text(result.get("text", ""))


# =========================
# BGE embedding
# =========================
def load_bge_model():
    tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_NAME)

    model = AutoModel.from_pretrained(
        BGE_MODEL_NAME,
        use_safetensors=True
    ).to(DEVICE)

    model.eval()
    return tokenizer, model


def encode_bge(text, tokenizer, model):
    if not text.strip():
        text = "空文本"

    inputs = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0]
        emb = F.normalize(emb, p=2, dim=1)

    return emb.squeeze(0).cpu().numpy()


def build_context_text(weibo_text, ocr_text, asr_text):
    parts = []

    if weibo_text.strip():
        parts.append(weibo_text.strip())

    if ocr_text.strip():
        parts.append(ocr_text.strip())

    if asr_text.strip():
        parts.append(asr_text.strip())

    if not parts:
        return "空文本"

    return " [SEP] ".join(parts)


# =========================
# 单个话题处理
# =========================
def process_topic(topic_dir, tokenizer, bge_model, ocr_model, whisper_model):
    csv_path = find_csv_file(topic_dir)

    if csv_path is None:
        print(f"未找到CSV，跳过: {topic_dir}")
        return 0, 0, 0

    video_dir = os.path.join(topic_dir, VIDEO_DIR_NAME)

    if not os.path.exists(video_dir):
        print(f"未找到videos文件夹，跳过: {topic_dir}")
        return 0, 0, 0

    df = read_csv_safely(csv_path)
    id_col = ID_COL if ID_COL is not None else infer_id_column(df)

    embed_dir = os.path.join(topic_dir, OUTPUT_EMBED_DIR_NAME)
    check_dir = os.path.join(topic_dir, CHECK_DIR_NAME)

    os.makedirs(embed_dir, exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)

    print(f"\n处理话题文件夹: {topic_dir}")
    print(f"CSV: {csv_path}")
    print(f"ID列: {id_col}")
    print(f"输出embedding: {embed_dir}")
    print(f"输出check: {check_dir}")

    success = 0
    skipped = 0
    fail = 0

    for idx, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"处理 {os.path.basename(topic_dir)}",
            ncols=100
    ):
        video_id = None

        try:
            video_id = str(row[id_col]).strip()

            if video_id == "" or video_id.lower() == "nan":
                video_id = f"row_{idx}"

            sid = safe_filename(video_id)

            npy_path = os.path.join(embed_dir, sid + ".npy")
            check_path = os.path.join(check_dir, sid + ".json")

            # =========================
            # 单视频缓存：已完成则直接跳过
            # =========================
            if is_video_done(npy_path, check_path):
                skipped += 1
                continue

            video_path = find_video_file(video_dir, video_id)

            weibo_text = clean_text(row.get(WEIBO_TEXT_COL, ""))
            ocr_text = ""
            asr_text = ""

            if video_path is not None:
                if ENABLE_OCR:
                    try:
                        ocr_text = extract_ocr_text(video_path, ocr_model)
                    except Exception as e:
                        print(f"OCR失败: {video_id}, {e}")

                if ENABLE_ASR:
                    try:
                        asr_text = extract_asr_text(video_path, whisper_model)
                    except Exception as e:
                        print(f"ASR失败: {video_id}, {e}")
            else:
                print(f"未找到视频文件: {video_id}")

            context_text = build_context_text(
                weibo_text=weibo_text,
                ocr_text=ocr_text,
                asr_text=asr_text
            )

            embedding = encode_bge(context_text, tokenizer, bge_model)

            # 先写临时文件，再原子替换，避免中途中断留下损坏 npy
            tmp_npy_path = npy_path + ".tmp.npy"
            np.save(tmp_npy_path, embedding)
            os.replace(tmp_npy_path, npy_path)

            check_info = {
                "video_id": video_id,
                "safe_video_id": sid,
                "video_path": video_path,

                "weibo_text": weibo_text,
                "ocr_text": ocr_text,
                "asr_text": asr_text,

                "used_weibo": bool(weibo_text.strip()),
                "used_ocr": bool(ocr_text.strip()),
                "used_asr": bool(asr_text.strip()),

                "weibo_length": len(weibo_text),
                "ocr_length": len(ocr_text),
                "asr_length": len(asr_text),
                "context_length": len(context_text),

                "context_text": context_text,
                "embedding_path": npy_path,
                "embedding_shape": list(embedding.shape)
            }

            tmp_check_path = check_path + ".tmp"
            with open(tmp_check_path, "w", encoding="utf-8") as f:
                json.dump(check_info, f, ensure_ascii=False, indent=2)
            os.replace(tmp_check_path, check_path)

            success += 1

        except Exception as e:
            fail += 1
            print(f"[失败] 话题: {topic_dir} | 第{idx}行 | video_id={video_id} | 错误: {e}")

            append_failed_log({
                "topic_dir": topic_dir,
                "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                "video_id": video_id,
                "error": str(e)
            })

        finally:
            # 单进程长时间跑，定期清理显存/内存
            if (success + skipped + fail) % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"话题完成: 新处理成功 {success}, 已存在跳过 {skipped}, 失败 {fail}")
    return success, skipped, fail


# =========================
# 单进程顺序处理整个数据集
# =========================
def process_dataset_single():
    topic_dirs = [
        os.path.join(ROOT_DIR, name)
        for name in os.listdir(ROOT_DIR)
        if os.path.isdir(os.path.join(ROOT_DIR, name))
    ]

    print(f"Using device: {DEVICE}")
    print(f"共发现 {len(topic_dirs)} 个话题文件夹")
    print("运行模式: 单进程顺序处理")
    print(f"Whisper模型: {WHISPER_MODEL_NAME}")
    print(f"OCR抽帧数: {NUM_OCR_FRAMES}")
    print(f"ENABLE_OCR: {ENABLE_OCR}")
    print(f"ENABLE_ASR: {ENABLE_ASR}")
    print("说明: 已存在合法 .npy + .json 的视频会自动跳过，不会重复处理。")

    print(f"\n加载BGE模型: {BGE_MODEL_NAME}")
    tokenizer, bge_model = load_bge_model()

    print("加载PaddleOCR模型")
    ocr_model = None
    if ENABLE_OCR:
        ocr_model = PaddleOCR(
            use_angle_cls=True,
            lang="ch",
            show_log=False,
            use_gpu=False
        )

    print(f"加载Whisper模型: {WHISPER_MODEL_NAME}")
    whisper_model = None
    if ENABLE_ASR:
        whisper_model = whisper.load_model(
            WHISPER_MODEL_NAME,
            device=DEVICE
        )

    total_success = 0
    total_skipped = 0
    total_fail = 0
    topic_fail = 0

    for topic_dir in tqdm(
            topic_dirs,
            total=len(topic_dirs),
            desc="整体话题进度",
            ncols=100
    ):
        try:
            s, skipped, f = process_topic(
                topic_dir=topic_dir,
                tokenizer=tokenizer,
                bge_model=bge_model,
                ocr_model=ocr_model,
                whisper_model=whisper_model
            )

            total_success += s
            total_skipped += skipped
            total_fail += f

            print(f"\n[完成] {topic_dir}")
            print(f"新处理成功: {s}, 已存在跳过: {skipped}, 失败: {f}")

        except Exception as e:
            topic_fail += 1
            total_fail += 1

            print(f"\n[话题级失败] {topic_dir}")
            print(f"错误: {e}")

            append_failed_log({
                "topic_dir": topic_dir,
                "row_index": None,
                "video_id": None,
                "error": str(e)
            })

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n全部处理完成")
    print(f"本次新处理成功视频数: {total_success}")
    print(f"已存在跳过视频数: {total_skipped}")
    print(f"失败视频/话题数: {total_fail}")
    print(f"话题级失败数: {topic_fail}")
    print(f"失败日志: {os.path.join(ROOT_DIR, FAILED_LOG_NAME)}")


if __name__ == "__main__":
    process_dataset_single()
