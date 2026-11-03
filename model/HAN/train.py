import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import clean_path, set_random_seed, infer_embedding_dim, infer_topic_embedding_dim
from data_builder import build_global_graph
from model import GlobalTopicRegressor_HAN


def split_indices(n, train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train, n_val = int(n * train_ratio), int(n * val_ratio)
    return torch.tensor(idx[:n_train]), torch.tensor(idx[n_train:n_train+n_val]), torch.tensor(idx[n_train+n_val:])

@torch.no_grad()
def evaluate(model, graph, labels, indices, device):
    model.eval()
    pred, _ = model(graph, device)
    labels = labels.to(device)
    indices = indices.to(device)
    y = labels[indices].squeeze(-1)
    p = pred[indices].squeeze(-1)
    # loss = F.mse_loss(p, y)
    loss = F.smooth_l1_loss(p, y)  # 替代 MSE
    ys, ps = y.cpu().numpy().astype(np.float32), p.cpu().numpy().astype(np.float32)
    mae = float(np.mean(np.abs(ys - ps))) if len(ys) else 0.0
    rmse = float(np.sqrt(np.mean((ys - ps) ** 2))) if len(ys) else 0.0
    nmse = float(np.mean((ys - ps) ** 2) / (np.std(ys) ** 2 + 1e-12)) if len(ys) > 1 and np.std(ys) > 1e-12 else 0.0
    mask = ys != 0
    mape = float(np.mean(np.abs((ys[mask] - ps[mask]) / (ys[mask] + 1e-12)))) if mask.sum() > 0 else 0.0
    r2 = float(1.0 - np.sum((ys - ps) ** 2) / np.sum((ys - ys.mean()) ** 2)) if len(ys) > 1 and np.var(ys) > 1e-12 else 0.0
    return {"loss": float(loss.item()), "MAE": mae, "RMSE": rmse, "nMSE": nmse, "MAPE": mape, "R2": r2, "y_true": ys.tolist(), "y_pred": ps.tolist()}

@torch.no_grad()
def save_outputs(model, graph, labels, meta, output_dir, device):
    model.eval()
    pred, h = model(graph, device)
    topic_emb = h["topic"].cpu().numpy().astype(np.float32)
    topic_pred = pred.cpu().numpy().astype(np.float32).reshape(-1)
    topic_label = labels.cpu().numpy().astype(np.float32).reshape(-1)
    np.save(output_dir / "topic_node_embeddings.npy", topic_emb)
    np.save(output_dir / "topic_pred_log10_read.npy", topic_pred)
    np.save(output_dir / "topic_label_log10_read.npy", topic_label)
    emb_dir = output_dir / "node_embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    for ntype in h:
        np.save(emb_dir / f"{ntype}_embeddings.npy", h[ntype].cpu().numpy().astype(np.float32))
    pd.DataFrame([{"topic_idx": i, "topic_name": name, "label_log10_read": float(topic_label[i]), "pred_log10_read": float(topic_pred[i]), "abs_error": float(abs(topic_label[i] - topic_pred[i]))} for i, name in enumerate(meta["topic_names"])]).to_csv(output_dir / "topic_predictions.csv", index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--video_dim", type=int, default=768)
    parser.add_argument("--audio_dim", type=int, default=512)
    parser.add_argument("--text_dim", type=int, default=768)
    parser.add_argument("--topic_sem_dim", type=int, default=768)
    parser.add_argument("--max_topics", type=int, default=-1)
    args = parser.parse_args()

    set_random_seed(args.seed)
    dataset_root, output_dir = clean_path(args.dataset_root), clean_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    print(f"Using device: {device}")
    print("dataset_root =", repr(str(dataset_root)))
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root 不存在：{dataset_root}")

    topic_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    video_dim = infer_embedding_dim(topic_dirs, ["videoembedding"], args.video_dim)
    image_dim = infer_embedding_dim(topic_dirs, ["imageembedding"], 768)
    audio_dim = infer_embedding_dim(topic_dirs, ["audioembedding"], args.audio_dim)
    text_dim = infer_embedding_dim(topic_dirs, ["textembedding", "text_embeddings", "bertembedding", "contextembedding"], args.text_dim)
    topic_sem_dim = infer_topic_embedding_dim(topic_dirs, args.topic_sem_dim)
    print(f"Inferred dims: video={video_dim}, audio={audio_dim}, text={text_dim}, topic_sem={topic_sem_dim}")

    graph, labels, meta = build_global_graph(dataset_root, video_dim, audio_dim, image_dim, text_dim, topic_sem_dim, args.max_topics)

    print("Global graph built.")
    print("Node counts:", {ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})
    print("Input dims:", meta["input_dims"])
    print(f"Label range: min={labels.min().item():.4f}, max={labels.max().item():.4f}, mean={labels.mean().item():.4f}")

    with open(output_dir / "global_graph_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    train_idx, val_idx, test_idx = split_indices(graph.number_of_nodes("topic"), seed=args.seed)
    with open(output_dir / "split_indices.json", "w", encoding="utf-8") as f:
        json.dump({"train_idx": train_idx.tolist(), "val_idx": val_idx.tolist(), "test_idx": test_idx.tolist()}, f, ensure_ascii=False, indent=2)

    model = GlobalTopicRegressor_HAN(
        graph,
        input_dim_dict={ntype: int(graph.nodes[ntype].data["feat"].shape[1]) for ntype in graph.ntypes},
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, patience_counter, history = float("inf"), 0, []
    best_path = output_dir / "Global_HAN_weibo_topic.pkl"
    labels_device = labels.to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pred, _ = model(graph, device)
        y_train = labels_device[train_idx.to(device)].squeeze(-1)
        p_train = pred[train_idx.to(device)].squeeze(-1)
        loss = F.mse_loss(p_train, y_train)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        val_result = evaluate(model, graph, labels, val_idx, device)
        test_result = evaluate(model, graph, labels, test_idx, device)

        row = {
            "epoch": epoch,
            "train_loss": float(loss.item()),
            **{f"val_{k}": v for k, v in val_result.items() if k not in ["y_true", "y_pred"]},
            **{f"test_{k}": v for k, v in test_result.items() if k not in ["y_true", "y_pred"]}
        }
        history.append(row)
        print(f"[Epoch {epoch:03d}] train_loss={loss.item():.4f} | val_MAE={val_result['MAE']:.4f}, val_RMSE={val_result['RMSE']:.4f}, val_nMSE={val_result['nMSE']:.4f}, val_MAPE={val_result['MAPE']:.4f}, val_R2={val_result['R2']:.4f} | test_MAE={test_result['MAE']:.4f}, test_RMSE={test_result['RMSE']:.4f}, test_nMSE={test_result['nMSE']:.4f}, test_MAPE={test_result['MAPE']:.4f}, test_R2={test_result['R2']:.4f}")

        if val_result["loss"] < best_val:
            best_val, patience_counter = val_result["loss"], 0
            torch.save(model.state_dict(), best_path)
            print(f"  saved best model to {best_path}")
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    with open(output_dir / "train_history_global_han.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    model.load_state_dict(torch.load(best_path, map_location=device))
    final_train = evaluate(model, graph, labels, train_idx, device)
    final_val = evaluate(model, graph, labels, val_idx, device)
    final_test = evaluate(model, graph, labels, test_idx, device)

    final_result = {
        "train": {k: v for k, v in final_train.items() if k not in ["y_true", "y_pred"]},
        "val": {k: v for k, v in final_val.items() if k not in ["y_true", "y_pred"]},
        "test": {k: v for k, v in final_test.items() if k not in ["y_true", "y_pred"]},
        "best_model_path": str(best_path)
    }
    with open(output_dir / "final_result_global_han.json", "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    save_outputs(model, graph, labels, meta, output_dir, device)

    print("Done.")
    print(f"模型参数已保存：{best_path}")
    print(f"topic 节点表示已保存：{output_dir / 'topic_node_embeddings.npy'}")
    print(f"预测结果已保存：{output_dir / 'topic_predictions.csv'}")


if __name__ == "__main__":
    main()