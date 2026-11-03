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
from model import GATModel


def split_indices(n, train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train, n_val = int(n * train_ratio), int(n * val_ratio)
    return torch.tensor(idx[:n_train]), torch.tensor(idx[n_train:n_train+n_val]), torch.tensor(idx[n_train+n_val:])


def evaluate(model, graph, labels, indices, device):
    model.eval()
    with torch.no_grad():
        pred, _ = model(graph, device)
    labels = labels.to(device)
    indices = indices.to(device)
    y_true = labels[indices].squeeze(-1)
    y_pred = pred[indices].squeeze(-1)

    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    mae = float(np.mean(np.abs(y_true_np - y_pred_np)))
    rmse = float(np.sqrt(np.mean((y_true_np - y_pred_np) ** 2)))
    mse = float(np.mean((y_true_np - y_pred_np) ** 2))
    y_var = float(np.var(y_true_np))
    nmse = mse / (y_var + 1e-12)
    mask = y_true_np != 0
    mape = float(np.mean(np.abs((y_true_np[mask] - y_pred_np[mask]) / (y_true_np[mask] + 1e-12))) * 100) if mask.sum() > 0 else 0.0
    ss_res = np.sum((y_true_np - y_pred_np) ** 2)
    ss_tot = np.sum((y_true_np - np.mean(y_true_np)) ** 2) + 1e-12
    r2 = float(1 - ss_res / ss_tot)
    return mae, rmse, nmse, mape, r2


def main():
    parser = argparse.ArgumentParser(description="GAT end-to-end regression")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max_topics", type=int, default=-1)
    parser.add_argument("--num_runs", type=int, default=1, help="Number of repeated experiments")
    args = parser.parse_args()

    set_random_seed(args.seed)
    dataset_root = clean_path(args.dataset_root)
    output_dir = clean_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    print(f"Using device: {device}")

    # 构建图（只构建一次）
    topic_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    video_dim = infer_embedding_dim(topic_dirs, ["videoembedding"], 768)
    image_dim = infer_embedding_dim(topic_dirs, ["imageembedding"], 768)
    audio_dim = infer_embedding_dim(topic_dirs, ["audioembedding"], 512)
    text_dim = infer_embedding_dim(topic_dirs, ["textembedding", "text_embeddings", "bertembedding", "contextembedding"], 768)
    topic_sem_dim = infer_topic_embedding_dim(topic_dirs, 768)
    graph, labels, meta = build_global_graph(dataset_root, video_dim, audio_dim, image_dim, text_dim, topic_sem_dim, args.max_topics)
    total_topics = graph.number_of_nodes("topic")
    input_dim_dict = {ntype: int(graph.nodes[ntype].data["feat"].shape[1]) for ntype in graph.ntypes}
    labels_np = labels.squeeze().numpy()

    all_results = []
    for run in range(1, args.num_runs + 1):
        run_seed = args.seed + run - 1
        print(f"\n========== Run {run}/{args.num_runs} (seed={run_seed}) ==========")
        set_random_seed(run_seed)
        train_idx, val_idx, test_idx = split_indices(total_topics, seed=run_seed)

        run_dir = output_dir / f"run_{run}"
        run_dir.mkdir(exist_ok=True)

        model = GATModel(
            graph=graph,
            input_dim_dict=input_dim_dict,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            n_heads=args.num_heads,
            dropout=args.dropout
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = run_dir / "best_gat_model.pkl"
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

            model.eval()
            with torch.no_grad():
                pred_val, _ = model(graph, device)
                y_val = labels_device[val_idx.to(device)].squeeze(-1)
                p_val = pred_val[val_idx.to(device)].squeeze(-1)
                val_loss = F.mse_loss(p_val, y_val).item()
            model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            if epoch % 20 == 0:
                print(f"  Epoch {epoch:03d} | train_loss={loss.item():.4f} | val_loss={val_loss:.4f}")

        # 加载最佳模型并测试
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_mae, test_rmse, test_nmse, test_mape, test_r2 = evaluate(model, graph, labels, test_idx, device)
        print(f"Run {run} Test: MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, nMSE={test_nmse:.4f}, MAPE={test_mape:.2f}%, R2={test_r2:.4f}")

        all_results.append({
            "run": run, "seed": run_seed,
            "MAE": test_mae, "RMSE": test_rmse, "nMSE": test_nmse, "MAPE": test_mape, "R2": test_r2
        })

        # 保存预测结果
        model.eval()
        with torch.no_grad():
            pred, _ = model(graph, device)
            topic_pred = pred.cpu().numpy().reshape(-1)
        pd.DataFrame({
            "topic_idx": test_idx.numpy(),
            "topic_name": [meta["topic_names"][i] for i in test_idx.numpy()],
            "label": labels_np[test_idx.numpy()],
            "pred_gat": topic_pred[test_idx.numpy()]
        }).to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    # 汇总多次实验
    metrics = ["MAE", "RMSE", "nMSE", "MAPE", "R2"]
    print("\n===== Summary over runs =====")
    summary = {}
    for m in metrics:
        values = [res[m] for res in all_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        summary[m] = {"mean": mean_val, "std": std_val}
        if m == "MAPE":
            print(f"{m}: {mean_val:.2f}% ± {std_val:.2f}%")
        else:
            print(f"{m}: {mean_val:.4f} ± {std_val:.4f}")

    with open(output_dir / "all_runs_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(all_results).to_csv(output_dir / "all_runs.csv", index=False)
    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()