from pathlib import Path
import csv
import json
import torch
import torch.nn.functional as F


def train_bc(model, train_loader, val_loader, cfg: dict, device: str):
    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )

    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = Path(cfg["train"].get("metrics_json", ckpt_dir / "metrics.json"))
    metrics_csv = Path(cfg["train"].get("metrics_csv", ckpt_dir / "metrics.csv"))
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history = []
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        total = 0.0
        for step, (obs, act) in enumerate(train_loader, start=1):
            obs = obs.to(device)
            act = act.to(device)
            pred = model(obs)
            loss = F.mse_loss(pred, act)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

            if step % int(cfg["train"]["log_interval"]) == 0:
                print(f"[train] epoch={epoch} step={step} loss={loss.item():.6f}")

        train_loss = total / max(len(train_loader), 1)
        val_loss = evaluate_bc(model, val_loader, device)
        print(f"[epoch {epoch}] train={train_loss:.6f} val={val_loss:.6f}")
        history.append({"epoch": epoch, "train_loss": float(train_loss), "val_loss": float(val_loss)})

        with metrics_json.open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        with metrics_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
            writer.writeheader()
            writer.writerows(history)

        last_ckpt = ckpt_dir / "last.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, last_ckpt)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, ckpt_dir / "best.pt")

    return history


@torch.no_grad()
def evaluate_bc(model, val_loader, device: str):
    model.eval()
    total = 0.0
    n = 0
    for obs, act in val_loader:
        obs = obs.to(device)
        act = act.to(device)
        pred = model(obs)
        loss = F.mse_loss(pred, act)
        total += loss.item()
        n += 1
    return total / max(n, 1)
