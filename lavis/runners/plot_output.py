import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def parse_log(path):
    train_epochs, train_losses = [], []
    val_epochs, val_losses = [], []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "run" in obj and "model" in obj and "datasets" in obj:
                continue

            if "train_loss" in obj:
                train_epochs.append(int(obj["train_epoch"]))
                train_losses.append(float(obj["train_loss"]))
            elif "val_loss" in obj:
                val_epochs.append(int(obj["val_epoch"]))
                val_losses.append(float(obj["val_loss"]))

    df_train = pd.DataFrame({"epoch": train_epochs, "train_loss": train_losses})
    df_val = pd.DataFrame({"epoch": val_epochs, "val_loss": val_losses})
    return df_train.sort_values("epoch"), df_val.sort_values("epoch")


def main(log_path):
    df_train, df_val = parse_log(log_path)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df_train["epoch"], df_train["train_loss"], label="Train Loss")
    plt.plot(df_val["epoch"], df_val["val_loss"], "o-", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save in result folder
    log_path = Path(log_path)
    result_dir = log_path.parent / "result"
    result_dir.mkdir(exist_ok=True)
    out_path = result_dir / "loss_curve.png"
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved plot to: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_losses.py /path/to/log.txt")
        sys.exit(1)
    main(sys.argv[1])



    # to run 
    # python plot_output.py ../output/BLIP2/CAM_FRONT_Qform_mini/20250724143/log.txt

    # python plot_output.py ../output/BLIP2/mini_learnable_position_added/20250726102/log.txt
    # python plot_output.py ../output/BLIP2/mini_learnable_position_added/20250726153/log.txt




