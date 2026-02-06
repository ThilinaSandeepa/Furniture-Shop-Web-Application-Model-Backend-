import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from model import RoomPlannerModel

K = 5
df = pd.read_excel("Dataset/dataset.xlsx")
hits, mrrs, price_gaps = [], [], []

for i, row in df.iterrows():
    df_train = df.drop(index=i)
    tmp_path = "Dataset/tmp_eval.xlsx"
    df_train.to_excel(tmp_path, index=False)

    model = RoomPlannerModel(tmp_path)
    res = model.search(row.room_size, row.room_type, row.style, row.budget_range, limit=K)
    suggestions = [s["id"] for s in res["suggestions"]]

    # Normalize strings to avoid hidden whitespace / case issues
    def norm(s):
        return str(s).strip().lower()

    df_train_norm = df_train.copy()
    for col in ["room_size", "room_type", "style", "budget_range"]:
        df_train_norm[col] = df_train_norm[col].apply(norm)

    row_norm = {
        "room_size": norm(row.room_size),
        "room_type": norm(row.room_type),
        "style": norm(row.style),
        "budget_range": norm(row.budget_range),
    }

    # any item matching the full attributes counts as relevant
    relevant = set(df_train.loc[
        (df_train_norm.room_size == row_norm["room_size"]) &
        (df_train_norm.room_type == row_norm["room_type"]) &
        (df_train_norm.style == row_norm["style"]) &
        (df_train_norm.budget_range == row_norm["budget_range"])
    ]["id"])

    if relevant:
        hit = int(any(sid in relevant for sid in suggestions))
        hits.append(hit)
        rank = next((r for r, sid in enumerate(suggestions, 1) if sid in relevant), None)
        mrrs.append(1/rank if rank else 0)
        # price gap vs closest relevant item
        price_gaps.append(abs(row.price - df_train.loc[df_train.id.isin(relevant), "price"].mean()))
    # If there are no comparable items for this query, skip it
    # so we do not distort the metric denominators.
    else:
        continue

total_evals = len(hits)
output_path = Path("evaluation_metrics.png")

# Debug info to understand evaluable queries
print(f"Comparable queries processed: {total_evals} (dataset rows: {len(df)})")

if total_evals == 0:
    print("No comparable items found in dataset to compute metrics. Please add more overlapping samples.")
    # Save an empty/zero chart so the user can still find the file
    metrics = ["Hit@K", "MRR", "Avg Price Gap"]
    values = [0, 0, 0]
    plt.figure(figsize=(6, 4))
    plt.bar(metrics, values, color=["steelblue", "seagreen", "darkorange"])
    plt.ylabel("Value")
    plt.title("Model Evaluation Metrics (no comparable samples)")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved chart to {output_path.resolve()}")
else:
    hit_rate = sum(hits) / total_evals
    mrr_score = sum(mrrs) / total_evals
    avg_gap = pd.Series(price_gaps).mean()

    print(f"Evaluated queries: {total_evals}")
    print(f"Hit@{K}: {hit_rate:.3f}")
    print(f"MRR: {mrr_score:.3f}")
    print(f"Avg price gap: {avg_gap:.1f}")

    # Save a simple bar chart of the metrics
    metrics = ["Hit@K", "MRR", "Avg Price Gap"]
    values = [hit_rate, mrr_score, avg_gap]
    plt.figure(figsize=(6, 4))
    plt.bar(metrics, values, color=["steelblue", "seagreen", "darkorange"])
    plt.ylabel("Value")
    plt.title("Model Evaluation Metrics")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved chart to {output_path.resolve()}")