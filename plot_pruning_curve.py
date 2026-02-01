import json

import matplotlib.pyplot as plt


log_path = "pruning_debug.jsonl"
output_path = "pruning_curve.png"
show_plot = True


baseline_score = None
final_params = None
compression = None
points = []

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        event_type = event.get("event")
        if event_type == "baseline":
            baseline_score = event.get("baseline_score")
        if event_type == "final":
            final_params = event.get("final_params")
            compression = event.get("compression")
        if event_type == "accept":
            params = event.get("params")
            score = event.get("score_full")
            if params is not None and score is not None:
                points.append((params, score))

if final_params is None or compression is None:
    raise ValueError("Missing final_params or compression in pruning_debug.jsonl.")

initial_params = int(round(final_params / max(1.0 - compression, 1e-9)))

xs = []
ys = []

if baseline_score is not None:
    xs.append(0.0)
    ys.append(float(baseline_score))

for params, score in points:
    prune_pct = 100.0 * (1.0 - (params / initial_params))
    xs.append(prune_pct)
    ys.append(float(score))

plt.figure(figsize=(8, 5))
plt.plot(xs, ys, marker="o", linewidth=1.5)
plt.xlabel("Pruning percentage (%)")
plt.ylabel("Embedding similarity")
plt.title("Pruning curve")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_path, dpi=150)

if show_plot:
    plt.show()
