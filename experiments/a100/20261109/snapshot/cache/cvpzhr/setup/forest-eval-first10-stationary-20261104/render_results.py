from collections import defaultdict
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import evaluate_static as e

out=e.OUT
report=e.aug.read_json(out/"metrics.json")
records=[json.loads(line) for line in (out/"predictions.jsonl").read_text().splitlines()]
episodes=defaultdict(list)
for row in records: episodes[row["episode_id"]].append(row)
episode_metrics={episode:e.summarize(rows) for episode,rows in episodes.items()}
e.aug.write_json(out/"per-episode-metrics.json",episode_metrics)
tasks=e.PLAN["tasks"]
labels=[f"C{i+1:02d}" for i in range(len(tasks))]
fig,ax=plt.subplots(figsize=(11,4.5),layout="constrained")
xs=list(range(len(tasks)));width=.36
ax.bar([x-width/2 for x in xs],[report["by_task"][t]["mae_pp"] for t in tasks],width,label="All augmented observations")
ax.bar([x+width/2 for x in xs],[report["by_task"][t]["stationary_mae_pp"] for t in tasks],width,label="Inserted stationary observations")
ax.set_xticks(xs,labels);ax.set_ylabel("MAE (percentage points)")
ax.set_title("Forest epoch 1: stationary augmentation on the first 10 validation tasks")
ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True);ax.legend()
fig.savefig(out/"task-mae.png",dpi=180);plt.close(fig)

fig,axes=plt.subplots(5,2,figsize=(14,17),sharey=True,layout="constrained")
chosen={}
for task,ax in zip(tasks,axes.flat):
    candidates=sorted([name for name in episodes if name.startswith(task+"/")],key=lambda name:(episode_metrics[name]["mae_pp"],name))
    name=candidates[len(candidates)//2];chosen[task]=name
    rows=episodes[name]
    ax.plot([r["timestamp_seconds"] for r in rows],[100*r["target_pt"] for r in rows],color="#222222",lw=1.7,label="PT ground truth")
    ax.plot([r["timestamp_seconds"] for r in rows],[100*r["predicted_pt"] for r in rows],color="#2563eb",lw=1.2,label="Predicted progress")
    active=None
    for i,row in enumerate(rows):
        if row["kind"]=="stationary" and active is None: active=row["timestamp_seconds"]-.1
        if active is not None and (row["kind"]!="stationary" or i+1==len(rows)):
            end=(rows[i-1]["timestamp_seconds"] if row["kind"]!="stationary" else row["timestamp_seconds"])+.1
            ax.axvspan(active,end,color="#f59e0b",alpha=.22)
            active=None
    task_name=task.split("/")[1]
    ax.set_title(f"{task_name}\n{name.split('/')[-1]} | episode MAE {episode_metrics[name]['mae_pp']:.2f} pp",fontsize=9)
    ax.set_ylim(-2,102);ax.set_xlabel("Augmented time (s)");ax.set_ylabel("Progress (%)");ax.grid(alpha=.2)
axes.flat[0].legend(fontsize=8)
fig.suptitle("Representative episode per task (median overall episode MAE)\nOrange spans: inserted stationary frames; PT held at anchor value",fontsize=13)
fig.savefig(out/"progress-curves.png",dpi=160);plt.close(fig)
e.aug.write_json(out/"plot-selection.json",dict(rule="upper median episode by overall augmented MAE within each task",episodes=chosen))
print(json.dumps(dict(plots=[str(out/"task-mae.png"),str(out/"progress-curves.png")],episodes=len(episode_metrics))))
