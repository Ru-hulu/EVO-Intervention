# ProPriMem

31 tasks · 816 trajectories.

| Suite | Tasks | Trajectories |
|---|---:|---:|
| [Count](Count/README.md) | 14 | 350 |
| [Order](Order/README.md) | 7 | 166 |
| [State](State/README.md) | 10 | 300 |

Each task contains videos/, annotations/, metadata.json (task details), and metadata.jsonl (trajectory index). Paths in each index are relative to its task directory.

Dataset v1.0.0 · Split v1.0.0 · 796 train / 20 val. Assignments in [manifest.json](manifest.json) are frozen.

Original task names, provenance, and citation links are included in each task metadata.

Local A100 subset: at most 30 trajectories per task, retaining the lowest numeric episode IDs. Original split assignments are filtered without reassignment; some tasks now have no validation episodes.
Additional root tasks (outside the 31-task manifest): insert-screw 30 trajectories; insert-screw-recover 22 trajectories. Total across all 33 tasks: 868 trajectories.
Source snapshot: 9ea7859fd65408152cab93a3fb4fb0ad0acf6dca. This is a locally reduced derivative, not the full upstream snapshot.
