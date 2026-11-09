# ProPriMem

38 tasks · 2,013 trajectories.

| Suite | Tasks | Trajectories |
|---|---:|---:|
| [Count](Count/README.md) | 14 | 844 |
| [Order](Order/README.md) | 14 | 551 |
| [State](State/README.md) | 10 | 618 |

Each task contains videos/, annotations/, metadata.json (task details), and metadata.jsonl (trajectory index). Paths in each index are relative to its task directory.

Dataset v1.3.0 · Split v1.3.0 · 1,606 train / 407 val. Assignments in [manifest.json](manifest.json) are frozen.

Original task names, provenance, and citation links are included in each task metadata.

The seven EvoMind Order tasks O08–O14 contain 237 unannotated trajectories. Original recordings have no stage-boundary labels; source instructions are retained. Their existing frozen split assignments are unchanged.

v1.2.0 removes the last 50 indexed trajectories from each of 11 tasks that had 100 trajectories. Each affected task now has 40 training and 10 validation trajectories. In v1.2.0, the other 27 task splits and both insert-screw datasets were unchanged. The [change record](provenance/prune-tail-50-20261101/manifest.json) lists the removed samples and files, and the previous repository revision.

v1.3.0 supplements six existing AgiBot/RoboMME tasks to 50 paired trajectories each. Existing sample split assignments are preserved; new samples follow the existing validation proportion.
