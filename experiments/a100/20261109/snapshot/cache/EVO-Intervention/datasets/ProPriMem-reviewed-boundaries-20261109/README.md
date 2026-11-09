# ProPriMem

47 tasks · 2,074 trajectories.

| Suite | Tasks | Trajectories |
|---|---:|---:|
| [Count](Count/README.md) | 14 | 700 |
| [Order](Order/README.md) | 14 | 551 |
| [State](State/README.md) | 19 | 823 |

Each task contains videos/, metadata.json (task details), and metadata.jsonl (trajectory index); annotations/ is present when stage labels are available. Paths in each index are relative to its task directory.

Dataset v1.3.0+state.b682e418.cap50 · Split v1.3.0+state.b682e418.cap50 · 1,656 train / 418 val. Assignments in [manifest.json](manifest.json) are frozen.

Original task names, provenance, and citation links are included in each task metadata.

The seven EvoMind Order tasks O08–O14 contain 237 unannotated trajectories. Original recordings have no stage-boundary labels; source instructions are retained. Their existing frozen split assignments are unchanged.

v1.2.0 removes the last 50 indexed trajectories from each of 11 tasks that had 100 trajectories. Each affected task now has 40 training and 10 validation trajectories. In v1.2.0, the other 27 task splits and both insert-screw datasets were unchanged. The [change record](provenance/prune-tail-50-20261101/manifest.json) lists the removed samples and files, and the previous repository revision.

v1.3.0 supplements six existing AgiBot/RoboMME tasks to 50 paired trajectories each. Existing sample split assignments are preserved; new samples follow the existing validation proportion.

Local extension (2026-11-03): State S11–S19 were imported directly on this server from syding9/ProPriMem-State at b682e418c6aad6d721ddef164da7c79a22d4c67a. These 323 real-robot trajectories have synchronized three-view videos and metadata, with no stage annotations. All existing tasks and split assignments are preserved; new tasks use the upstream frozen splits. See provenance/state-extension-20261103/import.json.

Local cap (2026-11-03): every Count/Order/State task now has at most 50 trajectories. C07–C09 and S08–S10 retain the first 40 existing training and first 10 existing validation trajectories in metadata order. Retained episode IDs and split assignments are unchanged. The 262 removed trajectories include all their views and annotation files. See [change record](provenance/cap50-20261103/manifest.json).
