"""Finish the batch delivery by copying its four selected videos to Windows.

This is one bounded batch step, not an assistant monitoring loop. It waits for
the remote selection-complete marker, tolerates SSH reconnects, and then exits.
"""

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="4090B")
    parser.add_argument("--jump", default="4090A")
    parser.add_argument("--remote", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--desktop", type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    remote_wait = f"""from pathlib import Path
import json,time
p=Path({args.remote!r})
deadline=time.monotonic()+72*3600
while time.monotonic()<deadline:
 if (p/'SELECTED_COMPLETE.json').exists():
  files=sorted((p/'selected').glob('*.mp4'))
  assert len(files)==4
  print(json.dumps([{{'path':str(f),'bytes':f.stat().st_size}} for f in files]),flush=True)
  break
 if (p/'status.json').exists():
  state=json.loads((p/'status.json').read_text())
  if state.get('phase')=='failed':raise RuntimeError(state.get('error'))
 time.sleep(60)
else:raise TimeoutError('Batch selection did not finish within 72 hours')
"""
    options = [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
    ]
    if args.jump:
        options += ["-o", f"ProxyJump={args.jump}"]
    deadline = time.monotonic() + 72 * 3600
    while True:
        result = subprocess.run(
            ["ssh", *options, args.host, "python3 -c " + shlex.quote(remote_wait)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if result.returncode == 0:
            files = json.loads(result.stdout)
            break
        if result.returncode != 255 or time.monotonic() > deadline:
            raise RuntimeError(result.stderr)
        time.sleep(60)
    for item in files:
        name = Path(item["path"]).name
        target = args.destination / name
        for attempt in range(5):
            transfer = subprocess.run(["scp", *options, args.host + ":" + item["path"], str(target)], check=False)
            if transfer.returncode == 0 and target.stat().st_size == item["bytes"]:
                break
            if attempt == 4:
                raise RuntimeError(f"Could not transfer {name}")
            time.sleep(30)
        shutil.copy2(target, args.desktop / name)
    for name in ("ranking.csv", "ranking.json", "selection.json", "run_manifest.json", "SELECTED_COMPLETE.json"):
        subprocess.run(
            ["scp", *options, args.host + ":" + args.remote + "/" + name, str(args.destination / name)], check=True
        )
    (args.destination / "DELIVERY_COMPLETE.json").write_text(
        json.dumps(
            {
                "videos": [Path(f["path"]).name for f in files],
                "desktop": str(args.desktop),
                "completed_at_unix": time.time(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
