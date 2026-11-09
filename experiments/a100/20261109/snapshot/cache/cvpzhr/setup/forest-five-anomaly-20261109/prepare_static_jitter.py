"""Stationary and temporal-jitter augmentation, with anchor-fixed PT."""
from collections import Counter, defaultdict
from fractions import Fraction
import json, math, os, random, shutil, time
from pathlib import Path
import prepare_repeats as original

now=original.now
read_json=original.read_json
write_json=original.write_json
sha=original.sha
rows_at=original.rows_at
canonical_hash=original.canonical_hash
source_fps=original.source_fps
number=original.number
check_source=original.check_source
encode_raw=original.encode_raw
position_features=original.position_features
assigned_episodes=original.assigned_episodes
VIEWS=original.VIEWS

POLICY=read_json(Path(__file__).parent/"augmentation-policy.json")
assert POLICY["status"]=="confirmed"
CONTRACT=dict(version="static-jitter-anchor-pt-v1",seed=42,sample_fps=5,
              scope="1656 independent training episodes; one augmented version each",
              stationary=POLICY["stationary"],jitter=POLICY["jitter"],
              shared_anchor_order=POLICY["shared_anchor_order"],
              views="all available views replay the same source observation index",
              pt="original rows use source PT; every inserted row uses its event anchor PT",
              history="recompute causal memory after every original or inserted observation",
              validation="unchanged base validation paths; one explicitly excluded review episode",
              training_started=False)

def choose_events(rows,episode_id,fps,source_frames):
    rng=random.Random(f"42:{episode_id}:static-jitter-anchor-pt-v1")
    stationary_count=rng.randint(1,3)
    jitter_count=math.ceil(Fraction(source_frames,1)/fps/15)
    events=[]
    for kind,count,maximum in [("stationary",stationary_count,15),("jitter",jitter_count,5)]:
        for anchor in sorted(rng.sample(range(len(rows)),count)):
            requested=rng.randint(0,maximum)
            frames=min(requested,anchor) if kind=="jitter" else requested
            events.append(dict(kind=kind,anchor=anchor,anchor_frame_index=rows[anchor]["frame_index"],
                               frames=frames,requested_frames=requested))
    return events

def expand(rows,events,fps):
    by_anchor=defaultdict(list)
    for event in events:by_anchor[event["anchor"]].append(event)
    added=0;sample_index=0
    for i,anchor in enumerate(rows):
        position=Fraction(anchor["frame_index"])+Fraction(added,5)*fps
        yield dict(frame_index=number(position),source_frame_index=anchor["frame_index"],
                   source_sample_index=i,anchor_sample_index=i,sample_index=sample_index,
                   timestamp_seconds=float(position/fps),kind="original",images=anchor["images"].copy(),pt=anchor["pt"])
        sample_index+=1
        for event in sorted(by_anchor[i],key=lambda e:e["kind"]!="stationary"):
            n=event["frames"]
            if event["kind"]=="stationary":
                path=[(i,"stationary")]*n
            else:
                path=[(j,"reverse") for j in range(i-1,i-n-1,-1)]+[(j,"forward") for j in range(i-n+1,i+1)]
            for source,kind in path:
                added+=1;position=Fraction(anchor["frame_index"])+Fraction(added,5)*fps
                frame=rows[source]
                yield dict(frame_index=number(position),source_frame_index=frame["frame_index"],
                           source_sample_index=source,anchor_sample_index=i,sample_index=sample_index,
                           timestamp_seconds=float(position/fps),kind=kind,images=frame["images"].copy(),pt=anchor["pt"])
                sample_index+=1

def build_episode(plan,episode,visual,processor):
    import torch
    from PIL import Image
    from mem_lite.memory.manager import MemoryManager
    check_source(episode)
    output,source=Path(episode["output_dir"]),Path(episode["source_dir"])
    if (output/"summary.json").exists():return verify_episode(plan,episode)
    assert not output.exists()
    temporary=output.with_name(output.name+".partial")
    ownership=dict(plan_sha256=plan["plan_sha256"],episode_id=episode["episode_id"])
    if temporary.exists():
        assert read_json(temporary/".augmentation-owned.json")==ownership
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    write_json(temporary/".augmentation-owned.json",ownership)
    (temporary/"history").mkdir()
    summary=read_json(source/"summary.json");rows=rows_at(source)
    settings=summary["history_source"];fps=source_fps(summary)
    memory=MemoryManager(summary["history_dim"],**settings["memory_config"])
    previous=None;recent={};token_max=0;history_bytes=0;started=time.monotonic();encoded=0
    with (temporary/"samples.jsonl").open("w") as out:
        for row in expand(rows,episode["events"],fps):
            index=row["source_sample_index"]
            if row["kind"]=="original":
                for relative in row["images"].values():
                    if relative is not None:
                        target=temporary/relative;target.parent.mkdir(parents=True,exist_ok=True)
                        os.link(source/relative,target)
                with Image.open(source/row["images"]["cam_high"]) as im:
                    recent[index]=encode_raw(visual,processor,im.convert("RGB"),settings)
                for old in list(recent):
                    if old<index-5:del recent[old]
                encoded+=1
            features,height,width=recent[index]
            history=memory.get_memory_tokens()
            tokens=0 if history is None else len(history)
            assert tokens<=summary["history_capacity"]
            history_path=None
            if history is not None:
                history_path=f'history/{row["sample_index"]:08d}.pt'
                torch.save(dict(tokens=history.cpu(),through_frame=previous),temporary/history_path)
                history_bytes+=(temporary/history_path).stat().st_size
            row["history"]=history_path;row["history_token_count"]=tokens
            out.write(json.dumps(row,ensure_ascii=False)+"\n")
            token_max=max(token_max,tokens)
            memory.update(position_features(features,height,width,row["frame_index"]))
            previous=row["frame_index"]
    assert row["sample_index"]+1==episode["augmented_samples"] and encoded==episode["source_samples"]
    write_json(temporary/"augmentation.json",dict(plan_sha256=plan["plan_sha256"],**episode))
    summary.update(num_samples=episode["augmented_samples"],
       augmentation=dict(**CONTRACT,plan_sha256=plan["plan_sha256"],original_episode_id=episode["episode_id"],
                         original_cache=str(source),source_samples=episode["source_samples"],
                         added_samples=episode["added_samples"],events=episode["events"],
                         frame_index_unit="source-fps frame-equivalent position; may be fractional",
                         frame_index_note="source_frame_index identifies replayed image; anchor_sample_index identifies PT source"),
       generation=dict(completed_at=now(),seconds=time.monotonic()-started,history_bytes=history_bytes,
                       max_history_tokens=token_max,raw_visual_forward_calls=encoded,memory_updates=episode["augmented_samples"]))
    write_json(temporary/"summary.json",summary)
    temporary.rename(output)
    return verify_episode(plan,episode)

def verify_episode(plan,episode):
    import torch
    check_source(episode)
    directory,source=Path(episode["output_dir"]),Path(episode["source_dir"])
    summary=read_json(directory/"summary.json");source_summary=read_json(source/"summary.json")
    rows,original_rows=rows_at(directory),rows_at(source)
    assert summary["augmentation"]["plan_sha256"]==plan["plan_sha256"]
    assert summary["history_source"]==source_summary["history_source"]
    assert len(rows)==summary["num_samples"]==episode["augmented_samples"]
    original=[r for r in rows if r["kind"]=="original"]
    assert [r["source_sample_index"] for r in original]==list(range(len(original_rows)))
    by_anchor=defaultdict(list)
    for r in rows:
        if r["kind"]!="original":by_anchor[r["anchor_sample_index"]].append(r)
    expected_by_anchor=defaultdict(list)
    for event in sorted(episode["events"],key=lambda e:(e["anchor"],e["kind"]!="stationary")):
        anchor,n=event["anchor"],event["frames"]
        if event["kind"]=="stationary":
            expected_by_anchor[anchor].extend([("stationary",anchor)]*n)
        else:
            expected_by_anchor[anchor].extend(("reverse",j) for j in range(anchor-1,anchor-n-1,-1))
            expected_by_anchor[anchor].extend(("forward",j) for j in range(anchor-n+1,anchor+1))
    assert {i:[(r["kind"],r["source_sample_index"]) for r in v] for i,v in by_anchor.items()}=={
        i:v for i,v in expected_by_anchor.items() if v}
    fps=source_fps(source_summary);previous=None;added=0;checked_images=set();checked_histories=0
    selected={1,len(rows)-1}
    for i,row in enumerate(rows):
        if row["kind"]!="original" and (i==0 or rows[i-1]["kind"]!=row["kind"] or i+1==len(rows) or rows[i+1]["kind"]!=row["kind"]):
            selected.update(j for j in (i-1,i,i+1) if 0<=j<len(rows))
    for i,row in enumerate(rows):
        src=original_rows[row["source_sample_index"]];anchor=original_rows[row["anchor_sample_index"]]
        inserted=row["kind"]!="original"
        if inserted:added+=1
        else:assert row["anchor_sample_index"]==row["source_sample_index"]
        position=Fraction(anchor["frame_index"])+Fraction(added,5)*fps
        assert row["sample_index"]==i and row["source_frame_index"]==src["frame_index"]
        assert row["images"]==src["images"] and row["pt"]==anchor["pt"]
        assert abs(row["frame_index"]-float(position))<1e-8
        assert abs(row["timestamp_seconds"]-float(position/fps))<1e-8
        assert previous is None or row["frame_index"]>previous
        if inserted:
            assert i>0 and row["pt"]==rows[i-1]["pt"]
            assert abs(row["timestamp_seconds"]-rows[i-1]["timestamp_seconds"]-.2)<1e-8
        for relative in row["images"].values():
            if relative is not None and relative not in checked_images:
                a,b=(directory/relative).stat(),(source/relative).stat()
                assert (a.st_dev,a.st_ino,a.st_size)==(b.st_dev,b.st_ino,b.st_size)
                checked_images.add(relative)
        assert (row["history"] is None)==(i==0)
        if row["history"] is not None:
            path=directory/row["history"];assert path.is_file() and path.stat().st_size>100
            if i in selected:
                value=torch.load(path,map_location="cpu",weights_only=True);tokens=value["tokens"]
                assert value["through_frame"]==previous and tokens.ndim==2
                assert tokens.shape[1]==summary["history_dim"] and len(tokens)<=summary["history_capacity"]
                assert len(tokens)==row["history_token_count"] and tokens.dtype==torch.bfloat16
                assert torch.isfinite(tokens).all().item()
                checked_histories+=1
        previous=row["frame_index"]
    assert len(list((directory/"history").glob("*.pt")))==len(rows)-1
    counts=Counter(r["kind"] for r in rows)
    assert added==episode["added_samples"]
    return dict(episode_id=episode["episode_id"],samples=len(rows),added_samples=added,
                anchors=len(episode["events"]),stationary_anchors=sum(e["kind"]=="stationary" for e in episode["events"]),
                jitter_anchors=sum(e["kind"]=="jitter" for e in episode["events"]),
                stationary_samples=counts["stationary"],jitter_samples=counts["reverse"]+counts["forward"],
                inserted_pt_fixed_to_anchor=True,checked_histories=checked_histories,checked_images=len(checked_images),
                history_bytes=summary["generation"]["history_bytes"],seconds=summary["generation"]["seconds"],
                max_history_tokens=summary["generation"]["max_history_tokens"])
