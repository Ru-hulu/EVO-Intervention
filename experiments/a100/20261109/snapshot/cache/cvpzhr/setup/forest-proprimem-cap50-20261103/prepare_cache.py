"""Build base and repeated-frame Forest caches only after every dense PT passes validation."""
import argparse, fcntl, json, os, shutil, signal, subprocess, sys, time
from collections import Counter
from fractions import Fraction
from pathlib import Path
import prepare_pt as p
import prepare_static_jitter as repeat

SETUP, BASE = p.SETUP, p.BASE
AUG = Path("/cache/cvpzhr/datasets/forest-proprimem-cap50-5fps-static-jitter-seed42-20261104")
repeat.CONTRACT = dict(repeat.CONTRACT, scope="1656 independent training episodes; one augmented version each")

def frame_indices(row):
    step=Fraction(row["fps_numerator"],row["fps_denominator"])/5
    result=[]; k=0
    while True:
        value=k*step
        index=(value.numerator+value.denominator-1)//value.denominator
        if index>=row["frames"]:break
        result.append(index);k+=1
    if result[-1]!=row["frames"]-1:result.append(row["frames"]-1)
    assert len(result)==len(set(result))
    return result

def make_cache_plan():
    plan=p.get_plan()
    decision=p.read(SETUP/"view-decision.json")
    assert decision["user_confirmed"] and decision["action"]=="exclude_from_cache_pending_review"
    assert decision["source_id"]=="Order/03-swap_blocks_with_confirmation/episode_001342"
    excluded=[r for r in plan["sources"] if r["source_id"]==decision["source_id"]]
    assert len(excluded)==1 and excluded[0]["split"]=="validation"
    sources=[dict(r,cache_samples=len(frame_indices(r))) for r in plan["sources"] if r["source_id"]!=decision["source_id"]]
    loads=[0]*plan["workers"]
    for row in sorted(sources,key=lambda r:(-r["cache_samples"],r["source_id"])):
        worker=min(range(len(loads)),key=loads.__getitem__)
        row["cache_worker"]=worker;loads[worker]+=row["cache_samples"]
    assert dict(Counter(r["split"] for r in sources))==dict(train=1656,validation=417)
    result=dict(created_at=p.now(),base_plan_sha256=p.sha(SETUP/"plan.json"),
                decision_sha256=p.sha(SETUP/"view-decision.json"),history_source=plan["history_source"],
                workers=plan["workers"],worker_samples=loads,sources=sources,
                excluded_sources=[dict(**excluded[0],cache_status="pending_review",review_reason=decision["reason"])])
    p.write(SETUP/"cache-plan.json",result)
    p.write(BASE/"review.sources.json",result["excluded_sources"])
    p.write(SETUP/"pipeline-status.json",dict(phase="prepared",updated_at=p.now(),base_episodes=2073,
                                            train=1656,validation=417,review=1))
    print(json.dumps(dict(phase="prepared",base_episodes=2073,samples=sum(loads),worker_samples=loads,review=1)),flush=True)

def get_cache_plan():
    result=p.read(SETUP/"cache-plan.json")
    assert result["base_plan_sha256"]==p.sha(SETUP/"plan.json")
    assert result["decision_sha256"]==p.sha(SETUP/"view-decision.json")
    p.verify_sources(p.read(SETUP/"plan.json"))
    for path, digest in result.get("implementation_sha256", {}).items():
        assert p.sha(path) == digest, path
    return result

def label_gate():
    report=p.read(BASE/"labels-report.json")
    assert report["status"]=="completed" and report["episodes"]==2074
    assert report["reused_episodes"]==816 and report["generated_episodes"]==1258
    assert report["source_manifest_sha256"]==p.sha(p.ROOT/"manifest.json")
    assert p.read(SETUP/"pt-status.json")["phase"]=="completed"
    return report

def verify_base(row, directory, origin):
    import torch
    from mem_lite.datasets.m3_dataset import M3Dataset,VIEW_ORDER
    summary=p.read(directory/"summary.json")
    records=[json.loads(l) for l in (directory/"samples.jsonl").read_text().splitlines() if l.strip()]
    metadata,values=p.progress_check(row)
    assert summary["history_source"]==origin
    assert summary["episode_id"]==row["source_id"] and summary["task_prompt"]==row["task_prompt"]
    assert summary["history_dim"]==2560 and summary["history_capacity"]==1008
    assert summary["source"]["split"]==row["split"]
    assert summary["source"]["videos"]==row["videos"]
    assert summary["source"]["progress_sha256"]==metadata["units"][0]["progress_sha256"]
    expected=frame_indices(row)
    assert len(records)==summary["num_samples"]==len(expected)
    assert [r["frame_index"] for r in records]==expected
    chosen={1,len(records)//2,len(records)-1}
    previous=None; history_bytes=0;checked=0
    for i,sample in enumerate(records):
        assert sample["pt"]==values[sample["frame_index"]]
        assert (sample["history"] is None)==(i==0)
        assert set(sample["images"])==set(VIEW_ORDER)
        for slot,view in enumerate(VIEW_ORDER):
            relative=sample["images"][view]
            assert (relative is None)==(row["videos"][slot] is None)
            if relative is not None:assert (directory/relative).is_file()
        if sample["history"] is not None:
            path=directory/sample["history"]
            assert path.is_file() and path.stat().st_size>100
            history_bytes+=path.stat().st_size
            if i in chosen:
                cached=torch.load(path,map_location="cpu",weights_only=True)
                tokens=cached["tokens"]
                assert cached["through_frame"]==previous
                assert tokens.ndim==2 and tokens.shape[1]==2560 and 0<len(tokens)<=1008
                assert tokens.dtype==torch.bfloat16 and torch.isfinite(tokens).all().item()
                checked+=1
        previous=sample["frame_index"]
    assert len(list((directory/"history").glob("*.pt")))==len(records)-1
    # Exercise the actual loader on the beginning and end, including missing-view filling.
    dataset=M3Dataset([str(directory)])
    for i in {0,len(records)-1}:
        item=dataset[i];assert item["target_pt"]==records[i]["pt"] and len(item["images"])==3
    return dict(source_id=row["source_id"],split=row["split"],samples=len(records),
                history_bytes=history_bytes,checked_histories=checked,
                summary_sha256=p.sha(directory/"summary.json"),samples_sha256=p.sha(directory/"samples.jsonl"),
                directory=str(directory),verified_at=p.now())

def base_episode(row, visual, processor, origin):
    from mem_lite.memory.stream import StreamMemory
    from mem_lite.memory.vision import QwenFrameEncoder
    from mem_lite.datasets.prepare_stream import build_episode,iter_observations
    directory=BASE/"cache"/row["split"]/row["source_id"]
    if (directory/"summary.json").exists():
        return verify_base(row,directory,origin)
    assert not directory.exists(), "Incomplete output must be in the owned .partial directory"
    partial=directory.with_name(directory.name+".partial")
    assert partial.is_relative_to(BASE/"cache") and not partial.is_symlink()
    if partial.exists():shutil.rmtree(partial)
    metadata,values=p.progress_check(row)
    stream=StreamMemory(QwenFrameEncoder(visual),processor,origin["memory_config"],65536,4096)
    observations=iter_observations(row["videos"],dict(enumerate(values)),sample_fps=5.0,start_frame=0,end_frame=row["frames"])
    build_episode(observations,stream,partial,row["task_prompt"],row["source_id"],origin,
                  source={**row,**metadata["units"][0]})
    # Verify before exposing a final episode directory.
    receipt=verify_base(row,partial,origin)
    partial.rename(directory)
    receipt["directory"]=str(directory)
    return receipt

def receipt_path(stage, source_id):
    return SETUP/(stage+"-receipts")/(source_id+".json")

def cache_worker(stage,worker,limit=None):
    import torch
    from mem_lite.memory.vision import load_vision_encoder
    lock=(SETUP/("%s-worker-%d.lock"%(stage,worker))).open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(0.22)
    label_gate();plan=get_cache_plan()
    if stage=="base":
        entries=[r for r in plan["sources"] if r["cache_worker"]==worker]
        entries.sort(key=lambda r:(-r["cache_samples"],r["source_id"]))
    else:
        augment=p.read(AUG/"index.json")
        assert augment["contract"]==repeat.CONTRACT
        assert augment["protected_hashes"]=={path:p.sha(path) for path in augment["protected_hashes"]}
        entries=repeat.assigned_episodes(augment,worker,plan["workers"])
    if limit is not None:entries=entries[:limit]
    path=SETUP/("%s-worker-%d.json"%(stage,worker))
    state=dict(pid=os.getpid(),worker=worker,phase="loading_vision",total=len(entries),completed=0,samples=0,updated_at=p.now())
    p.write(path,state);started=time.monotonic()
    try:
        visual,processor,digest=load_vision_encoder(p.MODEL,"cuda")
        assert digest==p.VISION
        for row in entries:
            if shutil.disk_usage(BASE).free<200*2**30:raise RuntimeError("Less than 200 GiB free")
            source_id=row["source_id"] if stage=="base" else row["episode_id"]
            state.update(phase="building",current=source_id,updated_at=p.now());p.write(path,state)
            result=base_episode(row,visual,processor,plan["history_source"]) if stage=="base" else repeat.build_episode(augment,row,visual,processor)
            if stage=="augment":
                directory=Path(row["output_dir"])
                result.update(directory=str(directory),summary_sha256=p.sha(directory/"summary.json"),
                              samples_sha256=p.sha(directory/"samples.jsonl"),verified_at=p.now())
            p.write(receipt_path(stage,source_id),result)
            state.update(completed=state["completed"]+1,samples=state["samples"]+result["samples"],
                         elapsed_seconds=time.monotonic()-started,updated_at=p.now(),
                         peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            p.write(path,state)
            print(json.dumps(dict(stage=stage,worker=worker,source=source_id,completed=state["completed"],
                                  total=len(entries),samples=state["samples"],elapsed=state["elapsed_seconds"])),flush=True)
        state.update(phase="completed",current=None,updated_at=p.now());p.write(path,state)
    except BaseException as exc:
        state.update(phase="failed",error=repr(exc),updated_at=p.now());p.write(path,state);raise

def checked_receipts(stage,ids):
    results=[]
    for source_id in ids:
        result=p.read(receipt_path(stage,source_id));directory=Path(result["directory"])
        assert p.sha(directory/"summary.json")==result["summary_sha256"]
        assert p.sha(directory/"samples.jsonl")==result["samples_sha256"]
        results.append(result)
    return results

def finalize_base():
    from mem_lite.datasets.m3_dataset import M3Dataset
    plan=get_cache_plan();label_gate()
    results=checked_receipts("base",[r["source_id"] for r in plan["sources"]])
    lists={s:[str(BASE/"cache"/s/r["source_id"]) for r in plan["sources"] if r["split"]==s] for s in ("train","validation")}
    assert len(lists["train"])==1656 and len(lists["validation"])==417
    assert not set(lists["train"])&set(lists["validation"])
    samples={}
    for split,paths in lists.items():
        data=M3Dataset(paths);samples[split]=len(data);del data
        assert samples[split]==sum(r["samples"] for r in results if r["split"]==split)
        p.write(BASE/(split+".sample_files.json"),paths)
    assert sum(samples.values())==sum(r["cache_samples"] for r in plan["sources"])
    report=dict(status="completed_with_review_exclusion",completed_at=p.now(),episodes={s:len(v) for s,v in lists.items()},
                samples=samples,excluded_sources=plan["excluded_sources"],history_source=plan["history_source"],
                history_bytes=sum(r["history_bytes"] for r in results),labels_verified=True,
                history_causality_checked=True,split_manifest_preserved=True,training_started=False)
    p.write(BASE/"report.json",report);return report

def make_augmentation_plan():
    base=p.read(BASE/"report.json")
    assert base["status"]=="completed_with_review_exclusion"
    paths=p.read(BASE/"train.sample_files.json");validation=p.read(BASE/"validation.sample_files.json")
    assert len(paths)==1656 and len(validation)==417
    episodes=[];fps_counts=Counter();common=None
    for value in paths:
        directory=Path(value);summary=p.read(directory/"summary.json");rows=repeat.rows_at(directory)
        if common is None:common=summary["history_source"]
        assert summary["history_source"]==common and summary["source"]["split"]=="train"
        fps=repeat.source_fps(summary);fps_counts[str(fps)]+=1
        source_id=summary["source"]["unit_id"]
        events=repeat.choose_events(rows,source_id,fps,summary["source"]["frames"])
        added=sum(e["frames"]*(2 if e["kind"]=="jitter" else 1) for e in events)
        episodes.append(dict(episode_id=source_id,source_dir=str(directory),output_dir=str(AUG/"cache/train"/source_id),
          source_samples_sha256=p.sha(directory/"samples.jsonl"),source_summary_sha256=p.sha(directory/"summary.json"),
          source_samples=len(rows),augmented_samples=len(rows)+added,
          source_fps_numerator=fps.numerator,source_fps_denominator=fps.denominator,added_samples=added,events=events))
    protected={str(path):p.sha(path) for path in [BASE/"train.sample_files.json",BASE/"validation.sample_files.json",
               BASE/"index.json",BASE/"labels-report.json",SETUP/"cache-plan.json",SETUP/"prepare_repeats.py",
               SETUP/"prepare_static_jitter.py",SETUP/"augmentation-policy.json",Path(__file__)]}
    payload=dict(contract=repeat.CONTRACT,source=str(BASE),output=str(AUG),code=str(p.CODE),model=str(p.MODEL),
                 history_source=common,protected_hashes=protected,source_fps_counts=dict(fps_counts),episodes=episodes)
    result=dict(**payload,plan_sha256=repeat.canonical_hash(payload),created_at=p.now(),
                original_samples=sum(e["source_samples"] for e in episodes),
                augmented_samples=sum(e["augmented_samples"] for e in episodes),
                added_samples=sum(e["added_samples"] for e in episodes),
                total_anchors=sum(len(e["events"]) for e in episodes),validation_paths=validation,
                review_sources=base["excluded_sources"])
    preflight=p.read(SETUP/"cache-preflight.json")
    assert result["original_samples"]==preflight["train_base_samples"]
    assert result["augmented_samples"]==preflight["augmented_train_samples"]
    assert result["total_anchors"]==preflight["stationary_anchors"]+preflight["jitter_anchors"]
    p.write(AUG/"index.json",result)
    p.write(AUG/"planned.train.sample_files.json",[e["output_dir"] for e in episodes])
    p.write(AUG/"validation.sample_files.json",validation)
    return result

def finalize_augmentation():
    from mem_lite.datasets.m3_dataset import M3Dataset
    plan=p.read(AUG/"index.json")
    receipts=checked_receipts("augment",[e["episode_id"] for e in plan["episodes"]])
    assert all(p.sha(path)==digest for path,digest in plan["protected_hashes"].items())
    paths=[e["output_dir"] for e in plan["episodes"]]
    data=M3Dataset(paths);sample_count=len(data);del data
    assert sample_count==sum(r["samples"] for r in receipts)==plan["augmented_samples"]
    assert sum(r["anchors"] for r in receipts)==plan["total_anchors"]
    assert all(r["inserted_pt_fixed_to_anchor"] for r in receipts)
    assert p.read(AUG/"validation.sample_files.json")==plan["validation_paths"]
    p.write(AUG/"train.sample_files.json",paths)
    report=dict(status="completed_with_review_exclusion",completed_at=p.now(),episodes=1656,validation_episodes=417,
                original_samples=plan["original_samples"],augmented_samples=sample_count,
                added_samples=sum(r["added_samples"] for r in receipts),anchors=plan["total_anchors"],
                stationary_anchors=sum(r["stationary_anchors"] for r in receipts),
                jitter_anchors=sum(r["jitter_anchors"] for r in receipts),
                stationary_samples=sum(r["stationary_samples"] for r in receipts),
                jitter_samples=sum(r["jitter_samples"] for r in receipts),
                checked_histories=sum(r["checked_histories"] for r in receipts),
                history_bytes=sum(r["history_bytes"] for r in receipts),
                dense_pt_files_unchanged=True,original_observation_pt_unchanged=True,inserted_pt_fixed_to_anchor=True,
                views_synchronized=True,histories_causal=True,
                validation_unchanged=True,training_started=False,review_sources=plan["review_sources"])
    p.write(AUG/"report.json",report);return report

def phase_workers(stage):
    plan=get_cache_plan();children=[];logs=[]
    try:
        for worker in range(plan["workers"]):
            env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(worker%4),OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",
                       TOKENIZERS_PARALLELISM="false",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
                       PYTORCH_ALLOC_CONF="expandable_segments:True")
            log=(SETUP/("%s-worker-%d.log"%(stage,worker))).open("a");logs.append(log)
            children.append(subprocess.Popen([sys.executable,str(Path(__file__)),stage+"-worker","--worker",str(worker)],
                                             cwd=SETUP,env=env,stdout=log,stderr=subprocess.STDOUT))
        p.write(SETUP/(stage+"-pids.json"),[x.pid for x in children])
        while any(x.poll() is None for x in children):
            failed=[x for x in children if x.poll() not in (None,0)]
            if failed:raise RuntimeError(stage+" worker failed: "+str([(x.pid,x.returncode) for x in failed]))
            time.sleep(3)
        assert all(x.returncode==0 for x in children)
    except BaseException:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        raise
    finally:
        for log in logs:log.close()

def publish_preparation_report():
    import copy
    plan=p.get_plan();cache=get_cache_plan()
    report=p.read(p.EXPORTS/"report.json")
    backup=SETUP/"previous-split-export-report.json"
    if not backup.exists():p.write(backup,copy.deepcopy(report))
    cached=[];suite_cache=Counter();split_cache=Counter()
    for row in cache["sources"]:
        directory=BASE/"cache"/row["split"]/row["source_id"]
        summary=p.read(directory/"summary.json")
        cached.append(dict(source_id=row["source_id"],suite=row["suite"],sample_id=row["sample_id"],
                           old_split=row["split"],split=row["split"],cache_dir=str(directory),
                           num_samples=summary["num_samples"],
                           history_source_sha256=repeat.canonical_hash(summary["history_source"])))
        suite_cache[row["suite"]]+=1;split_cache[row["split"]]+=1
    excluded_ids={r["source_id"] for r in cache["excluded_sources"]}
    original_rows=[]
    for split in ("train","validation"):
        original_rows.extend(json.loads(l) for l in (p.EXPORTS/(split+".trajectories.jsonl")).read_text().splitlines())
    missing=[r for r in original_rows if r["source_id"] in excluded_ids]
    assert len(cached)==2073 and len(missing)==1
    for name,rows in [("cache-reusable.jsonl",cached),("cache-missing.jsonl",missing),
                      ("cache-excluded-pending-review.jsonl",cache["excluded_sources"])]:
        path=p.EXPORTS/name;tmp=path.with_name(path.name+".tmp")
        tmp.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows));tmp.replace(path)
    for suite,stats in report["suites"].items():
        stats.update(cache_reusable=suite_cache[suite],cache_missing=stats["total"]-suite_cache[suite],
                     pt_complete=stats["total"])
    report["cache_audit"]=dict(root=str(BASE),augmentation_root=str(AUG),status="completed_with_review_exclusion",
                              reusable=2073,missing=1,excluded_pending_review=cache["excluded_sources"],
                              available_cache_splits=dict(split_cache),training_ready=True,
                              previous_audit=str(backup),preparation_setup=str(SETUP))
    report["pt_preparation"]=p.read(BASE/"labels-report.json")
    report["verification"].update(cache_referenced_files_present=True,
                                  cache_checks_are_historical_before_deletion=False,
                                  cache_check_scope="2073 generated episodes; one explicitly excluded validation episode")
    report["updated_at"]=p.now()
    p.write(p.EXPORTS/"report.json",report)
    return dict(report=str(p.EXPORTS/"report.json"),base_cache_episodes=2073,
                training_episodes=1656,validation_episodes=417,review_episodes=1)

def pipeline():
    lock=(SETUP/"cache-controller.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(pid=os.getpid(),phase="waiting_for_all_pt",started_at=p.now(),updated_at=p.now())
    def update(**kwargs):
        state.update(**kwargs,updated_at=p.now());p.write(SETUP/"pipeline-status.json",state)
    update()
    try:
        while True:
            pt=p.read(SETUP/"pt-status.json")
            if pt["phase"]=="completed":break
            if pt["phase"]=="failed":raise RuntimeError("PT stage failed; caches will not start")
            if pt.get("pid"):os.kill(pt["pid"],0)
            time.sleep(10)
        label_gate();get_cache_plan()
        p.write(SETUP/"pt-to-cache-gate.json",dict(verified_at=p.now(),pt_report_sha256=p.sha(BASE/"labels-report.json"),
                                                full_pt_episodes=2074,base_cache_episodes=2073,review_excluded=1))
        update(phase="building_base_cache")
        phase_workers("base")
        update(phase="validating_base_cache")
        state["base_report"]=finalize_base()
        augment=make_augmentation_plan()
        update(phase="building_augmented_cache",augmented_samples=augment["augmented_samples"])
        phase_workers("augment")
        update(phase="validating_augmented_cache")
        state["augmentation_report"]=finalize_augmentation()
        state["published_report"]=publish_preparation_report()
        update(phase="completed",completed_at=p.now(),training_started=False)
        print(json.dumps(state),flush=True)
    except BaseException as exc:
        update(phase="failed",error=repr(exc));raise

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["plan","pipeline","base-worker","augment-worker","finalize-base","finalize-augment"])
    parser.add_argument("--worker",type=int,default=0);parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    if args.command=="plan":make_cache_plan()
    elif args.command=="pipeline":pipeline()
    elif args.command.endswith("-worker"):cache_worker(args.command[:-7],args.worker,args.limit)
    elif args.command=="finalize-base":print(json.dumps(finalize_base()))
    else:print(json.dumps(finalize_augmentation()))
