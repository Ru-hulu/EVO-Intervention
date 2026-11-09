import json,math,sys,unittest
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import prepare_pt
import prepare_static_jitter as a
from test_repeats import FakeVisual,FakeProcessor
import torch
from PIL import Image
from mem_lite.memory.stream import StreamMemory
from mem_lite.memory.vision import QwenFrameEncoder
from mem_lite.datasets.m3_dataset import M3Dataset

class StaticJitterTests(unittest.TestCase):
    def test_user_example_has_eight_independent_fixed_pt_samples(self):
        rows=[dict(frame_index=i+1,pt=i/9,images={"cam_high":str(i+1)}) for i in range(10)]
        event=dict(kind="jitter",anchor=8,frames=4,requested_frames=4)
        out=list(a.expand(rows,[event],Fraction(5)))
        self.assertEqual([r["source_frame_index"] for r in out],list(range(1,10))+[8,7,6,5,6,7,8,9]+[10])
        self.assertEqual(len(out),18)
        self.assertEqual([r["pt"] for r in out[9:17]],[rows[8]["pt"]]*8)
        self.assertEqual(out[-1]["pt"],rows[9]["pt"])
        self.assertEqual([r["sample_index"] for r in out],list(range(18)))
        for before,after in zip(out,out[1:]):
            self.assertAlmostEqual(after["timestamp_seconds"]-before["timestamp_seconds"],.2)
        self.assertNotEqual(out[10]["pt"],rows[out[10]["source_sample_index"]]["pt"])

    def test_policy_sampling_and_zero_duration(self):
        rows=[dict(frame_index=i*5) for i in range(100)]
        for index in range(30):
            events=a.choose_events(rows,str(index),Fraction(25),496)
            self.assertEqual(events,a.choose_events(rows,str(index),Fraction(25),496))
            stationary=[e for e in events if e["kind"]=="stationary"]
            jitter=[e for e in events if e["kind"]=="jitter"]
            self.assertTrue(1<=len(stationary)<=3)
            self.assertEqual(len(jitter),2)
            self.assertEqual(len({e["anchor"] for e in stationary}),len(stationary))
            self.assertEqual(len({e["anchor"] for e in jitter}),len(jitter))
            self.assertTrue(all(0<=e["frames"]<=15 for e in stationary))
            self.assertTrue(all(e["frames"]==min(e["requested_frames"],e["anchor"]) and 0<=e["frames"]<=5 for e in jitter))
        originals=[dict(frame_index=i,pt=i/2,images={}) for i in range(3)]
        zero=[dict(kind="stationary",anchor=0,frames=0),dict(kind="jitter",anchor=0,frames=0)]
        self.assertEqual(len(list(a.expand(originals,zero,Fraction(5)))),3)

    def test_multiview_missing_view_fixed_pt_and_causal_memory(self):
        torch.set_num_threads(2)
        for missing in (False,True):
            with TemporaryDirectory(prefix="forest-static-jitter-test-") as d:
                base=Path(d);source=base/"source";source.mkdir();output=base/"output";rows=[]
                for i,pt in enumerate([0,.2,.6,.3,.8,1]):
                    images={}
                    for j,view in enumerate(a.VIEWS):
                        if missing and j==2:images[view]=None;continue
                        relative=f"{view}/{i:08d}.png"
                        (source/view).mkdir(exist_ok=True)
                        Image.new("RGB",(8,8),(i*20+j*3,10,30)).save(source/relative)
                        images[view]=relative
                    rows.append(dict(frame_index=10+i*4,pt=pt,images=images,history=None))
                (source/"samples.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
                config=dict(num_attention_heads=1,st_memory_windows=[1,2],st_memory_tokens=[64,16],
                            event_split_window=2,long_memory_tokens_per_frame=16,long_memory_tokens_quota=32,
                            sim_weight_g=.4,time_weight_a=.2,merge_weight_b=.4)
                settings=dict(memory_config=config,max_pixels=65536,min_pixels=4096,sample_fps=5)
                summary=dict(format_version="streamforest-m3-data-v1",episode_id="task/episode",task_prompt="test",
                             num_samples=6,history_dim=32,history_capacity=144,history_source=settings,
                             source=dict(fps=50/3,frames=31,split="train",unit_id="task/episode"))
                a.write_json(source/"summary.json",summary)
                events=[dict(kind="stationary",anchor=0,frames=0,requested_frames=0),
                        dict(kind="stationary",anchor=3,frames=3,requested_frames=3),
                        dict(kind="jitter",anchor=3,frames=2,requested_frames=2)]
                episode=dict(episode_id="task/episode",source_dir=str(source),output_dir=str(output),
                    source_samples_sha256=a.sha(source/"samples.jsonl"),source_summary_sha256=a.sha(source/"summary.json"),
                    source_samples=6,augmented_samples=13,added_samples=7,events=events)
                plan=dict(plan_sha256="test-only")
                visual,processor=FakeVisual(),FakeProcessor()
                receipt=a.build_episode(plan,episode,visual,processor)
                self.assertTrue(receipt["inserted_pt_fixed_to_anchor"])
                self.assertEqual(receipt["stationary_samples"],3)
                self.assertEqual(receipt["jitter_samples"],4)
                generated=a.rows_at(output);stream=StreamMemory(QwenFrameEncoder(visual),processor,config)
                for row in generated:
                    if row["kind"]!="original":self.assertEqual(row["pt"],.3)
                    with Image.open(output/row["images"]["cam_high"]) as im:
                        reference=stream.step(im.convert("RGB"),row["frame_index"],"episode","test")
                    if row["history"] is None:self.assertIsNone(reference)
                    else:self.assertTrue(torch.equal(torch.load(output/row["history"],weights_only=True)["tokens"],reference))
                dataset=M3Dataset([str(output)])
                self.assertEqual(len(dataset),13)
                for i in [0,4,7,10,12]:
                    sample=dataset[i];self.assertEqual(sample["target_pt"],generated[i]["pt"])
                    self.assertEqual(len(sample["images"]),3)
                    if missing:self.assertEqual(sample["images"][2].getpixel((0,0)),(0,0,0))
                self.assertEqual(a.build_episode(plan,episode,visual,processor)["samples"],13)

if __name__=="__main__":unittest.main(verbosity=2)
