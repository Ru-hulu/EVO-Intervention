import json
from pathlib import Path
from fractions import Fraction
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
from PIL import Image

import prepare_repeats as a
from mem_lite.memory.stream import StreamMemory
from mem_lite.memory.vision import QwenFrameEncoder
from mem_lite.datasets.m3_dataset import M3Dataset


class FakeVisual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1,dtype=torch.bfloat16),requires_grad=False)
        self.config = SimpleNamespace(out_hidden_size=32, spatial_merge_size=2)

    def forward(self,pixels,grid_thw,return_dict=True):
        features = (torch.arange(64*32,dtype=torch.float32).reshape(64,32)*0.013+pixels.flatten()[0]).sin()
        return SimpleNamespace(pooler_output=features.to(self.weight.dtype))


class FakeProcessor:
    def __call__(self,images,**kwargs):
        return dict(pixel_values=torch.tensor([images[0].getpixel((0,0))[0]/255]),
                    image_grid_thw=torch.tensor([[1,16,16]]))


class RepeatTests(unittest.TestCase):
    def test_seed_and_uniform_sample_contract(self):
        rows=[dict(frame_index=6*i) for i in range(33)]
        first=a.choose_holds(rows,"task/episode")
        self.assertEqual(first,a.choose_holds(rows,"task/episode"))
        self.assertNotEqual(first,a.choose_holds(rows,"task/other"))
        self.assertEqual(len({h["source_sample_index"] for h in first}),5)
        self.assertTrue(all(10<=h["repeat_frames"]<=25 for h in first))

    def test_timeline_all_source_rates_and_boundary_holds(self):
        for fps in [Fraction(10),Fraction(20),Fraction(25),Fraction(30),Fraction(50,3)]:
            rows=[dict(frame_index=100+i*7,pt=p,images={}) for i,p in enumerate([0,.5,.2,1])]
            holds=[dict(source_sample_index=0,repeat_frames=10),
                   dict(source_sample_index=1,repeat_frames=25),
                   dict(source_sample_index=3,repeat_frames=11)]
            result=list(a.expand(rows,holds,fps))
            self.assertEqual(len(result),4+46)
            self.assertEqual([r["pt"] for r in result if not r["is_repeat"]],[0,.5,.2,1])
            for left,right in zip(result,result[1:]):
                self.assertGreater(right["frame_index"],left["frame_index"])
                if right["is_repeat"]:
                    self.assertAlmostEqual(right["timestamp_seconds"]-left["timestamp_seconds"],.2)
                    self.assertEqual(right["pt"],left["pt"])
                    self.assertEqual(right["source_frame_index"],left["source_frame_index"])
            self.assertAlmostEqual(result[-1]["timestamp_seconds"],rows[-1]["frame_index"]/float(fps)+46/5)

    def test_actual_memory_replay_multiview_missing_view_and_loader(self):
        torch.set_num_threads(2)
        for missing in [False,True]:
            with TemporaryDirectory(prefix="forest-repeat-test-") as d:
                base=Path(d); source=base/"source"; source.mkdir()
                output=base/"output"
                rows=[]
                for i,pt in enumerate([0,.2,.6,.3,.8,1]):
                    images={}
                    for v,view in enumerate(a.VIEWS):
                        if missing and v==2:
                            images[view]=None
                        else:
                            relative=f"{view}/{i:08d}.png"
                            (source/view).mkdir(exist_ok=True)
                            Image.new("RGB",(8,8),(i*20+v*3,10,30)).save(source/relative)
                            images[view]=relative
                    rows.append(dict(frame_index=10+i*4,images=images,history=None,pt=pt))
                (source/"samples.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
                config=dict(num_attention_heads=1,st_memory_windows=[1,2],st_memory_tokens=[64,16],
                            event_split_window=2,long_memory_tokens_per_frame=16,long_memory_tokens_quota=32,
                            sim_weight_g=.4,time_weight_a=.2,merge_weight_b=.4)
                settings=dict(memory_config=config,max_pixels=65536,min_pixels=4096,sample_fps=5)
                summary=dict(format_version="streamforest-m3-data-v1",episode_id="task/episode",
                             task_prompt="test",num_samples=6,history_dim=32,history_capacity=144,
                             history_source=settings,source=dict(fps=50/3,split="train",unit_id="task/episode"))
                a.write_json(source/"summary.json",summary)
                holds=[dict(source_sample_index=i,source_frame_index=rows[i]["frame_index"],repeat_frames=10)
                       for i in [0,1,2,4,5]]
                episode=dict(episode_id="task/episode",source_dir=str(source),output_dir=str(output),
                             source_samples_sha256=a.sha(source/"samples.jsonl"),
                             source_summary_sha256=a.sha(source/"summary.json"),
                             source_samples=6,augmented_samples=56,added_samples=50,holds=holds)
                plan=dict(plan_sha256="test-only")
                visual,processor=FakeVisual(),FakeProcessor()
                receipt=a.build_episode(plan,episode,visual,processor)
                self.assertEqual(receipt["anchors"],5)
                self.assertEqual(receipt["added_samples"],50)
                self.assertLessEqual(receipt["max_history_tokens"],144)
                generated=a.rows_at(output)
                stream=StreamMemory(QwenFrameEncoder(visual),processor,config)
                for row in generated:
                    with Image.open(output/row["images"]["cam_high"]) as im:
                        reference=stream.step(im.convert("RGB"),row["frame_index"],"episode","test")
                    if row["history"] is None:
                        self.assertIsNone(reference)
                    else:
                        cached=torch.load(output/row["history"],weights_only=True)
                        self.assertTrue(torch.equal(cached["tokens"],reference))
                dataset=M3Dataset([str(output)])
                self.assertEqual(len(dataset),56)
                for i in [0,1,10,11,55]:
                    sample=dataset[i]
                    self.assertEqual(sample["target_pt"],generated[i]["pt"])
                    self.assertEqual(len(sample["images"]),3)
                    if missing:
                        self.assertEqual(sample["images"][2].getpixel((0,0)),(0,0,0))
                self.assertEqual(a.build_episode(plan,episode,visual,processor)["samples"],56)


if __name__=="__main__":
    unittest.main(verbosity=2)

