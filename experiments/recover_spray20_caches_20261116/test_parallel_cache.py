import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


class ParallelCache(unittest.TestCase):
    def test_four_shards_partition_twenty_episodes(self):
        records=list(range(20)); shards=[records[i::4] for i in range(4)]
        self.assertEqual(sorted(sum(shards,[])),records)
        self.assertTrue(all(len(s)==5 for s in shards))

    def test_linked_images_preserve_pixels_memory_and_rollback(self):
        import torch
        from PIL import Image
        from mem_lite.datasets.prepare_stream import build_episode
        from mem_lite.datasets.m3_dataset import VIEW_ORDER
        class Stream:
            encoder=SimpleNamespace(hidden_size=3)
            memory=SimpleNamespace(token_budget=4)
            def bind_history_source(self,source):return source
            def reset(self):self.previous=None
            def step(self,image,frame_index,episode_id,prompt):
                result=self.previous
                self.previous=torch.full((2,3),float(frame_index))
                return result
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); samples=[]; sources={}
            for i,pt in enumerate([0.,.6,.3]):
                images=[];sources[i*6]={}
                for view in VIEW_ORDER:
                    image=Image.new('RGB',(32,24),(i*50,20,80))
                    path=root/f'{i}-{view}.png';image.save(path)
                    sources[i*6][view]=path;images.append(image)
                samples.append(dict(frame_index=i*6,pt=pt,images=images))
            def writer(image,path,frame,view):os.link(sources[frame][view],path)
            for name,callback in [('encoded',None),('linked',writer)]:
                build_episode(samples,Stream(),root/name,'goal','episode',{'same':True},image_writer=callback)
            self.assertEqual((root/'encoded/samples.jsonl').read_bytes(),(root/'linked/samples.jsonl').read_bytes())
            self.assertEqual((root/'encoded/summary.json').read_bytes(),(root/'linked/summary.json').read_bytes())
            rows=[json.loads(line) for line in (root/'linked/samples.jsonl').read_text().splitlines()]
            for row in rows:
                for view,relative in row['images'].items():
                    self.assertEqual((root/'encoded'/relative).read_bytes(),(root/'linked'/relative).read_bytes())
                    self.assertTrue(os.path.samefile(root/'linked'/relative,sources[row['frame_index']][view]))
                if row['history']:
                    a=torch.load(root/'encoded'/row['history'],weights_only=True)
                    b=torch.load(root/'linked'/row['history'],weights_only=True)
                    self.assertEqual(a['through_frame'],b['through_frame'])
                    self.assertTrue(torch.equal(a['tokens'],b['tokens']))


if __name__=='__main__':unittest.main()
