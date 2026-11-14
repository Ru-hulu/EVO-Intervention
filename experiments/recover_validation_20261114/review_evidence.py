"""Persist the independently viewed 5 FPS fall-boundary neighborhoods, before inference."""
import time
from .common import *

BOUNDARIES = {
    'inspect-spray-bottles_episode_000030': dict(onset=762,previous=756,
        observation='At 756 the upright bottle is still at the rack under the gripper; at 762 it visibly tilts/drops off its support; by 768 it is airborne in front of the rack.'),
    'inspect-spray-bottles_episode_000031': dict(onset=768,previous=762,
        observation='At 762 the bottle is upright at the rack; at 768 its base has moved down beyond the front support after release; at 774 it is visibly falling and rotating.')}


def main():
    import av
    from PIL import Image, ImageDraw
    allrows=read(ROOT/'selection.json')['records']
    for row in allrows:
        if row['sample_id'] not in BOUNDARIES:continue
        value=BOUNDARIES[row['sample_id']];folder=ROOT/'event-review'/row['sample_id'];folder.mkdir(parents=True,exist_ok=True)
        assert sha(row['videos'][0])==row['video_hashes'][0]
        wanted=list(range(value['onset']-24,value['onset']+25,6));images=[];hashes={}
        with av.open(row['videos'][0]) as v:
            for i,frame in enumerate(v.decode(video=0)):
                if i in wanted:
                    im=frame.to_image().convert('RGB');p=folder/f'{i:08d}.png';im.save(p);hashes[str(p)]=sha(p);images.append((i,im))
                if i>max(wanted):break
        assert [i for i,im in images]==wanted
        sheet=Image.new('RGB',(1800,720),'white');draw=ImageDraw.Draw(sheet)
        for k,(i,im) in enumerate(images):
            w,h=im.size;im=im.crop((int(w*.24),int(h*.43),int(w*.52),int(h*.95)));im.thumbnail((360,330))
            x=k%5*360;y=k//5*360;sheet.paste(im,(x,y+25));draw.text((x+5,y+4),f'frame {i} / t={i/30:.1f}s',fill='black')
        p=folder/'boundary-contact-sheet.jpg';sheet.save(p,quality=92);hashes[str(p)]=sha(p)
        proof=dict(schema='recover-physical-fall-review-v1',key=row['key'],video_sha256=row['video_hashes'][0],
            stage_annotation_sha256=row['annotation_sha256'],failure_onset_frame=value['onset'],
            preceding_sample_frame=value['previous'],observation=value['observation'],
            definition='first sampled frame with visible uncontrolled displacement off rack support, after release',
            sampling_fps=5,temporal_uncertainty_seconds=.2,human_verified=False,
            reviewer='assistant visual review of original-source 5 FPS contact sheets and enlarged crops',
            predictions_consulted=False,source_frame_sha256=hashes,reviewed=time.time())
        save(folder/'review.json',proof)


if __name__=='__main__':main()
