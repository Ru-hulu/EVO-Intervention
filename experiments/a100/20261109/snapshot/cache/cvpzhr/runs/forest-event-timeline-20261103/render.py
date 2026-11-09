"""Render the causal event forest, not a final partition painted backwards."""
from bisect import bisect_right
import colorsys
import hashlib
import html
import json
from pathlib import Path
import subprocess

import av
from PIL import Image, ImageDraw, ImageFont, ImageOps

OUT = Path('/cache/cvpzhr/result/第一次实验/评估结果/事件动态合并-20261103')
STEM = '重复挤洗洁精-episode_001500-Forest事件动态合并'
FONT = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'


def color(identifier):
    index = int(identifier[1:]) + (41 if identifier[0] == 'M' else 0)
    rgb = colorsys.hsv_to_rgb((index * .61803398875) % 1, .65, .77)
    return tuple(round(c * 255) for c in rgb)


def main():
    trace = json.loads((OUT / 'trace.json').read_text())
    rows, nodes = trace['rows'], trace['nodes']
    frames = [r['frame'] for r in rows]
    fps, total = trace['source']['fps'], trace['source']['frames']
    duration = total / fps
    merge_rows = [r for r in rows if any(op['kind'] == 'merge' for op in r['operations'])]
    first_merge = merge_rows[0]['frame'] if merge_rows else None
    font = {size:ImageFont.truetype(FONT, size) for size in (16,18,20,23,26,32,42)}
    movie = OUT / (STEM + '.mp4')
    command = ['ffmpeg','-y','-hide_banner','-loglevel','error','-f','rawvideo','-pixel_format','rgb24',
        '-video_size','1280x930','-framerate',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast',
        '-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)]
    process = subprocess.Popen(command,stdin=subprocess.PIPE)
    latest_ops = []
    observed = -1
    written = 0
    previews = []
    with av.open(trace['source']['videos'][0]) as video:
        for frame_index, decoded in enumerate(video.decode(video=0)):
            index = bisect_right(frames, frame_index) - 1
            row = rows[index]
            if index != observed:
                observed = index
                if row['operations']:
                    latest_ops = [(row, op) for op in row['operations']][-2:]
            t = frame_index / fps
            canvas = Image.new('RGB',(1280,930),'#F8FAFC')
            draw = ImageDraw.Draw(canvas)
            def text(x,y,value,size=20,fill='#243449'):
                draw.text((x,y),str(value),font=font[size],fill=fill)
            text(40,20,'重复挤洗洁精 · Forest 事件记忆动态回放',32)
            text(40,62,'episode_001500  |  最终第 3 轮检查点  |  5 FPS 在线推理，视频 30 FPS 播放',18,'#65758A')
            picture = ImageOps.contain(decoded.to_image(),(800,450),Image.Resampling.LANCZOS)
            canvas.paste(picture,(40+(800-picture.width)//2,100+(450-picture.height)//2))
            text(880,110,'进度预测 / 真实 PT',20,'#65758A')
            text(880,147,f"{row['predicted_pt']*100:.1f}%",42,'#1E64AF')
            text(880,202,f"参考标签 {row['target_pt']*100:.1f}%",23)
            text(880,270,f"长期事件节点：{len(row['events_after'])}",23)
            text(880,309,f"长期 token：{row['long_tokens_after']} / 800",23)
            text(880,348,f"全部记忆 token：{row['total_tokens_after']} / 1008",20)
            text(880,389,f"本次预测使用历史：{row['history_tokens_used']} token",18,'#65758A')
            text(880,440,'颜色 = 当前记忆中的事件节点',18)
            draw.rectangle((880,478,900,496),fill='#ADB8C7');text(910,474,'灰色：仍在短期队列',18)
            draw.rectangle((880,510,900,528),fill='#E8EDF3');text(910,506,'浅色：尚未播放',18)
            left,right=40,1240
            x=lambda seconds:left+(right-left)*seconds/duration
            text(left,568,'播放位置',20)
            text(1000,568,f'{t:05.2f} / {duration:05.2f} 秒',20)
            draw.rectangle((left,605,right,613),fill='#E2E8F0')
            draw.rectangle((left,605,x(t),613),fill='#1E64AF')
            draw.ellipse((x(t)-5,604,x(t)+5,614),fill='#1E64AF')
            text(left,634,'事件时间线 · 当前帧写入后的记忆状态',20)
            draw.rectangle((left,670,right,719),fill='#E8EDF3')
            pending_time=frames[row['pending_start']]/fps
            draw.rectangle((x(pending_time),670,max(x(pending_time),x(t)),719),fill='#ADB8C7')
            for identifier in row['events_after']:
                node=nodes[identifier]
                start,end=frames[node['start']]/fps,frames[node['end']]/fps
                assert end<=row['time'] and start<end
                a,b=x(start),x(end)
                draw.rectangle((round(a),670,round(b)-1,719),fill=color(identifier))
                if b-a>=draw.textlength(identifier,font=font[16])+5:
                    width=draw.textlength(identifier,font=font[16])
                    text((a+b-width)/2,684,identifier,16,'#FFFFFF')
            draw.line((x(t),662,x(t),725),fill='#172033',width=2)
            for tick in [0,5,10,15]:
                text(x(tick)- (0 if tick==0 else 10),729,f'{tick}s',16,'#65758A')
            text(left,765,'最近的记忆维护操作',20)
            for j,(operation_row,op) in enumerate(latest_ops):
                node=nodes[op['node']]
                interval=f"{frames[node['start']]/fps:.2f}–{frames[node['end']]/fps:.2f}s"
                if op['kind']=='merge':
                    message=f"{operation_row['time']:.2f}s  合并 {' + '.join(op['parents'])} → {op['node']}  |  {op['tokens_before']} → {op['tokens_after']} token  |  覆盖 {interval}"
                else:
                    message=f"{operation_row['time']:.2f}s  新事件 {op['node']}  |  {op['tokens_after']} token  |  覆盖 {interval}"
                text(left,800+30*j,message,18)
            text(left,882,'时间线使用当时的事件分区；合并会重画历史色段。记忆更新从下一次进度预测开始生效。',18,'#65758A')
            if frame_index in {0,first_merge-1 if first_merge is not None else 0,first_merge,total-1}:
                name=f'preview-frame-{frame_index:04d}.png';canvas.save(OUT/name);previews.append(name)
            process.stdin.write(canvas.tobytes())
            written+=1
    process.stdin.close()
    assert process.wait()==0 and written==total
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=nb_read_frames,width,height,avg_frame_rate','-of','json',str(movie)],text=True))
    assert int(probe['streams'][0]['nb_read_frames'])==total
    actions=[dict(time=r['time'],**op) for r in rows for op in r['operations']]
    merge_actions=[a for a in actions if a['kind']=='merge']
    options=''.join(f'<option value="{a["time"]}">{a["time"]:.2f}s · {html.escape(" + ".join(a["parents"]))} → {a["node"]} · {a["tokens_before"]}→{a["tokens_after"]} token</option>' for a in merge_actions)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Forest 事件动态合并</title>
<style>body{margin:24px auto;max-width:1100px;padding:0 18px;font:16px system-ui;background:#f8fafc;color:#243449}video{width:100%;height:auto;background:#e8edf3}h1{font-size:24px}p{color:#65758a;line-height:1.6}nav{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:16px 0}button,select{padding:9px;font:inherit;border:1px solid #bdc8d6;border-radius:6px;background:white;color:#243449}select{max-width:100%}</style>
<h1>重复挤洗洁精 · 事件新增与合并</h1><p>视频下方两条等长时间线分别显示播放位置和当时的事件分区。合并时原有色段统一成新节点的颜色；灰色部分还没有形成事件。</p>
<video id="video" controls preload="metadata" src="__VIDEO__"></video>
<nav><button id="previous">上一次合并</button><button id="next">下一次合并</button><label>播放速度 <select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option></select></label><label>定位合并 <select id="event"><option value="">选择一次合并</option>__OPTIONS__</select></label></nav>
<p>使用原始 Forest 最终检查点重新进行 5 FPS 在线推理；没有添加重复帧。进度预测使用当前帧写入之前的历史，所以此刻的记忆更新影响下一次预测。</p>
<script>const video=document.getElementById('video');const times=__TIMES__;function jump(t){video.pause();video.currentTime=t+1/60;}document.getElementById('event').onchange=e=>{if(e.target.value!=='')jump(Number(e.target.value));};document.getElementById('speed').onchange=e=>video.playbackRate=Number(e.target.value);document.getElementById('next').onclick=()=>{const t=times.find(t=>t>video.currentTime+.025);if(t!==undefined)jump(t);};document.getElementById('previous').onclick=()=>{const t=times.filter(t=>t<video.currentTime-.025).at(-1);if(t!==undefined)jump(t);};</script></html>'''
    page=page.replace('__VIDEO__',html.escape(movie.name)).replace('__OPTIONS__',options).replace('__TIMES__',json.dumps(sorted({r['time'] for r in merge_rows})))
    (OUT/(STEM+'.html')).write_text(page)
    report=dict(status='completed',video=str(movie),player=str(OUT/(STEM+'.html')),trace=str(OUT/'trace.json'),
        episode=trace['episode'],checkpoint=trace['checkpoint'],fresh_online_inference=True,samples=len(rows),raw_frames=total,
        video_seconds=duration,new_events=trace['new_events'],merges=trace['merges'],final_nodes=len(rows[-1]['events_after']),
        first_merge_seconds=merge_rows[0]['time'] if merge_rows else None,mae_pp=trace['mae_pp'],rmse_pp=trace['rmse_pp'],
        original_manager_exact_match_all_frames=trace['original_manager_exact_match_all_frames'],previews=previews,
        timeline_is_causal=True,tail_flushed=False,encoded_frame_count_verified=True,source_unchanged=True,
        artifacts={p.name:dict(size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in OUT.iterdir() if p.suffix in ['.mp4','.html','.png']})
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
