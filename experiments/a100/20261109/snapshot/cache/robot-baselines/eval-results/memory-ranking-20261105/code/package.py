"""Validate recorded physical regression and package one Recover per task."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

TASKS = {
    'rearrange_blocks': ('01', '重新排列：历史身份混淆', '将中间方块移到空垫，按一次按钮，再将原先在垫子上的方块移到中间。', ['A 在目标垫','合法确认已完成','B 在中间']),
    'swap_blocks': ('02', '交换方块：撤销已完成的交换', '借助空托盘交换两个方块的初始位置，最后按一次按钮确认。', ['A 在 B 原位','B 在 A 原位','最终确认']),
    'swap_T': ('03', '交换 T 块：历史朝向混淆', '交换两个 T 块的位置和朝向，匹配对方的初始位姿。', ['红 T 位姿正确','蓝 T 位姿正确']),
    'battery_try': ('04', '电池试错：重复旧候选并翻坏正确电池', '尝试电池方向组合，直到两节电池正确安装并使仪表指示成功。', ['左电池原生槽位/方向且已降低至槽位高度','右电池原生槽位/方向且已降低至槽位高度']),
    'place_block_mat': ('05', '多物体归位：误记已归位方块的原位', '三个方块分别访问绿垫，再放回各自初始蓝垫。', ['方块 A 已访问绿垫且在原位','方块 B 已访问绿垫且在原位','方块 C 已访问绿垫且在原位']),
    'blocks_ranking_try': ('06', '尝试积木排列', '调整三个积木的左右排列，每次按按钮验证，直到找到正确排列。',
                           [f'已记录第 {i} 个不同排列的失败反馈' for i in range(1, 6)] + ['正确排列保持成立', '最终按键验证通过']),
}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def package(raw, dest):
    if (dest/'boundaries').exists():
        raise ValueError('Boundary-labelled dataset already exists; package raw data into a new directory.')
    dest.mkdir(parents=True, exist_ok=True)
    rows, validations = [], []
    for task, (number, name, prompt, atom_names) in TASKS.items():
        candidates = []
        for file in raw.glob(task+'*/result.json'):
            result = json.loads(file.read_text())
            if result['native_success'] and result['error'] is None and (task != 'battery_try' or result.get('recording_tail')):
                candidates.append((file, result))
        if not candidates:
            continue
        file, result = sorted(candidates)[-1]
        source = file.parent
        states = [json.loads(x) for x in (source/'states.jsonl').read_text().splitlines()]
        n = len(states)
        assert n == result['frames'] and result['plan_success'] and states[-1]['native_success']
        first_native_frame = next(i for i,s in enumerate(states) if s['native_success'])
        if task != 'battery_try':
            assert first_native_frame == n-1, 'Early native success'
        else:
            assert result.get('recording_tail'), 'Battery requires explicit native/physical distinction'
        error = next(x for x in result['events'] if x['kind']=='memory_error')
        retrieval = next(x for x in result['events'] if x['kind']=='memory_retrieval')
        restored = next(x for x in result['events'] if x['kind']=='restored')
        assert restored['frame'] < first_native_frame, 'Recovery must finish before native success'
        before = states[error['frame']-1]
        damaged = states[retrieval['frame']-1]
        after = states[restored['frame']-1]
        lost = [i for i,(a,b) in enumerate(zip(before['atoms'],damaged['atoms'])) if a and not b]
        assert lost and all(after['atoms'][i] for i in lost)
        assert damaged['progress'] < before['progress'] <= after['progress']
        assert states[-1]['progress'] == 1
        loss_frame = next(i for i in range(error['frame'],retrieval['frame'])
                          if any(not states[i]['atoms'][j] for j in lost))
        restore_frame = next(i for i in range(retrieval['frame'],restored['frame'])
                             if all(states[i]['atoms'][j] for j in lost))
        key = number+'-'+task+'-recover'
        folder = dest/key
        folder.mkdir(exist_ok=True)
        views = {}
        media = []
        for view in ['main_view','left_view','right_view']:
            video = source/(view+'.mp4')
            probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                      '-show_entries','stream=nb_read_frames,avg_frame_rate,width,height,codec_name','-of','json',str(video)]))['streams'][0]
            assert int(probe['nb_read_frames']) == n
            a,b=map(int,probe['avg_frame_rate'].split('/'))
            assert abs(a/b-result['fps']) < 1e-5
            shutil.copy2(video,folder/video.name)
            views[view] = dict(video_path=key+'/'+video.name)
            media.append(dict(view=view,**probe))
        for filename in ['result.json','states.jsonl','joints.npz']:
            shutil.copy2(source/filename,folder/filename)
        dense = [s['progress'] for s in states]
        physical_events=[]
        for i,s in enumerate(states):
            if i==0 or s['atoms'] != states[i-1]['atoms']:
                physical_events.append(dict(frame=i,progress=s['progress'],atoms=s['atoms'],
                    kind='regression' if i and s['progress']<states[i-1]['progress'] else 'complete',
                    description=s['phase']))
        nav=[dict(frame=loss_frame,kind='regression',progress=states[loss_frame]['progress'],step='成果真实失效'),
             dict(frame=retrieval['frame'],kind='retrieval',step='读取历史纠错'),
             dict(frame=restore_frame,kind='restored',step='同一成果恢复'),
             dict(frame=n-1,kind='success',step='落座收尾完成（原生已成功）' if task=='battery_try' else '原生成功')]
        progress=dict(policy='memory-retained-goals-v1',total=len(atom_names),atom_names=atom_names,
            dense_progress=dense,points=[[x['frame'],sum(x['atoms'])] for x in physical_events],
            source=('物理成果计分；电池另要求 z<0.79；原生先判成功，最后放置收尾单独记录' if task=='battery_try' else '按当前物理成果计算；无时间插值；原生成功规则未修改'),events=nav,
            physical_events=physical_events,first_issue=nav[0],recoverable=True,frozen_from=None)
        boundaries=sorted(set([0,n]+[x['frame'] for x in result['events']]+[x['frame'] for x in physical_events]))
        stages=[]
        for a,b in zip(boundaries,boundaries[1:]):
            if a>=n or b<=a:continue
            s=states[a]
            detail='；'.join(f'{label}：'+('成立' if val else '未成立') for label,val in zip(atom_names,s['atoms']))
            stages.append(dict(stage_id=len(stages),start_frame=a,end_frame=b,
                               state_description=f"{s['phase']}。P={s['progress']:.3f}。{detail}"))
        annotation=dict(fps=result['fps'],total_frames=n,frame_index_base=0,frame_range_convention='[start_frame, end_frame)',
                        stages=stages,num_stages=len(stages),annotation_status='physics_verified_scripted_demo',progress=progress)
        dump(folder/'annotation.json',annotation)
        dump(dest/'progress'/(key+'.json'),progress)
        validation=dict(sample_id=key,task=task,seed=result['seed'],frames=n,fps=result['fps'],native_success=True,
                        same_atom_restored=[atom_names[i] for i in lost],before=before['progress'],damaged=damaged['progress'],
                        restored=after['progress'],loss_frame=loss_frame,restore_frame=restore_frame,success_frame=n-1,first_native_success_frame=first_native_frame,recording_tail=result.get('recording_tail'),
                        media=media,source_directory=str(source))
        dump(folder/'validation.json',validation)
        validations.append(validation)
        extra='仓库额外原生任务，不属于论文九个核心任务。' if task=='place_block_mat' else 'RMBench 论文核心原生任务。'
        if task=='battery_try':
            extra += ' 原生只检查槽位平面距离和方向；已记录首次原生成功，视频额外录完最后放置收尾，progress 另要求高度低于 0.79m。'
        rows.append(dict(sample_id=key,task_id=number,task_name=name,episode_index=result['seed'],
                         model_name='脚本化 memory Recover 示范',suite='State',trajectory_class='Recover',task_success=True,
                         failure_type='memory_error_recovered',failure_reason='记忆错误导致有效物理成果下降；同一成果恢复后原生成功',
                         fps=result['fps'],total_frames=n,video_path=views['main_view']['video_path'],video_views=views,
                         annotation_path=key+'/annotation.json',task_prompt=prompt+'\n'+extra+' 使用脚本化错误检索与历史恢复，不代表 VLA 自主纠错。',
                         source_commit=result['source_commit'],training_eligible=False,evidence_path=key+'/states.jsonl'))
    index=dest/'metadata.jsonl.tmp'
    index.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    index.replace(dest/'metadata.jsonl')
    dump(dest/'validation.json',dict(count=len(rows),episodes=validations))
    hashes={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file() and p.name!='sha256.json'}
    dump(dest/'sha256.json',hashes)
    print(json.dumps(validations,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('raw',type=Path);parser.add_argument('dest',type=Path)
    args=parser.parse_args();package(args.raw,args.dest)
