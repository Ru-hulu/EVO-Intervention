"""Five scripted memory-error Recover demos in unmodified RMBench tasks.

Run from the upstream repository. Only the driver, recorder and passive scorer
are new. Native check_success runs after every physics step, as in evaluation.
"""
import importlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def pose(actor):
    p = actor.get_pose()
    return dict(p=p.p.copy(), q=p.q.copy())


def near(actor, target, xy=.03, z=.77):
    p = actor.get_pose().p
    return bool(np.all(np.abs(p[:2] - np.asarray(target)[:2]) < xy) and p[2] < z)


def angle(q, ref):
    q, ref = np.asarray(q), np.asarray(ref)
    return float(2 * np.arccos(np.clip(abs(np.dot(q, ref) / np.linalg.norm(q) / np.linalg.norm(ref)), 0, 1)))


class NativeDone(Exception):
    pass


class Demo:
    def __init__(self, task, seed, out):
        from scripts.collect_data import get_embodiment_config
        self.task, self.seed, self.out = task, seed, Path(out)
        self.out.mkdir(parents=True, exist_ok=False)
        cfg = yaml.safe_load(Path('env_cfg/task_config/demo_clean.yml').read_text())
        robot = yaml.safe_load(Path('env_cfg/task_config/_embodiment_config.yml').read_text())[cfg['embodiment'][0]]['file_path']
        rc = get_embodiment_config(robot)
        cfg.update(left_robot_file=robot, right_robot_file=robot, left_embodiment_config=rc,
                   right_embodiment_config=rc, dual_arm_embodied=True, task_name=task,
                   need_plan=True, save_data=False, save_freq=None, save_path=str(self.out),
                   eval_video_log=False, render_freq=0)
        cfg['data_type']['third_view'] = False
        random.seed(seed)
        self.e = getattr(importlib.import_module('envs.' + task), task)()
        self.e.setup_demo(seed=seed, now_ep_num=0, **cfg)
        self.rows, self.joints, self.events, self.writers = [], [], [], {}
        self.tick, self.native_success = 0, False
        self.phase = '初始观察：保存物体身份与初始位置'
        self.memory = {}
        self.initial = self.snapshot()
        self.initial['frame'] = 0
        self.e._take_picture = lambda: None
        driver = self
        scene = self.e.scene
        self.dt = float(scene.get_timestep())
        self.fps = 1 / (15 * self.dt)

        class SceneDriver:
            def __getattr__(self, name):
                return getattr(scene, name)

            def step(self):
                scene.step()
                driver.tick += 1
                driver.native_success = bool(driver.e.check_success())
                if driver.tick % 15 == 0 or driver.native_success:
                    driver.capture()
                if driver.native_success:
                    raise NativeDone()

        self.e.scene = SceneDriver()

    def snapshot(self):
        e = self.e
        result = {'poses': {}}
        for name in ['block1', 'block2', 'T_block1', 'T_block2', 'battery1', 'battery2']:
            if hasattr(e, name):
                result['poses'][name] = pose(getattr(e, name))
        for name in ['blocks_lst', 'mat_lst', 'baskets']:
            if hasattr(e, name):
                result['poses'][name] = [pose(a) for a in getattr(e, name)]
        for name in ['stage_id', 'press_cnt', 'press_flag', 'flag', 'selected_idx',
                     'correct_combination', 'target_pose1_p', 'target_pose2_p']:
            if hasattr(e, name):
                result[name] = getattr(e, name)
        if self.task == 'battery_try':
            result['combination'] = e.get_curr_combination()
            result['dashboard_on'] = e.check_dashboard_on()
        return result

    def atoms(self):
        e, t = self.e, self.task
        if t == 'rearrange_blocks':
            return [near(e.block1, e.final_occupy_mat.get_pose().p), e.stage_id >= 1,
                    near(e.block2, [0.13, e.block_y_lim])]
        if t == 'swap_blocks':
            h1 = e.baskets[e.with_block_basket_idx_1].get_pose().p
            h2 = e.baskets[e.with_block_basket_idx_2].get_pose().p
            def occupied(a, p):
                q = a.get_pose().p
                return bool(abs(q[0]-p[0]) < .04 and abs(q[1]-p[1]) < .06 and q[2] < .765)
            return [occupied(e.block1, h2), occupied(e.block2, h1), self.native_success]
        if t == 'swap_T':
            return [bool(np.linalg.norm(a.get_pose().p[:2]-p[:2]) < .025 and a.get_pose().p[2] < .77
                         and angle(a.get_pose().q, q) < np.deg2rad(15))
                    for a, p, q in [(e.T_block1, e.target_pose2, e.verify_T_block1_q),
                                    (e.T_block2, e.target_pose1, e.verify_T_block2_q)]]
        if t == 'battery_try':
            return [bool(e.check_battery_in_slot(a, p) and e.get_battery_state(a) == v)
                    for a, p, v in zip([e.battery1,e.battery2], [e.target_pose1_p,e.target_pose2_p], e.correct_combination)]
        return [bool(e.flag[i] == 'True' and near(a, e.mat_lst[e.selected_idx[i]].get_pose().p))
                for i, a in enumerate(e.blocks_lst)]

    def capture(self):
        obs = self.e.get_obs()
        for view, camera in [('main_view','head_camera'),('left_view','left_camera'),('right_view','right_camera')]:
            rgb = np.ascontiguousarray(obs['observation'][camera]['rgb'])
            if view not in self.writers:
                h, w = rgb.shape[:2]
                self.writers[view] = subprocess.Popen(['ffmpeg','-v','error','-nostdin','-f','rawvideo',
                    '-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(self.fps),'-i','-',
                    '-c:v','libx264','-threads','1','-preset','fast','-crf','20','-pix_fmt','yuv420p',
                    '-movflags','+faststart',str(self.out/(view+'.mp4'))], stdin=subprocess.PIPE)
            self.writers[view].stdin.write(rgb.tobytes())
        row = self.snapshot()
        row.update(frame=len(self.rows), physics_step=self.tick, simulation_time=self.tick*self.dt,
                   phase=self.phase, atoms=self.atoms(), native_success=self.native_success)
        row['progress'] = sum(row['atoms']) / len(row['atoms'])
        self.rows.append(row)
        self.joints.append(obs['joint_action']['vector'])

    def mark(self, phase, kind='phase', **evidence):
        self.phase = phase
        event = dict(frame=len(self.rows), physics_step=self.tick, kind=kind, description=phase, **evidence)
        self.events.append(event)
        print(json.dumps(event, ensure_ascii=False, default=serial), flush=True)

    def hold(self, seconds=.5):
        for _ in range(round(seconds/self.dt)):
            self.e.scene.step()

    def move(self, *actions):
        assert self.e.move(*actions), 'Motion planning failed'

    def transfer(self, actor, arm, target, **place_kwargs):
        self.move(self.e.grasp_actor(actor, arm_tag=arm, pre_grasp_dis=.08, grasp_dis=.02))
        self.move(self.e.move_by_displacement(arm_tag=arm, z=.10))
        self.move(self.e.place_actor(actor, arm_tag=arm, target_pose=np.asarray(target).copy(),
                                    functional_point_id=2, dis=.01, **place_kwargs))
        self.move(self.e.back_to_origin(arm_tag=arm))
        self.hold()

    def press(self):
        e = self.e
        self.move(e.grasp_actor(e.button, arm_tag='left', pre_grasp_dis=.08, grasp_dis=.08, contact_point_id=0))
        self.move(e.move_by_displacement(arm_tag='left', z=-.04))
        self.move(e.move_by_displacement(arm_tag='left', z=.04))
        self.move(e.back_to_origin(arm_tag='left'))
        self.hold()

    def rearrange(self):
        e = self.e
        self.memory = {'A_target':e.target_pose1.p.copy(), 'middle':e.target_pose2.p.copy(), 'original_mat_object':'block2'}
        self.mark('正常准备：将最初位于中间的 A 放到空垫子')
        self.transfer(e.block1, 'right', self.memory['A_target'])
        self.mark('合法按一次按钮，保留 A 的正确摆放')
        self.press()
        assert e.stage_id == 1 and e.press_cnt == 1 and self.atoms() == [True, True, False]
        self.mark('错误记忆：将 A 误认成原先在垫子上的 B；把 A 搬到中间', 'memory_error',
                  truth='original_mat_object=block2', corrupted='original_mat_object=block1', source_frames=[0])
        self.transfer(e.block1, 'right', self.memory['middle'])
        assert self.atoms() == [False, True, False]
        self.mark('重新读取初始身份记录：将 A 放回它已完成的目标垫子', 'memory_retrieval', source_frames=[0])
        self.transfer(e.block1, 'right', self.memory['A_target'])
        assert self.atoms() == [True, True, False]
        self.mark('恢复完成：移动真正的 B 到中间，不重复按按钮', 'restored')
        self.transfer(e.block2, 'right', self.memory['middle'])

    def swap(self):
        e = self.e
        h1,h2,tmp = [e.baskets[i].get_functional_point(0).copy() for i in
                     [e.with_block_basket_idx_1,e.with_block_basket_idx_2,e.empty_basket_idx]]
        self.memory = dict(A_home=h1, B_home=h2, temporary=tmp)
        def transfer(a, start, end):
            arm = 'right' if max(start[0],end[0]) > .12 else 'left'
            self.transfer(a,arm,end)
        self.mark('正常准备：A 移到临时托盘；B 完成交换到 A 原位')
        transfer(e.block1,h1,tmp)
        transfer(e.block2,h2,h1)
        assert self.atoms() == [False, True, False]
        self.mark('错误记忆：混淆 B 的历史目标，把已到位 B 搬回 B 原位', 'memory_error',
                  truth=h1, corrupted=h2, source_frames=[0])
        transfer(e.block2,h1,h2)
        assert self.atoms() == [False,False,False] and e.flag
        self.mark('读取初始位置及搬运记录：B 应在 A 原位', 'memory_retrieval', source_frames=[0])
        transfer(e.block2,h2,h1)
        assert self.atoms() == [False,True,False]
        self.mark('B 的交换成果恢复；继续把 A 放到 B 原位', 'restored')
        transfer(e.block1,tmp,h2)
        assert self.atoms() == [True,True,False] and e.flag
        self.mark('两块均到目标，执行唯一一次最终确认')
        self.press()

    def swap_t(self):
        e = self.e
        assert angle(e.verify_T_block1_q,e.verify_T_block2_q) > np.deg2rad(30), 'Seed has insufficient angle separation'
        red_target, blue_target = e.target_pose2.copy(), e.target_pose1.copy()
        self.memory = dict(red_target=red_target, blue_target=blue_target)
        self.mark('双臂拿起 T 块，将红 T 放到蓝 T 的初始位姿')
        self.move(e.grasp_actor(e.T_block1,arm_tag='left',pre_grasp_dis=.05,gripper_pos=.2),
                  e.grasp_actor(e.T_block2,arm_tag='right',pre_grasp_dis=.05,gripper_pos=.2))
        self.move(e.move_by_displacement(arm_tag='left',z=.1),e.move_by_displacement(arm_tag='right',z=.1))
        self.move(e.back_to_origin(arm_tag='left'),e.back_to_origin(arm_tag='right'))
        self.move(e.place_actor(e.T_block1,arm_tag='left',target_pose=red_target,pre_dis=.05,constrain='align'))
        self.move(e.back_to_origin(arm_tag='left'))
        self.hold()
        assert self.atoms() == [True,False]
        wrong = red_target.copy(); wrong[3:] = blue_target[3:]
        self.mark('错误检索：把红 T 初始朝向误当成蓝 T 初始朝向，重摆已正确的红 T', 'memory_error',
                  truth=red_target, corrupted=wrong, source_frames=[0])
        def reposition(target):
            # The red block is now on the right: free the right arm by parking blue
            # only if required. This version uses the original left-arm reach.
            self.move(e.grasp_actor(e.T_block1,arm_tag='left',pre_grasp_dis=.05,gripper_pos=.2))
            self.move(e.move_by_displacement(arm_tag='left',z=.1))
            self.move(e.place_actor(e.T_block1,arm_tag='left',target_pose=target,pre_dis=.05,constrain='align'))
            self.move(e.back_to_origin(arm_tag='left')); self.hold()
        reposition(wrong)
        assert self.atoms() == [False,False]
        self.mark('重新读取初始画面的蓝 T 朝向，恢复红 T 位姿', 'memory_retrieval', source_frames=[0])
        reposition(red_target)
        assert self.atoms() == [True,False]
        self.mark('红 T 成果恢复；完成蓝 T 的交换', 'restored')
        self.move(e.place_actor(e.T_block2,arm_tag='right',target_pose=blue_target,pre_dis=.05,constrain='align'))
        self.move(e.back_to_origin(arm_tag='right'))

    def battery(self):
        e = self.e
        assert e.correct_combination == [1,1], 'Seed must use native target 11'
        self.memory = {'attempts':[]}
        def put(actor,arm,target,orientation):
            q = getattr(e,'quat_of_target_pose_'+str(orientation))
            self.move(e.grasp_actor(actor,arm_tag=arm,pre_grasp_dis=.1,grasp_dis=.03))
            self.move(e.move_by_displacement(arm_tag=arm,z=.08))
            self.move(e.place_actor(actor,arm_tag=arm,target_pose=list(target)+list(q),functional_point_id=0,constrain='align'))
            self.move(e.back_to_origin(arm_tag=arm));self.hold()
        def attempt():
            record=dict(combination=e.get_curr_combination(),dashboard_on=e.check_dashboard_on(),frame=len(self.rows)-1)
            assert not record['dashboard_on']
            self.memory['attempts'].append(record)
            self.mark('记录实际失败组合 '+str(record['combination']), 'feedback', attempt=record)
        self.mark('试错历史：安装 00，观察仪表未亮')
        put(e.battery1,'left',e.target_pose1_p,0);put(e.battery2,'right',e.target_pose2_p,0);attempt()
        self.mark('试错历史：尝试 10，记录未成功')
        put(e.battery1,'left',e.target_pose1_p,1);attempt()
        self.mark('试错历史：先回到 00，再尝试 01，避免提前经过成功组合')
        put(e.battery1,'left',e.target_pose1_p,0)
        put(e.battery2,'right',e.target_pose2_p,1);attempt()
        seen={tuple(r['combination']) for r in self.memory['attempts']}
        remaining=set([(0,0),(0,1),(1,0),(1,1)])-seen
        assert remaining == {(1,1)} and self.atoms() == [False,True]
        self.mark('错误检索：误把已失败的 00 当成下一候选，翻坏已正确的右电池', 'memory_error',
                  truth=list(remaining)[0],corrupted=[0,0],source_frames=[r['frame'] for r in self.memory['attempts']])
        put(e.battery2,'right',e.target_pose2_p,0)
        assert self.atoms() == [False,False]
        self.mark('重读三次失败记录，排除 00、10、01，恢复右电池方向', 'memory_retrieval',
                  source_frames=[r['frame'] for r in self.memory['attempts']])
        inferred=list(remaining)[0]
        put(e.battery2,'right',e.target_pose2_p,inferred[1])
        assert self.atoms() == [False,True]
        self.mark('右电池成果恢复；改变左电池，完成 11', 'restored')
        put(e.battery1,'left',e.target_pose1_p,inferred[0])

    def multi_home(self):
        e = self.e
        homes=[e.mat_lst[i].get_pose().p.copy() for i in e.selected_idx]
        self.memory=dict(homes=homes)
        self.mark('正常完成第一个方块：访问绿垫，再归还初始蓝垫')
        e.place_back_block(0); self.hold()
        assert self.atoms() == [True,False,False]
        arm=e.choose_arm(int(e.selected_idx[0]))
        free=[i for i in range(6) if i not in e.selected_idx and e.choose_arm(i)==arm]
        assert free, 'No empty wrong mat in same arm workspace'
        wrong=e.mat_lst[free[0]].get_pose().p.copy()
        self.mark('错误记忆：误记第一个方块原位，搬走已经正确归位的方块', 'memory_error',
                  truth=homes[0],corrupted=wrong,source_frames=[0])
        self.transfer(e.blocks_lst[0],arm,wrong)
        assert self.atoms() == [False,False,False]
        self.mark('读取初始方块—蓝垫对应，恢复第一个方块原位', 'memory_retrieval',source_frames=[0])
        self.transfer(e.blocks_lst[0],arm,self.memory['homes'][0])
        assert self.atoms() == [True,False,False]
        self.mark('第一个归位成果恢复；完成其余两个方块的绿垫访问与归还', 'restored')
        e.place_back_block(1);e.place_back_block(2)

    def run(self):
        error = None
        try:
            self.capture();self.hold()
            {'rearrange_blocks':self.rearrange,'swap_blocks':self.swap,'swap_T':self.swap_t,
             'battery_try':self.battery,'place_block_mat':self.multi_home}[self.task]()
            raise AssertionError('Script ended without native success')
        except NativeDone:
            assert any(x['kind']=='restored' for x in self.events), 'Native success before recovery completed'
        except Exception as exc:
            error = repr(exc)
            raise
        finally:
            for w in self.writers.values():w.stdin.close()
            for w in self.writers.values():w.wait()
            np.savez_compressed(self.out/'joints.npz',joint_action=self.joints)
            (self.out/'states.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,default=serial)+'\n' for x in self.rows))
            result=dict(task=self.task,seed=self.seed,fps=self.fps,frames=len(self.rows),native_success=self.native_success,
                        plan_success=bool(self.e.plan_success),error=error,initial=self.initial,memory=self.memory,events=self.events,
                        source_commit='7d1cdfc7661c273a7a5c5f6126e2d1a7f0a3f4b7',
                        controller='scripted memory retrieval corruption and scripted history-based repair; not autonomous VLA',
                        checker='unmodified native check_success after every physics step; immediate stop at first success')
            (self.out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=serial))
            for name in ['left_proc','right_proc']:
                p=getattr(self.e.robot,name,None)
                if p is not None:p.terminate();p.join(3)


if __name__ == '__main__':
    Demo(sys.argv[1],int(sys.argv[2]),sys.argv[3]).run()
