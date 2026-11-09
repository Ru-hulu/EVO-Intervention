"""One native blocks_ranking_try Recover, inferred from five failed trials."""
import itertools
import sys

from collect import Demo, pose


class RankingDemo(Demo):
    def layout(self):
        return sorted(['block1', 'block2', 'block3'], key=lambda k: getattr(self.e, k).get_pose().p[0])

    def geometry(self):
        positions = [getattr(self.e, k).get_pose().p for k in self.layout()]
        return bool(all(p[2] < .77 for p in positions)
            and all(0 < b[0] - a[0] < .13 and abs(b[1] - a[1]) < .04 for a, b in zip(positions, positions[1:])))

    def snapshot(self):
        state = super().snapshot()
        state['poses']['block3'] = pose(self.e.block3)
        state.update(layout=self.layout(), layout_valid=self.geometry(),
                     right_open=bool(self.e.is_right_gripper_open()),
                     button_value=float(self.e.get_current_button_value('button')))
        return state

    def atoms(self):
        # The first five atoms are persistent observations, not actor positions.
        count = len(getattr(self, 'memory', {}).get('attempts', []))
        correct = self.geometry() and self.layout() == ['block1', 'block2', 'block3']
        return [count >= i for i in range(1, 6)] + [bool(correct), self.native_success]

    def execute(self):
        e = self.e
        self.memory = dict(attempts=[], initial_visual_identities=self.layout())
        # This seed makes the native fixed enumeration reach success last.
        # This assertion audits seed selection; the controller infers its target
        # exclusively from recorded failed permutations below.
        assert self.layout() == ['block2', 'block1', 'block3']

        def check():
            before = e.press_cnt
            e.press_button()
            self.hold()
            assert e.press_cnt > before and not self.native_success and self.geometry()
            record = dict(layout=self.layout(), native_success=False, frame=len(self.rows)-1,
                          press_count=e.press_cnt, button_was_pressed=True)
            assert record['layout'] not in [a['layout'] for a in self.memory['attempts']]
            self.memory['attempts'].append(record)
            self.mark('按键反馈：此排列失败，记录并排除', 'feedback', attempt=record)
            self.hold(.12)

        def swap(a, b):
            order = self.layout()
            self.mark('交换两个积木，尝试下一排列', 'swap_start', objects=[order[a], order[b]])
            e.swap_blocks(getattr(e, order[a]), getattr(e, order[b]), str(a), str(b))
            self.hold(.3)
            assert self.geometry(), 'Swap did not leave a valid row'

        self.mark('验证初始排列并保存失败反馈', 'trial_start')
        check()
        for a, b in [(1, 2), (0, 2), (0, 1), (1, 2)]:
            self.mark('交换积木并验证一个未尝试的排列', 'trial_start')
            swap(a, b)
            check()
        seen = {tuple(record['layout']) for record in self.memory['attempts']}
        remaining = set(itertools.permutations(self.memory['initial_visual_identities'])) - seen
        assert len(remaining) == 1
        inferred = list(remaining.pop())
        self.memory['inferred_target'] = inferred
        self.mark('排除五个失败排列，摆出唯一未尝试的排列', 'candidate_start',
                  source_frames=[a['frame'] for a in self.memory['attempts']], inferred=inferred)
        swap(0, 2)
        assert self.layout() == inferred and self.atoms()[5] and not self.native_success
        self.mark('候选排列已摆好，等待最终验证', 'candidate_ready')
        self.hold(.4)
        old = self.memory['attempts'][-1]
        self.mark('误把最后一次失败排列记成未尝试候选，交换已摆好的首尾积木', 'memory_error',
                  truth=inferred, corrupted=old['layout'], source_frames=[old['frame']])
        swap(0, 2)
        assert self.layout() == old['layout'] and not self.atoms()[5]
        self.mark('错误排列已摆好', 'wrong_placed')
        self.hold(.4)
        self.mark('重读五次失败记录，重新排除旧排列，恢复唯一剩余排列', 'memory_retrieval',
                  source_frames=[a['frame'] for a in self.memory['attempts']])
        seen = {tuple(record['layout']) for record in self.memory['attempts']}
        target, = set(itertools.permutations(self.memory['initial_visual_identities'])) - seen
        swap(0, 2)
        assert tuple(self.layout()) == target and self.atoms()[5]
        self.mark('排列恢复，按按钮执行最终验证', 'restored')
        e.press_button()


if __name__ == '__main__':
    RankingDemo('blocks_ranking_try', int(sys.argv[1]), sys.argv[2]).run()
