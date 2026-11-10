"""Materialize immutable task configurations; no dataset content is stored in Git."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS = ['pencil', 'flipcup', 'spray', 'pour', 'drawer']
TITLES = dict(pencil='笔袋恢复', flipcup='翻杯恢复', spray='喷瓶检查', pour='重复倒水', drawer='抽屉检查')


def main():
    configs = {}
    for task in TASKS:
        old = json.loads((ROOT/'baseline_configs'/f'{task}.json').read_text())
        config = dict(old)
        setup = ROOT/task
        setup.mkdir(exist_ok=True)
        cache = Path(f'/cache/cvpzhr/datasets/robometer-{task}-5fps-uniform8-current3-clean-20261110')
        config.update(setup_dir=str(setup), train_data=str(cache/'train.jsonl'),
                      validation_data=str(cache/'validation.jsonl'),
                      output_dir=f'/cache/cvpzhr/outputs/robometer-{task}-fromofficial-uniform8-clean-2gpu-e3-20261110',
                      temporal_sampling='uniform-prefix8-rint-rightpad-current-v1',
                      padding='repeat_current_frame_on_right', sampling_domain='observed_5fps_prefix_including_current',
                      source_cache=str(Path(old['train_data']).parent), title=TITLES[task],
                      runtime_code_path=str(ROOT/'runtime'), gpu_ids=[2,3])
        assert config['save_steps'] == config['steps_per_epoch']
        assert config['eval_steps'] == config['expected_steps'] == 3*config['steps_per_epoch']
        assert config['training']['per_device_train_batch_size']*config['training']['gradient_accumulation_steps']*2 == 18
        p=setup/'training.config.json'
        if p.exists():
            assert json.loads(p.read_text()) == config, f'Existing configuration differs: {p}'
        else:p.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
        configs[task] = str(p)
    policy = dict(task_order=TASKS, strict_serial=True, gpus=[2,3], tasks=configs,
                  epochs=3, learning_rate=1e-4, warmup_ratio=.03, scheduler='cosine', global_batch=18,
                  initialization='independent official Robometer-4B + fresh language LoRA for every task',
                  augmentation='none', temporal_sampling='uniform-prefix8-rint-rightpad-current-v1',
                  current_three_views_preserved=True, images_per_sample=10, save_each_epoch=True,
                  final_evaluation_epoch=3, next_task_requires='training, verification, metrics and all plots complete',
                  upstream='/cache/cvpzhr/code/robometer-current3-w8-e1-20261104/third_party/robometer-352d160389daa964788de1ec933d1925f3a6de4f',
                  python='/cache/cvpzhr/envs/robometer/bin/python')
    p=ROOT/'launch-policy.json'
    if p.exists():assert json.loads(p.read_text())==policy
    else:p.write_text(json.dumps(policy,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(policy,ensure_ascii=False))


if __name__=='__main__':main()
