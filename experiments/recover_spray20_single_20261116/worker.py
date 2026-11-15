"""Bind the existing Recovery singleton inference to the new final weights."""
import os
from .common import *


def load(name, info):
    import torch
    os.environ['RANK'] = '0'
    proof = read(info['proof']); cfg = info['config']; validate_proof(name, proof, cfg)
    p = Path(info['checkpoint'])
    if name == 'forest':
        from mem_lite.runtime import load_checkpoint
        from mem_lite.train_stage3 import trainable_state
        from experiments.prm_five_groups_20261114.forest_adapter import ForestRuntime, state_digest
        assert sha(p) == info['checkpoint_sha256']
        ck = torch.load(p, map_location='cpu', weights_only=True)
        assert ck['epoch'] == 1 and ck['next_batch'] == 0 and ck['step'] == 751
        assert ck['optimizer']['state'] and ck['scheduler'] is None and len(ck['rng_by_rank']) == 2
        assert ck['config']['data_signature'] == cfg['native']['data_signature']
        assert ck['config']['run_protocol']['selection_sha256'] == sha(TRAIN_SELECTION)
        base = read('/cache/cvpzhr/setup/qwen35-original-download-20261113/verification.json')
        assert base['all_official_hashes_verified']
        for filename, item in base['files'].items(): assert sha(Path(base['destination'])/filename) == item['sha256']
        expected = state_digest(ck['model']); del ck
        model, processor, config = load_checkpoint(p, device='cuda', event_split_method='max_depth')
        actual = state_digest(trainable_state(model)); assert actual == expected
        return ForestRuntime(model, processor, config), dict(loaded_tensor_digest=actual,
            strict_final_state=True, base_revision=base['revision'], official_base_files=base['files'])
    from experiments.soc35_finetune_20261114 import train as native_train
    from experiments.recover_finetune_20261114.train import robometer_api
    native_train.robometer_api = robometer_api
    for path, digest in cfg['weights_sha256'].items(): assert sha(path) == digest
    native = read(CACHE_SETUP/f'{name}-native-provenance.json')
    for path, digest in native['source_sha256'].items(): assert sha(path) == digest
    model, processor, original = native_train.load_model(cfg)
    assert sha(p/'trainable-state.pt') == info['checkpoint_sha256']
    state = torch.load(p/'trainable-state.pt', map_location='cpu', weights_only=True)
    current = {n:v for n,v in model.named_parameters() if v.requires_grad}
    assert set(state) == set(current)
    with torch.no_grad():
        for n, v in current.items():
            assert v.shape == state[n].shape and v.dtype == state[n].dtype; v.copy_(state[n])
    expected = read(p/'trainable-state.json')['sha256']; assert tensor_digest(current.items()) == expected
    model.to('cuda').eval().requires_grad_(False)
    audit = dict(loaded_tensor_digest=expected, strict_final_state=True, base_revision=cfg['official_revision'],
        base_weight_sha256=cfg['weights_sha256'], native_source_sha256=native['source_sha256'])
    if name == 'robometer-4b': return (model, processor, robometer_api()), audit
    from experiments.prm_five_groups_20261114.official_baseline import native_module
    worker = native_module(); shell = object.__new__(worker.Generative)
    shell.kind = 'proc' if name == 'procvlm-2b' else 'grm'; shell.processor = processor
    shell.template = (worker.ROOT/('proc_prompt.txt' if shell.kind == 'proc' else 'grm_prompt.txt')).read_text()
    return (model, processor, shell, worker), audit


if __name__ == '__main__':
    from experiments.recover_validation_20261114 import worker as native
    native.ROOT, native.OUT, native.IMAGES, native.MODELS = ROOT, OUT, IMAGES, MODELS
    native.REGIME = 'Recover-R03-20-train-trajectories-1-epoch'
    native.result_path, native.frame_inputs, native.load = result_path, frame_inputs, load
    native.main()
