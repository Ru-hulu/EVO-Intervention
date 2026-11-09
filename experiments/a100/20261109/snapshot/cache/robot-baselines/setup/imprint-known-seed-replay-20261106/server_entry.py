"""Expose the existing policy's RNG seed without changing its sampling recipe."""
import os,sys
from pathlib import Path
script=Path(os.environ['IMPRINT_BATCH_ROOT'])/'repos/imprint/benchmarks/rmbench/eval/runtime_overlay/adapter/serve_robomme_tcp.py'
source=script.read_text()
old='    server = TCPPolicyServer(get_model(config), host=args.host, port=args.port)'
new='''    model = get_model(config)
    import torch
    original_reset = model.policy.reset
    def reset_with_cache_release(*args, **kwargs):
        result = original_reset(*args, **kwargs)
        torch.cuda.empty_cache()
        return result
    model.policy.reset = reset_with_cache_release
    torch.cuda.empty_cache()
    def set_collection_seed(value):
        seed = int(value)
        model.policy._seed = seed
        return {"policy_seed": seed}
    model.set_collection_seed = set_collection_seed
    set_collection_seed(os.environ['IMPRINT_POLICY_SEED'])
    server = TCPPolicyServer(model, host=args.host, port=args.port)'''
assert old in source
exec(compile(source.replace(old,new),str(script),'exec'),{'__name__':'__main__','__file__':str(script)})
