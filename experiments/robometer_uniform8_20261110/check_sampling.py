"""Compare every prefix against the two pinned official sampling functions."""
import ast
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from robometer_training.data import ObservationWindow
from robometer_training.uniform import prefix_indices, POLICY


def main():
    root=Path(__file__).resolve().parent
    policy=json.loads((root/'launch-policy.json').read_text())
    source=Path(policy['upstream'])/'robometer/data/datasets/helpers.py'
    tree=ast.parse(source.read_text())
    names={'linspace_subsample_frames','pad_trajectory_to_max_frames_np'}
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert len(functions)==2
    ns=dict(np=np,List=List,Optional=Optional,Tuple=Tuple)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),ns)
    frames=np.arange(10001)
    window=ObservationWindow(8)
    for current in range(len(frames)):
        selected,indices=ns['linspace_subsample_frames'](frames,8,end_idx=current)
        padded,_=ns['pad_trajectory_to_max_frames_np'](selected,indices,8,pad_from='right')
        actual=prefix_indices(current)
        assert actual==padded.tolist(),current
        assert window.push('first',current)==actual
        assert len(actual)==8 and actual[0]==0 and actual[-1]==current
        assert all(0<=i<=current for i in actual)
    assert window.push('second',987)==[987]*8
    assert prefix_indices(1)==[0,1,1,1,1,1,1,1]
    assert prefix_indices(7)==list(range(8))
    assert prefix_indices(20)==[0,3,6,9,11,14,17,20]
    receipt=dict(status='passed',prefixes_tested=len(frames),temporal_sampling=POLICY,
                 official_source=str(source),episode_reset_passed=True,
                 early_prefix=[prefix_indices(i) for i in range(9)])
    (root/'sampling-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
