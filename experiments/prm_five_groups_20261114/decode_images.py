"""Pinned CPU decoder bridge for model environments without PyAV 19 wheels."""
import json
import os
import sys


def main():
    import av
    from .eval_common import ensure_images
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert av.__version__=='19.0.2'
    request=json.load(sys.stdin)
    root=ensure_images(request['episode'],request['main_only'])
    print(json.dumps({'path':str(root),'av_version':av.__version__}))


if __name__=='__main__':
    main()
