"""Explicit user cancellations amend execution scope without changing frozen inputs."""
import json
from pathlib import Path

from .eval_common import EVAL


def cancelled_models(evaluation=None):
    root = EVAL if evaluation is None else Path(evaluation)
    path = root / 'r2vlm-user-cancellation.json'
    if not path.exists():
        return set()
    receipt = json.loads(path.read_text())
    if receipt.get('status') != 'cancelled_by_user':
        return set()
    assert receipt.get('user_authorized') is True
    assert receipt['model'] == 'r2vlm-alfred-rl'
    assert receipt['scope'] == 'all evaluation groups in this experiment'
    assert receipt['preserve_existing_predictions'] is True
    return {receipt['model']}
