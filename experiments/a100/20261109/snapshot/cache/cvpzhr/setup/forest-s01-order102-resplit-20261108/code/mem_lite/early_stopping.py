"""Validation-MAE early stopping; state lives in the checkpoint config."""
import math


def validate_settings(config):
    patience = config.setdefault('early_stopping_patience', 3)
    delta = config.setdefault('early_stopping_min_delta_pp', 0.0)
    if type(patience) is not int or patience < 0:
        raise ValueError('early_stopping_patience must be an integer >= 0 (0 disables)')
    if not math.isfinite(delta) or delta < 0:
        raise ValueError('early_stopping_min_delta_pp must be finite and >= 0')
    if patience and not config.get('validation_sample_files'):
        raise ValueError('Early stopping requires validation_sample_files; set patience=0 to disable')


def update_early_stopping(config, mae_pp, step):
    """All ranks receive the same globally reduced MAE and update identically."""
    if not math.isfinite(mae_pp):
        raise ValueError('Non-finite validation MAE')
    patience = config['early_stopping_patience']
    if not patience:
        return False
    state = config.setdefault('early_stopping_state', {})
    if state.get('stopped', False):
        return True
    if mae_pp < state.get('reference_mae_pp', float('inf')) - config['early_stopping_min_delta_pp']:
        state.update(reference_mae_pp=mae_pp, reference_step=step, bad_validations=0)
    else:
        state['bad_validations'] += 1
    state['stopped'] = state['bad_validations'] >= patience
    if state['stopped']:
        state['stop_step'] = step
    return state['stopped']
