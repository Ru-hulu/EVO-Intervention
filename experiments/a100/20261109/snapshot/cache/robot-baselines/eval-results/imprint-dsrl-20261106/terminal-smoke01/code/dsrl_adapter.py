"""Preserve RMBench's official action loop and IMPRINT history updates."""
import robomme_tcp_policy as base


def compact(obs):
    return {'observation': {k: {'rgb': obs['observation'][k]['rgb']} for k in base.CAMERA_ORDER},
            'joint_action': {'vector': obs['joint_action']['vector']}}


def eval(task_env, model, observation):
    class Proxy:
        def call(self, func_name, obs=None):
            payload = dict(obs)
            if func_name == 'eval':
                payload['observation'] = compact(payload['observation'])
                payload.update(action_step=int(task_env.take_action_cnt), step_limit=int(task_env.step_lim))
            elif func_name == 'add_buffer':
                payload['observations'] = [compact(x) for x in payload['observations']]
                payload.update(success=bool(task_env.eval_success),
                               fail_flag=bool(getattr(task_env, 'fail_flag', False)),
                               action_step=int(task_env.take_action_cnt), step_limit=int(task_env.step_lim),
                               seed=int(task_env.seed))
            return model.call(func_name=func_name, obs=payload)
    return base.eval(task_env, Proxy(), observation)


reset_model = base.reset_model
