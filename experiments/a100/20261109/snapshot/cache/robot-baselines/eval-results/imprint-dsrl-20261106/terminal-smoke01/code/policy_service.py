"""Noise steering around the frozen, unmodified IMPRINT memory policy."""
import argparse
import importlib
import json

import jax.numpy as jnp
import numpy as np
import yaml

from serve_robomme_tcp import TCPPolicyServer


class NoisePolicy:
    def __init__(self, config):
        module = importlib.import_module(config.pop('adapter_module'))
        self.model = module.get_model(config)
        self.noise_dim = int(self.model.policy._model.action_dim)
        self.horizon = int(self.model.policy._model.action_horizon)
        self.pending_noise = None
        original = self.model.policy._sample_actions

        def sample(rng, observation, **kwargs):
            if self.pending_noise is not None:
                kwargs['noise'] = self.pending_noise
            return original(rng, observation, **kwargs)

        self.model.policy._sample_actions = sample

    def metadata(self):
        return dict(noise_dim=self.noise_dim, horizon=self.horizon,
                    provenance=self.model.get_runtime_provenance())

    def warmup_runtime(self):
        return self.model.warmup_runtime()

    def features(self, payload):
        if not self.model._episode_active:
            self.model._start_episode(payload['prompt'], payload['observation'])
        elif payload['prompt'] != self.model._prompt:
            raise ValueError('Prompt changed inside an episode')
        policy = self.model.policy
        if not policy._uses_gated_delta:
            raise ValueError('The first DSRL integration targets recurrent IMPRINT')
        inputs = policy._prepare_history({})
        batch = lambda name: jnp.asarray(inputs[name])[None]
        memory = policy._delta_read(policy._delta_state,
                    batch('cosmo_recent_emb'), batch('cosmo_action_emb'), batch('cosmo_proprio_emb'),
                    batch('cosmo_lang_emb'), batch('cosmo_recent_mask'),
                    batch('cosmo_action_mask'), batch('cosmo_proprio_mask'))
        memory = np.asarray(memory[0], dtype=np.float32).mean(axis=0)
        recent = inputs['cosmo_recent_emb'][inputs['cosmo_recent_mask']]
        recent = recent.mean(axis=0) if len(recent) else np.zeros(inputs['cosmo_recent_emb'].shape[-1])
        parts = [memory, recent, np.asarray(inputs['cosmo_lang_emb']).reshape(-1)]
        parts = [x / max(float(np.sqrt(np.mean(x*x))), 1e-6) for x in parts]
        feature = np.concatenate(parts).astype(np.float32)
        if not np.isfinite(feature).all():
            raise ValueError('Non-finite IMPRINT features')
        return {'feature': feature}

    def reset_model(self):
        self.pending_noise = None
        return self.model.reset_model()

    def eval(self, payload):
        noise = payload.get('noise')
        if noise is not None:
            noise = np.asarray(noise, dtype=np.float32)
            if noise.shape != (self.noise_dim,) or not np.isfinite(noise).all():
                raise ValueError('Invalid DSRL noise')
            # Same 1-vector -> horizon broadcast as RoboMeter pi0_integration.py.
            self.pending_noise = jnp.asarray(np.repeat(noise[None, None], self.horizon, axis=1))
        try:
            return self.model.eval(payload)
        finally:
            self.pending_noise = None

    def add_buffer(self, payload):
        return self.model.add_buffer(payload)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--port', type=int, required=True)
    a = p.parse_args()
    model = NoisePolicy(yaml.safe_load(open(a.config)))
    print(json.dumps({'event': 'policy_ready', **model.metadata()}), flush=True)
    TCPPolicyServer(model, host='127.0.0.1', port=a.port).serve_forever()
