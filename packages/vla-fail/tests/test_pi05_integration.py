"""Exercise the real sampler with a tiny Torch backbone; no checkpoints/GPU.

Only heavy model construction imports are stubbed. Prefix/suffix embedding,
Euler integration, LLMD extraction, masks and the semantic window are real.
"""

# Tests intentionally exercise internal sampler boundaries.
# ruff: noqa: SLF001
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[3]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def model(monkeypatch):
    for name in (
        "openpi",
        "openpi.models",
        "openpi.models.gemma",
        "openpi.models_pytorch",
        "openpi.models_pytorch.gemma_pytorch",
        "openpi.models_pytorch.preprocessing_pytorch",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["openpi.models_pytorch.gemma_pytorch"].PaliGemmaWithExpertModel = object
    window = load_module("test_window", ROOT / "src/openpi/models_pytorch/intervention_window.py")
    monkeypatch.setitem(sys.modules, "openpi.models_pytorch.intervention_window", window)
    module = load_module("test_pi05", ROOT / "src/openpi/models_pytorch/pi05_pytorch.py")

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.paligemma = types.SimpleNamespace(language_model=types.SimpleNamespace(config=types.SimpleNamespace()))
            self.gemma_expert = types.SimpleNamespace(model=types.SimpleNamespace(config=types.SimpleNamespace()))
            self.prefix_calls = 0
            self.suffix_calls = 0

        def embed_image(self, image):
            return torch.ones((image.shape[0], 5, 4)) * image.mean()

        def embed_language_tokens(self, tokens):
            return tokens[..., None].float().expand(-1, -1, 4)

        def forward(self, *, inputs_embeds, past_key_values, use_cache, return_prefix_hidden_states=False, **kwargs):
            prefix, suffix = inputs_embeds
            if prefix is not None:
                self.prefix_calls += 1
                assert past_key_values is None
                assert use_cache
                result = ([prefix, None], prefix.mean(dim=1, keepdim=True))
                return (*result, tuple(prefix for _ in range(18))) if return_prefix_hidden_states else result
            self.suffix_calls += 1
            assert not use_cache
            assert kwargs["attention_mask"].shape[-2:] == (3, 10)
            return ([None, suffix + past_key_values], None)

    torch.manual_seed(12)
    instance = module.PI05Pytorch.__new__(module.PI05Pytorch)
    torch.nn.Module.__init__(instance)
    instance.config = types.SimpleNamespace(action_horizon=3, action_dim=2)
    instance.paligemma_with_expert = Backbone()
    instance.action_in_proj = torch.nn.Linear(2, 4)
    instance.action_out_proj = torch.nn.Linear(4, 2)
    instance.time_mlp_in = torch.nn.Linear(4, 4)
    instance.time_mlp_out = torch.nn.Linear(4, 4)
    instance.gradient_checkpointing_enabled = False
    instance.intervention_window = window.VLMFeatureSlidingWindow(window_size=1)
    instance._preprocess_observation = lambda obs, train: (
        [torch.ones(1, 3, 2, 2)],
        [torch.ones(1, dtype=torch.bool)],
        torch.tensor([[1, 2]]),
        torch.ones(1, 2, dtype=torch.bool),
        obs.state,
    )
    return instance.eval()


def test_diagnostics_preserve_actions_rng_and_prefix_cache(model):
    observation = types.SimpleNamespace(state=torch.zeros(1, 2))
    noise = torch.arange(6).reshape(1, 3, 2).float() / 10
    fixed = np.ones((3, 2), dtype=np.float32)
    reference = model.sample_actions("cpu", observation, noise=noise, num_steps=2)
    model.intervention_window.clear()
    backbone = model.paligemma_with_expert
    backbone.prefix_calls = backbone.suffix_calls = 0
    rng_before = torch.random.get_rng_state().clone()
    actions, features = model.sample_actions_with_llmd("cpu", observation, noise=noise, fixed_noise=fixed, num_steps=2)
    torch.testing.assert_close(actions, reference, rtol=0, atol=0)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert backbone.prefix_calls == 1
    assert backbone.suffix_calls == 3
    assert features.shape == (1, 3, 4)
    assert features.dtype == torch.float32
    stored = model.intervention_window.get()
    assert stored["image_tokens"].shape[3] == 5  # Patch tokens, not the raw image's three channels.
    assert stored["prompt_tokens"].shape[3] == 2


def test_offline_features_are_exact_head_inputs_at_pure_noise_endpoint(model, monkeypatch):
    observation = types.SimpleNamespace(state=torch.zeros(1, 2))
    fixed = np.ones((3, 2), dtype=np.float32)
    times = []
    original = model.embed_suffix

    def embed(state, noisy_actions, timestep):
        times.append(timestep.detach().clone())
        return original(state, noisy_actions, timestep)

    monkeypatch.setattr(model, "embed_suffix", embed)
    rng = torch.random.get_rng_state().clone()
    features = model.extract_llmd_features("cpu", observation, fixed_noise=fixed)
    assert model.intervention_window.length == 0
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(times[0], torch.ones(1))
    state, masks, cache = model._encode_prefix(observation, update_window=False)
    head_inputs = []
    handle = model.action_out_proj.register_forward_pre_hook(
        lambda _, args: head_inputs.append(args[0].detach().clone())
    )
    model.denoise_step(state, masks, cache, torch.from_numpy(fixed)[None], torch.ones(1))
    handle.remove()
    torch.testing.assert_close(features, head_inputs[0], rtol=0, atol=0)
    with pytest.raises(ValueError, match="fixed_noise"):
        model.extract_llmd_features("cpu", observation, fixed_noise=np.zeros((3, 3)))
    model.train()
    with pytest.raises(RuntimeError, match="eval"):
        model.extract_llmd_features("cpu", observation, fixed_noise=fixed)


def test_policy_passes_post_transform_actions_and_resets(model, monkeypatch):
    # Isolate model-independent glue from optional JAX/Flax training dependencies.
    names = (
        "flax",
        "flax.traverse_util",
        "jax",
        "jax.numpy",
        "openpi_client",
        "openpi_client.base_policy",
        "openpi.transforms",
        "openpi.models.model",
        "openpi.shared",
        "openpi.shared.array_typing",
        "openpi.shared.nnx_utils",
    )
    modules = {}
    for name in names:
        modules[name] = types.ModuleType(name)
        modules[name].__path__ = []
        monkeypatch.setitem(sys.modules, name, modules[name])

    def tree_map(fn, item):
        return {key: tree_map(fn, value) for key, value in item.items()} if isinstance(item, dict) else fn(item)

    def compose(functions):
        def transform(item):
            for fn in functions:
                item = fn(item)
            return item

        return transform

    modules["jax"].tree = types.SimpleNamespace(map=tree_map)
    modules["openpi_client.base_policy"].BasePolicy = type("BasePolicy", (), {})
    modules["openpi.transforms"].compose = compose
    modules["openpi.transforms"].DataTransformFn = object
    modules["openpi.models.model"].BaseModel = object
    modules["openpi.models.model"].Observation = types.SimpleNamespace(
        from_dict=lambda item: types.SimpleNamespace(**item)
    )
    modules["openpi.shared.array_typing"].KeyArrayLike = object
    policy_module = load_module("test_policy", ROOT / "src/openpi/policies/policy.py")
    policy = policy_module.Policy(
        model,
        is_pytorch=True,
        sample_kwargs={"num_steps": 2},
        output_transforms=[lambda item: {"actions": item["actions"] * 10}],
    )
    observation = {"state": np.zeros(2, dtype=np.float32), "vla_fail": {"episode_id": "first"}}
    noise = np.arange(6, dtype=np.float32).reshape(3, 2) / 10
    reference = policy.infer(observation, noise=noise)["actions"]

    class Monitor:
        fixed_noise = np.ones((3, 2), dtype=np.float32)
        reset_count = 0

        def reset(self):
            self.reset_count += 1

        def begin(self, context):
            assert context["episode_id"] == "first"
            return True

        def step(self, *, features, actions, context):
            np.testing.assert_array_equal(actions, reference)
            assert features.shape == (3, 4)
            return {"need_intervention": None}

    monitor = Monitor()
    policy.attach_vla_fail_monitor(monitor)
    result = policy.infer(observation, noise=noise)
    assert result["vla_fail"]["need_intervention"] is None
    assert model.intervention_window.length == 1
    assert policy.extract_llmd_features(observation, fixed_noise=monitor.fixed_noise).shape == (3, 4)
    assert model.intervention_window.length == 1
    policy.reset()
    assert model.intervention_window.length == 0
    assert monitor.reset_count == 2
