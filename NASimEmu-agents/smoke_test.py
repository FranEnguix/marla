from pathlib import Path
from types import SimpleNamespace

import gym
import torch

from config import config
from nasim_problem import NASimRRL


ROOT = Path(__file__).resolve().parent
SCENARIO = (ROOT / "../NASimEmu/scenarios/uni.v2.yaml").resolve()
MODEL = (ROOT / "trained_models/mlp.pt").resolve()

assert SCENARIO.is_file(), f"Scenario not found: {SCENARIO}"
assert MODEL.is_file(), f"Model not found: {MODEL}"
assert torch.cuda.is_available(), "CUDA is unavailable"

# Match the configuration documented for trained_models/mlp.pt.
args = SimpleNamespace(
    batch=1,
    epoch=1,
    alpha_h=0.3,
    force_continue_epochs=0,
    emb_dim=64,
    mp_iterations=3,
    seed=123,
    device="cuda",
    cpus="1",
    lr=3e-3,
    max_norm=3.0,
    max_epochs=1,
    load_model=str(MODEL),
    emulate=False,
    scenario=str(SCENARIO),
    test_scenario=None,
    episode_step_limit=100,
    use_a_t=True,
    fully_obs=False,
    observation_format="list",
    augment_with_action=True,
    net_class="NASimNetMLP",
)

problem = NASimRRL()
problem_config = problem.make_config()

config.init(args)
problem_config.update_config(config, args)
problem.register_gym()

env = gym.make(
    problem.get_gym_name(),
    random_init=False,
)

net = problem.make_net()
net.load(str(MODEL))
net.eval()

parameter_device = next(net.parameters()).device
assert parameter_device.type == "cuda", (
    f"Model was created on {parameter_device}, not CUDA"
)

observation = env.reset()

with torch.inference_mode():
    actions, values, probabilities, raw_actions = net([observation])

assert len(actions) == 1
assert torch.isfinite(values).all()
assert torch.isfinite(probabilities).all()

action = actions[0]
next_observation, reward, done, info = env.step(action)

print("=" * 60)
print("Scenario:", SCENARIO)
print("Model:", MODEL)
print("Torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Model device:", parameter_device)
print("Observation shape:", observation.shape)
print("Selected action:", action)
print("Raw action:", raw_actions.detach().cpu().tolist())
print("Value estimate:", values.detach().cpu().flatten().tolist())
print("Action probability:", probabilities.detach().cpu().flatten().tolist())
print("Reward:", reward)
print("Done:", done)
print("Next observation shape:", next_observation.shape)
print("=" * 60)
print("NASimEmu-agents pretrained inference test PASSED")

env.close()
