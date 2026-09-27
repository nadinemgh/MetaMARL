import pytest
import ray

from src.ppo.config import PPOptimizerConfig


@pytest.mark.integration
def test_ppo_cartpole_training():
    ray.init(ignore_reinit_error=True, num_cpus=2)

    cfg = (
        PPOptimizerConfig().environment(env="CartPole-v1").training(lr=3e-4, gamma=0.99)
    )

    opt = cfg.build_optimizer()

    result = opt.train()

    reward = result.get("env_runners", {}).get("episode_return_mean", 0)
    assert reward > 10

    opt.stop()
    ray.shutdown()
