import uuid

import numpy as np
import pytest
import ray
from gymnasium import spaces
from src.es.config import ESConfig
from src.ppo.config import PPOptimizerConfig

from core.annotations import override
from core.envs.regulated import RegulatedEnv
from core.envs.regulator import RegulatorEnv
from core.mechanism.space import MechanismSpace
from core.types import OptimizerID
from core.world.base import World


@pytest.mark.integration
def test_es_regulator_loop():
    WORLD_NAME = f"global_world_{uuid.uuid4().hex[:8]}"
    world = World.options(name=WORLD_NAME).remote()

    optimum = np.array([0.8, 0.2, 0.6], dtype=np.float32)

    class DummyRegulatorEnv(RegulatorEnv):
        def __init__(
            self, *, world: World, opt_id: OptimizerID, optimum: np.ndarray, **kwargs
        ):
            super().__init__(world=world, opt_id=opt_id, optimizer=None)
            self.optimum = optimum

        @override(RegulatorEnv)
        def _step(self, theta: np.ndarray):
            x = np.asarray(theta)
            fitness = -np.sum((x - self.optimum) ** 2, axis=1)
            return None, fitness, False, False, {}

        @override(RegulatorEnv)
        def reward(self, reward: np.ndarray) -> np.ndarray:
            return reward

    es_cfg: ESConfig = (
        ESConfig()
        .training(
            dimension=optimum.shape[0],
            pop_size=4,
            sigma=0.1,
            mean_lr=0.2,
            sigma_lr=0.0,
        )
        .environment(env=DummyRegulatorEnv, env_config={"optimum": optimum})
    )

    es = es_cfg.build_optimizer(world=world)

    # TODO we need a main orchestrator
    for _ in range(3):
        es.train()

    assert len(ray.get(world.get_opt_ctx_ids.remote(es.id))) > 0


@pytest.mark.integration
def test_ppo_with_regulated_env():
    WORLD_NAME = f"global_world_{uuid.uuid4().hex[:8]}"
    world = World.options(name=WORLD_NAME).remote()

    class DummyRegulatedEnv(RegulatedEnv):
        def __init__(self, *, world, opt_id, **kwargs):
            super().__init__(world=world, opt_id=opt_id, **kwargs)
            self.observation_space = spaces.Box(-1, 1, (4,), np.float32)
            self.action_space = spaces.Discrete(2)
            self.t = 0
            self.max_steps = 25

        def _step(self, action):
            obs = np.random.randn(4).astype(np.float32)
            reward = 1.0
            terminated = self.t >= self.max_steps
            return obs, reward, terminated, False, {}

        def _reset(self):
            self.t = 0
            return np.random.randn(4).astype(np.float32)

        def violation_signal(self):
            return 0.5

        def violation_penalty(self):
            return 0.2

    # TODO perhaps a world config fc
    cfg = (
        PPOptimizerConfig()
        .environment(env=DummyRegulatedEnv, env_config={"world_name": WORLD_NAME})
        .framework(framework="torch")
        .resources(num_gpus=0)
        .training(train_batch_size=200)
    )

    ppo = cfg.build_optimizer(world=world, world_name=WORLD_NAME)

    # Run a few iterations
    for _ in range(3):
        result = ppo.train()

    assert result["learners"]["default_policy"]["num_module_steps_trained"] > 0
    ids = ray.get(world.get_opt_ctx_ids.remote(ppo.id))
    assert len(ids) > 0


@pytest.mark.integration
def test_full_bilevel_es_ppo_loop():
    WORLD_NAME = f"global_world_{uuid.uuid4().hex[:8]}"
    world = World.options(name=WORLD_NAME).remote()

    # Mechanism Definition
    class DummyMechanism:
        def __init__(self, x: np.ndarray):
            self.x = np.asarray(x, dtype=np.float32)

        def to_vector(self) -> list[float]:
            return self.x.tolist()

        @classmethod
        def from_vector(cls, v: list[float]):
            return cls(np.asarray(v, dtype=np.float32))

    class DummyMechanismSpace(MechanismSpace):
        def sample(self):
            return DummyMechanism(np.random.rand(1))

        def project(self, theta):
            return theta

        def clip(self, theta):
            return theta

        def from_vector(self, v):
            return DummyMechanism(np.asarray(v, dtype=np.float32))

    # TODO hide world interaction logic because its a bit hacky
    # --- Regulated Environment ---
    class DummyBilevelRegulatedEnv(RegulatedEnv):
        def __init__(self, *, world, opt_id, **kwargs):
            super().__init__(world=world, opt_id=opt_id, **kwargs)
            self.observation_space = spaces.Box(-1, 1, (1,), np.float32)
            self.action_space = spaces.Discrete(2)

        def _reset(self):
            # Return a valid initial observation
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        def _step(self, action):
            theta = ray.get(self.world.get_latest_mechanism.remote())

            x = theta.to_vector()[0]

            obs = np.array([x], dtype=np.float32)
            reward = 1.0
            return obs, reward, False, False, {}

        def violation_signal(self):
            theta = ray.get(self.world.get_latest_mechanism.remote())
            return abs(theta.to_vector()[0])

        def violation_penalty(self):
            return 0.1

    ppo_cfg = (
        PPOptimizerConfig()
        .environment(
            env=DummyBilevelRegulatedEnv, env_config={"world_name": WORLD_NAME}
        )
        .framework(framework="torch")
        .resources(num_gpus=0)
        .training(train_batch_size=200)
    )

    # TODO remore any world and world_name instantiation from build_optimizer
    ppo = ppo_cfg.build_optimizer(world=world, world_name=WORLD_NAME)

    # --- Regulator Environment ---
    # TODO passing a callable to the reward !
    class DummyBilevelRegulatorEnv(RegulatorEnv):
        @override(RegulatorEnv)
        def aggregate_rewards(self, rewards):
            return float(np.mean(rewards))

    es_cfg: ESConfig = (
        ESConfig()
        .training(
            dimension=1,
            pop_size=4,
            sigma=0.1,
            mean_lr=0.2,
            sigma_lr=0.0,
        )
        .environment(
            env=DummyBilevelRegulatorEnv,
            env_config={
                "mechanism_space": DummyMechanismSpace(),
            },
            train_iters=2,
            eval_iters=2,
        )
    )

    es = es_cfg.build_optimizer(world=world, inner_opt=ppo)

    # --- Run full bilevel loop ---
    for _ in range(5):
        es.train()

    assert len(ray.get(world.get_ctx_ids.remote())) > 1
    assert len(ray.get(world.get_opt_registry.remote())) > 1
