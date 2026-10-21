from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import ray
from ray.rllib.utils.typing import AgentID
from ray.train._internal.checkpoint_manager import _TrainingResult
from ray.actor import ActorHandle

from core.annotations import override
from core.optimizers.base import Optimizer
from core.world.base import World
from core.reporting.wandb import WandbReporter
from core.utils import to_float

# TODO temporary
from core.adaptors.ray.utils import (
    get_env_steps,
    get_episode_return_mean,
    get_policy_loss_if_present,
)
from core.reporting.utils.env_reduced import (
    ReductionSpec,
    build_default_fishery_reduction_specs,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.adaptors.ray.optimizer_config import RayOptimizerConfig


# TODO perhaps we would first want an adaptor for core ray algorithm and then the PPO inherits it
class RayOptimizer(Optimizer):
    def __init__(
        self,
        # algo: Algorithm,
        config: RayOptimizerConfig,
    ):
        super().__init__(config)
        # self.algo = algo
        # self.eval_episodes = config.eval_episodes
        # TODO fallback if rollout_fragment_length not in eval_cfg
        self.eval_episodes = (
            config.rllib_cfg.evaluation_duration
            // config.rllib_cfg.evaluation_config.get("rollout_fragment_length")
        )
        # self.rollout_fragment_length = config.rollout_fragment_length

        from core.adaptors.ray.policy_actor import PolicyActor

        self.policy_actor = PolicyActor.remote(config.rllib_cfg)

        # Track training metrics for plotting
        self._training_rewards: list[float] = []
        self._training_losses: list[float] = []
        self._inner_iter: int = 0
        self._es_round: int = 0

        # TODO remove this - temporary for testing
        self._env_reducers: list[ReductionSpec] = (
            getattr(config, "env_reducers", None) or []
        )

        if not self._env_reducers:
            self._env_reducers = build_default_fishery_reduction_specs()

    @property
    @override(Optimizer)
    def batch_capacity(self) -> int:
        num_envs = self.config.rllib_cfg.num_envs_per_env_runner
        num_seeds = len(self.config.seeds)
        num_mechanisms = num_envs // num_seeds
        return num_mechanisms

    # TODO move to utils
    def _build_agent_policy_map(self) -> dict[AgentID, str]:
        agent_to_policy: dict[AgentID, str] = {}

        for agent_type, spec in self.config.agent_specs.items():
            policy_id = spec["policy"]
            count = spec["count"]

            for i in range(count):
                agent_id = f"{agent_type}:{i}"
                agent_to_policy[agent_id] = policy_id

        return agent_to_policy

    def _get_policy_handle(self, policy_id: str):
        # RLModule API (newer)
        try:
            return self.algo.get_module(policy_id)
        except Exception:
            # Policy API (older / classic)
            return self.algo.get_policy(policy_id)

    @override(Optimizer)
    def run(self) -> None:
        logger.info("[PPO] Training step started")
        result = ray.get(self.policy_actor.train.remote())
        # step = int(to_float(result.get("training_iteration")) or 0)

        # Local inner-loop iteration. This resets for each outer ES round.
        self._inner_iter += 1
        step = self._inner_iter

        # RLlib's own lifetime training counter, retained only for debugging.
        rllib_training_iteration = int(
            to_float(result.get("training_iteration")) or 0
        )

        # TODO make this more dynamic NEW_STACK
        # TODO move this to world
        ray.get(
            self.reporting.plot_ray_result.remote(
                outer_iter=self._es_round,
                training_episode=step,
                results=result,
                prefix="appo/train",
            )
        )

        eval_result = result.get("evaluation")
        if eval_result:
            ray.get(
                self.reporting.plot_ray_result.remote(
                outer_iter=self._es_round,
                training_episode=step,
                results=eval_result,
                prefix="appo/eval",
            )
            )

        # TODO reduced env episode plotting
        if self._env_reducers:
            latest_env_ctxs = ray.get(
                self.world.get_new_env_step_contexts.remote(opt_id=self.opt_id)
            )

            if latest_env_ctxs:
                ray.get(
                    self.reporting.plot_env_reduced.remote(
                        ctxs=latest_env_ctxs,
                        outer_iter=self._es_round,
                        training_episode=step,
                        reducers=self._env_reducers,
                        prefix="env_reduced",
                    )
                )

        # TODO temporary to be moved to a logger Extract metrics
        ep_return = get_episode_return_mean(result)
        steps_iter, steps_life = get_env_steps(result)

        # Track metrics
        self._training_rewards.append(ep_return)
        policy_loss = get_policy_loss_if_present(result)
        self._training_losses.append(policy_loss)

        logger.info(
            "[PPO] Training step completed | "
            "outer_iter=%d | inner_iter=%d | rllib_iter_lifetime=%d | "
            "ep_return=%.4f | env_steps_iter=%d | "
            "env_steps_lifetime=%d | policy_loss=%s",
            self._es_round,
            step,
            rllib_training_iteration,
            ep_return,
            steps_iter,
            steps_life,
            f"{policy_loss:.6f}" if np.isfinite(policy_loss) else "NA",
        )

    @override(Optimizer)
    def evaluate(self) -> None:
        logger.info("[PPO] Evaluation started")
        ray.get(self.policy_actor.evaluate.remote())
        logger.info("[PPO] Evaluation completed")

    @override(Optimizer)
    def reset(self) -> None:
        """Reset policy weights to random initialization."""
        logger.info("[PPO] Resetting policy weights")
        self._training_rewards = []
        self._training_losses = []
        self._inner_iter = 0
        self._es_round += 1
        ray.get(self.policy_actor.reset.remote())

    @override(Optimizer)
    def stop(self) -> None:
        ray.get(self.policy_actor.stop.remote())

    @override(Optimizer)
    def save(self, checkpoint_dir: Optional[str] = None) -> _TrainingResult:
        # TODO
        pass
