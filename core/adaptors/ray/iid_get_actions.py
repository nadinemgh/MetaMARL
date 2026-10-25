from typing import Any, Optional
import torch
from ray.rllib.connectors.module_to_env import GetActions
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import EpisodeType


class IidGetActions(GetActions):
    def __init__(self):
        super().__init__()

        self.rng_states = {}

    @override(GetActions)
    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: dict[str, Any],
        episodes: list[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        is_multi_agent = isinstance(episodes[0], MultiAgentEpisode)

        if is_multi_agent:
            for module_id, module_data in batch.copy().items():
                if module_id not in self.rng_states:
                    seed = int(module_id.rsplit("_s", 1)[1])
                    self.rng_states[module_id] = (
                        torch.Generator(device="cpu").manual_seed(seed).get_state()
                    )

                with torch.random.fork_rng(devices=[]):
                    torch.set_rng_state(self.rng_states[module_id])
                    self._get_actions(module_data, rl_module[module_id], explore)

                    self.rng_states[module_id] = torch.get_rng_state().clone()
        else:
            self._get_actions(batch, rl_module, explore)

        return batch


def build_iid_module_to_env_connector(cfg, env=None, spaces=None):
    pipeline = type(cfg).build_module_to_env_connector(cfg, env=env, spaces=spaces)

    pipeline.remove(GetActions)
    pipeline.prepend(IidGetActions())

    return pipeline
