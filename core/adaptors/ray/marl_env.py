# TODO (nadine) wraps ray multiagentenv with MetaMARL abstraction

"""Multi-agent environment regulated by a published mechanism.

``MultiAgentRegulatedEnv`` is the RLlib-facing environment of the inner
optimization level. It combines :class:`core.envs.regulated.RegulatedEnv`
(mechanism lifecycle and regulated reward) with RLlib's ``MultiAgentEnv``. A
concrete benchmark implements the abstract pieces of the step:
``transition_kernel`` (``S_{t+1} = T(S_t, A_t)``), ``intrinsic_utility``
(``u_i = U(a_i, S_t)``), ``violation_signal`` and ``penalty`` (the regulated
reward ``u_i - lambda(M) * v_i``), ``_observation`` (``o_i = O_i(S_t)``) and
``_is_truncated``. The base class owns the step lifecycle: it computes the
intrinsic utilities, shapes and aggregates the rewards, advances the state,
appends the mechanism vector ``theta`` to every observation and publishes an
``EnvStepContext`` to the ``World``.

The mechanism in force is fetched from the ``World`` at ``reset`` by
``mechanism_id``. Until one is published the environment returns zero rewards
and does not advance its dynamics, so RLlib's environment checks can run
before training starts.

Metrics are optional: with a ``schema`` the env owns a ``MetricLogger`` fed by
``_log`` (episode identity at ``reset``, rewards and ``iter`` at every step),
and with a ``reporter_cfg`` a ``Reporter`` renders the configured ``queries``
against it (see ``core.callbacks``).
"""

import logging
from typing import Any, Optional

# TODO remove ray and gymnasium dependency
from ray.rllib import MultiAgentEnv as RllibMultiAgentEnv
from gymnasium import spaces

from core.annotations import override
from core.envs.marl_regulated import MultiAgentEnv
from core.mechanism.base import MDPState
from core.types import MultiAgentDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

# TODO future enhancement. decouple gym inheritance to support MPC, trajectory optimization etc.
class RLlibMultiAgentEnvAdapter(RllibMultiAgentEnv):

    _env: MultiAgentEnv
    _mdp: MDPState

    def __init__(
        self,
        env: MultiAgentEnv,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.env = env

        self.observation_spaces = {aid: agent.observation_space for aid, agent in self.env.agents.items()}
        self.observation_spaces = spaces.Dict(self.observation_spaces)

        # Multi-agent environment
        self.agents = self.env.agents.keys()
        self.possible_agents = list(self.agents)
        self.action_spaces = {
            aid: spaces.Dict({
                mid: m.action_space for mid, m in agent.mechanisms.items()
            })
            for aid, agent in self.env.agents.items()
        }
        self.action_spaces = spaces.Dict(self.action_spaces)

        # TODO (nadine) env serialization with numpy to avoid using MDPState and faster computation
        self._mdp = MDPState(
            aids=set(self.possible_agents),
            obs_space=self.observation_spaces, 
            action_spaces=self.action_spaces
        )


    @override(RllibMultiAgentEnv)
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[MultiAgentDict, MultiAgentDict]:
        # NOTE:
        # `seed` and `options` are environment-bound configuration.
        # The environment is seeded once at construction and owns a persistent RNG
        # (`self.rng`) for its lifetime. Episode resets do not reseed the environment.
        # `reset(seed=..., options=...)` keeps the Gymnasium/RLlib-compatible signature,
        # but these arguments are not used to mutate the environment's persistent
        # configuration after construction.
        self._mdp = self.env.reset(self._mdp)
        return self._mdp.obs, {}

    @override(RllibMultiAgentEnv)
    def step(
        self, action_dict: MultiAgentDict
    ) -> tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        self._mdp.actions = action_dict
        m: MDPState = self.env.step(self._mdp)
        return m.obs, m.rewards, m.terminateds, m.truncateds, {}
