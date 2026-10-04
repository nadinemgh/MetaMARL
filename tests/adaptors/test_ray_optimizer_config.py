"""``RayOptimizerConfig.build_optimizer``: missing inputs fail at build time.

Every environment fetches its mechanism from the shared ``World`` and builds
its followers from the declared agents, so a config without either cannot
produce a working optimizer. It must say so when it is built, not when RLlib
first creates an environment.
"""

import pytest

from core.optimizers.appo.config import APPOptimizerConfig


@pytest.mark.unit
def test_build_without_a_world_is_a_clear_error():
    with pytest.raises(ValueError, match="needs the shared World actor"):
        APPOptimizerConfig().build_optimizer(world=None)


@pytest.mark.unit
def test_build_without_agents_is_a_clear_error():
    with pytest.raises(ValueError, match=r"no agents: call \.agents\("):
        APPOptimizerConfig().build_optimizer(world=object(), world_name="w")
