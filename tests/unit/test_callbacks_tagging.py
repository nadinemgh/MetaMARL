"""Unit tests for ``tag_episode_with_env_idx`` in ``core.callbacks``.

The hook rewrites the RLlib episode ID from the identity of the sub-environment
(``mechanism_id``, ``seed``, ``policy_seed``) and guards the immutability of
``env_id``. It is exercised with duck-typed fakes of the env runner and of the
episode, so no Ray runtime is started. The episode-end hook and the evaluation
function are covered in ``test_callbacks_logging.py`` and
``test_callbacks_evaluation.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.callbacks import tag_episode_with_env_idx

# --------------------------------------------------------------------------- #
# tag_episode_with_env_idx
# --------------------------------------------------------------------------- #


class FakeSubEnv:
    """Minimal stand-in for a ``BaseEnv`` created by the env creator."""

    def __init__(self, mechanism_id=0, seed=101, policy_seed=101, env_id=None):
        self.mechanism_id = mechanism_id
        self.seed = seed
        self.policy_seed = policy_seed
        self.env_id = env_id


def make_runner(*sub_envs):
    """Build ``env_runner.env.envs[i].unwrapped`` around the given sub-envs."""
    wrapped = [SimpleNamespace(unwrapped=sub_env) for sub_env in sub_envs]
    return SimpleNamespace(env=SimpleNamespace(envs=wrapped))


def tag(episode, runner, env_index):
    tag_episode_with_env_idx(
        episode=episode, env_runner=runner, env=None, env_index=env_index
    )


@pytest.mark.unit
def test_tag_rewrites_episode_id_and_sets_env_id():
    sub_env = FakeSubEnv(mechanism_id=2, seed=7, policy_seed=11)
    episode = SimpleNamespace(id_="abc123")

    tag(episode, make_runner(FakeSubEnv(), sub_env), env_index=1)

    assert episode.id_ == "env=1|m=2|ps=11|ss=7|raw=abc123"
    assert sub_env.env_id == 1


@pytest.mark.unit
def test_tag_leaves_already_tagged_ids_untouched():
    sub_env = FakeSubEnv(env_id=0)
    episode = SimpleNamespace(id_="env=0|m=0|ps=1|ss=1|raw=x")

    tag(episode, make_runner(sub_env), env_index=0)

    assert episode.id_ == "env=0|m=0|ps=1|ss=1|raw=x"


@pytest.mark.unit
def test_tag_accepts_same_env_id_on_later_episodes():
    sub_env = FakeSubEnv(env_id=3)
    runner = make_runner(FakeSubEnv(), FakeSubEnv(), FakeSubEnv(), sub_env)
    episode = SimpleNamespace(id_="raw")

    tag(episode, runner, env_index=3)

    assert episode.id_.startswith("env=3|")


@pytest.mark.unit
def test_tag_raises_when_env_id_changes():
    sub_env = FakeSubEnv(env_id=0)
    runner = make_runner(FakeSubEnv(), sub_env)
    with pytest.raises(RuntimeError, match="Immutable env_id changed"):
        tag(SimpleNamespace(id_="raw"), runner, env_index=1)


@pytest.mark.unit
def test_tag_raises_without_mechanism_id():
    sub_env = FakeSubEnv(mechanism_id=None)
    with pytest.raises(RuntimeError, match="no mechanism_id"):
        tag(SimpleNamespace(id_="raw"), make_runner(sub_env), env_index=0)


@pytest.mark.unit
def test_tag_raises_without_seed():
    sub_env = FakeSubEnv(seed=None)
    with pytest.raises(RuntimeError, match="no seed"):
        tag(SimpleNamespace(id_="raw"), make_runner(sub_env), env_index=0)
