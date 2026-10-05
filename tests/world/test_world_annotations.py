"""Annotations of the ``World`` and ``Context`` API against what the code does.

The registry methods of ``World`` return values that the annotations must be
able to describe: ``get_mechanism_by_id`` returns ``None`` when it has nothing
new to hand out, and the mechanism registry is keyed by ``ContextID`` strings.
``Context.opt_id`` may be ``None`` for a context that no optimizer owns. The
envelope is a plain dataclass, not a pydantic model, so widening that
annotation cannot change what it accepts: the tests pin down that nothing is
validated, with ``None`` and with a value of the wrong type.
"""

from __future__ import annotations

import dataclasses
import typing

import pytest
from pydantic import BaseModel

from core.types import ContextID, OptimizerID
from core.world.context import Context, MechanismContext, MechanismStatus
from tests.world.conftest import WORLD_CLASS, OtherSchema


@pytest.mark.unit
def test_get_mechanism_by_id_is_annotated_as_optional():
    hints = typing.get_type_hints(WORLD_CLASS.get_mechanism_by_id)

    assert hints["return"] == typing.Optional[MechanismContext]


@pytest.mark.unit
def test_get_mechanism_registry_is_keyed_by_context_id():
    hints = typing.get_type_hints(WORLD_CLASS.get_mechanism_registry)

    assert hints["return"] == dict[ContextID, MechanismContext]


@pytest.mark.unit
def test_context_opt_id_is_annotated_as_optional():
    assert typing.get_type_hints(Context)["opt_id"] == typing.Optional[OptimizerID]


@pytest.mark.unit
def test_context_is_a_plain_dataclass_without_validation():
    assert dataclasses.is_dataclass(Context)
    assert not issubclass(Context, BaseModel)

    payload = OtherSchema()
    without_owner = Context(id=None, opt_id=None, step=0, env="e", payload=payload)
    wrong_type = Context(id=None, opt_id=123, step="x", env="e", payload=payload)

    assert without_owner.opt_id is None
    assert wrong_type.opt_id == 123


@pytest.mark.unit
def test_a_context_without_owner_is_stored_without_an_optimizer_entry(world):
    ctx = Context(
        id=None,
        opt_id=None,
        step=0,
        env="e",
        payload=MechanismContext(
            index=0,
            env_id=None,
            seed=0,
            status=MechanismStatus.published,
            mechanism=None,
            metrics=None,
        ),
    )

    cid = world.append_context(ctx)

    assert world.get_context(cid) is ctx
    assert world.get_opt_ids() == set()
