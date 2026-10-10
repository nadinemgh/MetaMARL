"""Context registration and lookup in the ``World``.

Three registries must stay consistent: the context registry, the mechanism
registry (keyed by the ``ContextID`` of the carrying context) and the
optimizer-to-contexts map. The tests exercise the public accessors and
``append_context`` on the plain class behind the Ray actor, including its error
paths (singleton violation, duplicate IDs). A rejected registration must leave
every registry as it was.
"""

from __future__ import annotations

import copy

import pytest

# --------------------------------------------------------------------------- #
# Construction and trivial accessors
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_new_world_has_empty_registries(world):
    assert world.get_mechanism_registry() == {}
    assert list(world.get_opt_registry()) == []
    assert world.get_ctx_ids() == set()
    assert world.get_opt_ids() == set()
    assert world.get_context("missing") is None
    assert world.get_opt_ctx_ids("missing") == []


@pytest.mark.unit
def test_copy_and_deepcopy_return_the_same_instance(world):
    assert copy.copy(world) is world
    assert copy.deepcopy(world) is world


@pytest.mark.unit
def test_get_opt_ctx_ids_returns_a_copy(world, make_context, make_other):
    cid = world.append_context(make_context(make_other()))

    listing = world.get_opt_ctx_ids("opt")
    listing.append("intruder")

    assert world.get_opt_ctx_ids("opt") == [cid]


# --------------------------------------------------------------------------- #
# append_context
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_append_context_registers_mechanism_and_optimizer(
    world, make_context, make_mechanism
):
    ctx = make_context(make_mechanism(index=3, seed=7))

    cid = world.append_context(ctx)

    assert ctx.id == cid
    assert world.get_context(cid) is ctx
    assert world.get_ctx_ids() == {cid}
    assert world.get_mechanism_registry() == {cid: ctx.payload}
    assert world.get_opt_ctx_ids("opt") == [cid]
    assert world.get_opt_ids() == {"opt"}
    assert "opt" in world.get_opt_registry()
    assert world.get_mechanism_by_index(3) is ctx.payload


@pytest.mark.unit
def test_append_context_accepts_a_mechanism_without_env_id(
    world, make_context, make_mechanism
):
    """The regulator publishes candidates with ``env_id=None``."""
    ctx = make_context(make_mechanism(env_id=None))

    cid = world.append_context(ctx)

    assert world.get_mechanism_registry() == {cid: ctx.payload}


@pytest.mark.unit
def test_append_context_without_optimizer_and_with_plain_payload(
    world, make_context, make_other
):
    ctx = make_context(make_other(), opt_id=None)

    cid = world.append_context(ctx)

    assert world.get_context(cid) is ctx
    assert world.get_mechanism_registry() == {}
    assert world.get_opt_ids() == set()


@pytest.mark.unit
def test_append_context_overwrites_a_caller_supplied_id(
    world, make_context, make_other
):
    ctx = make_context(make_other(), ctx_id="custom")

    cid = world.append_context(ctx)

    assert cid != "custom"
    assert ctx.id == cid
    assert world.get_context("custom") is None


@pytest.mark.unit
def test_append_context_rejects_an_id_that_already_exists(
    world, make_context, make_other
):
    cid = world.append_context(make_context(make_other()))

    with pytest.raises(ValueError, match="already exists"):
        world.append_context(make_context(make_other(), ctx_id=cid))

    assert world.get_ctx_ids() == {cid}


@pytest.mark.unit
def test_append_context_singleton_is_enforced_per_payload_type(
    world, make_context, make_other, make_step
):
    world.append_context(make_context(make_other()), singleton=True)

    with pytest.raises(ValueError, match="Singleton Context Schema OtherSchema"):
        world.append_context(make_context(make_other()), singleton=True)

    world.append_context(make_context(make_step()), singleton=True)
    assert len(world.get_ctx_ids()) == 2


@pytest.mark.unit
def test_singleton_check_compares_exact_payload_types(world, make_context, make_other):
    child_type = type("ChildSchema", (type(make_other()),), {})
    world.append_context(make_context(child_type()))

    # A parent schema is not considered present when only a child exists.
    world._validate_ctx_schema_exists(type(make_other()))

    with pytest.raises(ValueError, match="ChildSchema"):
        world._validate_ctx_schema_exists(child_type)


# --------------------------------------------------------------------------- #
# _set_new_opt_id
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_set_new_opt_id_is_idempotent_and_keeps_existing_contexts(
    world, make_context, make_other
):
    assert world._set_new_opt_id("reg") == "reg"
    assert world.get_opt_ctx_ids("reg") == []

    world.append_context(make_context(make_other(), opt_id="reg"))

    assert world._set_new_opt_id("reg") == "reg"
    assert len(world.get_opt_ctx_ids("reg")) == 1


@pytest.mark.unit
def test_set_new_opt_id_without_id_draws_distinct_identifiers(world):
    first = world._set_new_opt_id(None)
    second = world._set_new_opt_id(None)

    assert first != second
    assert world.get_opt_ids() == {first, second}
