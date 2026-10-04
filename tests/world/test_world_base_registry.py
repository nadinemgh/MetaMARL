"""Context registration, lookup and removal in the ``World``.

Three registries must stay consistent: the context registry, the mechanism
registry (keyed by the ``ContextID`` of the carrying context) and the
optimizer-to-contexts map. The tests exercise the public accessors and mutators
of the plain class behind the Ray actor, including the error paths (singleton
violation, duplicate IDs, missing ``env_id``, unknown context on update) and the
three env-step accessors used by the reduced-env plotting.

Four tests are expected failures (``xfail``) that record suspected defects:
a rejected context stays registered, and the cursor of
``get_new_env_step_contexts`` goes stale once contexts are removed.
``flush_ctx``, ``remove_context`` and ``get_new_env_step_contexts`` have no
caller inside ``core`` today.
"""

from __future__ import annotations

import copy

import pytest

# --------------------------------------------------------------------------- #
# Construction and trivial accessors
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_new_world_has_empty_registries(world):
    assert world.get_ctx_registry() == {}
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
    assert world.get_mechanism_by_index(cid) is ctx.payload


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


# --------------------------------------------------------------------------- #
# set_new_context
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_set_new_context_generates_an_id_and_tracks_the_mechanism(
    world, make_context, make_mechanism
):
    ctx = make_context(make_mechanism(env_id="env-0"))

    cid = world.set_new_context(ctx)

    assert ctx.id == cid
    assert world.get_mechanism_registry() == {cid: ctx.payload}
    assert world.get_opt_ctx_ids("opt") == [cid]


@pytest.mark.unit
def test_set_new_context_appends_to_an_existing_optimizer(
    world, make_context, make_other
):
    first = world.set_new_context(make_context(make_other(), opt_id="opt"))
    second = world.set_new_context(make_context(make_other(), opt_id="opt"))

    assert world.get_opt_ctx_ids("opt") == [first, second]
    assert world.get_opt_ids() == {"opt"}


@pytest.mark.unit
def test_set_new_context_keeps_the_caller_id_and_rejects_duplicates(
    world, make_context, make_other
):
    ctx = make_context(make_other(), opt_id=None, ctx_id="keep")

    assert world.set_new_context(ctx) == "keep"
    assert world.get_context("keep") is ctx

    with pytest.raises(ValueError, match="ContextID 'keep' already exists"):
        world.set_new_context(make_context(make_other(), opt_id=None, ctx_id="keep"))


@pytest.mark.unit
def test_set_new_context_singleton_violation(world, make_context, make_other):
    world.set_new_context(make_context(make_other()), singleton=True)

    with pytest.raises(ValueError, match="Singleton"):
        world.set_new_context(make_context(make_other()), singleton=True)


@pytest.mark.unit
def test_set_new_context_rejects_a_mechanism_without_env_id(
    world, make_context, make_mechanism
):
    with pytest.raises(ValueError, match="MechanismContext must include env_id"):
        world.set_new_context(make_context(make_mechanism(env_id=None)))


@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason="set_new_context stores the context in _contexts before it rejects "
    + "a MechanismContext without env_id, so a failed call still mutates the World",
)
def test_set_new_context_leaves_the_world_untouched_when_it_raises(
    world, make_context, make_mechanism
):
    with pytest.raises(ValueError):
        world.set_new_context(make_context(make_mechanism(env_id=None)))

    assert world.get_ctx_ids() == set()
    assert world.get_mechanism_registry() == {}


# --------------------------------------------------------------------------- #
# update_context
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_update_context_replaces_the_payload(
    world, make_context, make_mechanism, make_other
):
    cid = world.set_new_context(make_context(make_mechanism(env_id="env-0")))

    replacement = make_context(
        make_mechanism(index=9, env_id="env-1"), step=1, ctx_id=cid
    )
    world.update_context(replacement)

    assert world.get_context(cid) is replacement
    assert world.get_mechanism_registry()[cid].index == 9

    plain = make_context(make_other(), step=2, ctx_id=cid)
    world.update_context(plain)

    assert world.get_context(cid) is plain


@pytest.mark.unit
def test_update_context_rejects_an_unknown_context(world, make_context, make_other):
    with pytest.raises(KeyError, match="not registered"):
        world.update_context(make_context(make_other()))


@pytest.mark.unit
def test_update_context_rejects_a_mechanism_without_env_id(
    world, make_context, make_mechanism
):
    cid = world.set_new_context(make_context(make_mechanism(env_id="env-0")))

    with pytest.raises(ValueError, match="MechanismContext must include env_id"):
        world.update_context(make_context(make_mechanism(env_id=None), ctx_id=cid))


@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason="update_context replaces the stored context before it rejects a "
    + "MechanismContext without env_id, leaving the registries out of sync",
)
def test_update_context_keeps_the_old_context_when_it_raises(
    world, make_context, make_mechanism
):
    original = make_context(make_mechanism(index=1, env_id="env-0"))
    cid = world.set_new_context(original)

    with pytest.raises(ValueError):
        world.update_context(
            make_context(make_mechanism(index=2, env_id=None), ctx_id=cid)
        )

    assert world.get_context(cid) is original
    assert world.get_mechanism_registry()[cid] is original.payload


# --------------------------------------------------------------------------- #
# remove_context
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_remove_context_cleans_context_and_optimizer_map(
    world, make_context, make_other
):
    first = make_context(make_other())
    second = make_context(make_other())
    world.append_context(first)
    world.append_context(second)

    world.remove_context(first)

    assert first.id not in world.get_ctx_ids()
    assert world.get_opt_ctx_ids("opt") == [second.id]

    world.remove_context(second)

    assert world.get_opt_ids() == set()


@pytest.mark.unit
def test_remove_context_ignores_unknown_contexts(world, make_context, make_other):
    kept = make_context(make_other())
    world.append_context(kept)

    world.remove_context(make_context(make_other(), opt_id="nobody", ctx_id="x"))
    world.remove_context(make_context(make_other(), opt_id="opt", ctx_id="ghost"))

    assert world.get_opt_ctx_ids("opt") == [kept.id]
    assert world.get_ctx_ids() == {kept.id}


# --------------------------------------------------------------------------- #
# Env-step accessors
# --------------------------------------------------------------------------- #


@pytest.fixture
def populated(world, make_context, make_step, make_mechanism):
    """Optimizer ``a`` runs two episodes with a mechanism between them; ``b`` one."""
    ids = {}
    ids["a0"] = world.append_context(make_context(make_step(0), opt_id="a", step=0))
    ids["a1"] = world.append_context(make_context(make_step(1), opt_id="a", step=1))
    ids["mech"] = world.append_context(make_context(make_mechanism(), opt_id="a"))
    ids["a0b"] = world.append_context(make_context(make_step(0), opt_id="a", step=0))
    ids["a1b"] = world.append_context(make_context(make_step(1), opt_id="a", step=1))
    ids["b0"] = world.append_context(make_context(make_step(0), opt_id="b", step=0))

    return world, ids


@pytest.mark.unit
def test_get_env_step_contexts_filters_by_optimizer_in_insertion_order(populated):
    world, ids = populated

    assert [c.id for c in world.get_env_step_contexts()] == [
        ids["a0"],
        ids["a1"],
        ids["a0b"],
        ids["a1b"],
        ids["b0"],
    ]
    assert [c.id for c in world.get_env_step_contexts("a")] == [
        ids["a0"],
        ids["a1"],
        ids["a0b"],
        ids["a1b"],
    ]
    assert world.get_env_step_contexts("unknown") == []


@pytest.mark.unit
def test_get_env_step_contexts_skips_flushed_ids(populated):
    world, ids = populated
    world.flush_ctx([ids["a1b"]])

    assert [c.id for c in world.get_env_step_contexts("a")] == [
        ids["a0"],
        ids["a1"],
        ids["a0b"],
    ]


@pytest.mark.unit
def test_get_latest_env_step_contexts_returns_the_last_episode(populated):
    world, ids = populated

    assert [c.id for c in world.get_latest_env_step_contexts("a")] == [
        ids["a0b"],
        ids["a1b"],
    ]
    # Without an optimizer the most recent episode is the one of ``b``.
    assert [c.id for c in world.get_latest_env_step_contexts()] == [ids["b0"]]
    assert world.get_latest_env_step_contexts("none") == []


@pytest.mark.unit
def test_get_latest_env_step_contexts_without_a_reset_returns_everything(
    world, make_context, make_step
):
    world.append_context(make_context(make_step(1), step=1))
    world.append_context(make_context(make_step(2), step=2))

    assert len(world.get_latest_env_step_contexts("opt")) == 2


@pytest.mark.unit
def test_get_new_env_step_contexts_advances_a_cursor_per_optimizer(
    populated, make_context, make_step
):
    world, ids = populated

    first = world.get_new_env_step_contexts("a")
    assert [c.id for c in first] == [ids["a0"], ids["a1"], ids["a0b"], ids["a1b"]]
    assert world.get_new_env_step_contexts("a") == []

    new_id = world.append_context(make_context(make_step(0), opt_id="a", step=0))
    assert [c.id for c in world.get_new_env_step_contexts("a")] == [new_id]

    # The global cursor (no optimizer) is independent of the per-optimizer one.
    assert len(world.get_new_env_step_contexts()) == 6
    assert world.get_new_env_step_contexts() == []


@pytest.mark.unit
def test_get_new_env_step_contexts_skips_flushed_contexts(populated):
    world, ids = populated
    world.flush_ctx([ids["a0"]])

    assert [c.id for c in world.get_new_env_step_contexts("a")] == [
        ids["a1"],
        ids["a0b"],
        ids["a1b"],
    ]


@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason="the global cursor counts positions in _contexts, which flush_ctx "
    + "shrinks, so contexts appended after a flush are skipped",
)
def test_global_cursor_survives_a_flush_of_all_contexts(world, make_context, make_step):
    for step in range(3):
        world.append_context(make_context(make_step(step), opt_id=None, step=step))

    assert len(world.get_new_env_step_contexts()) == 3

    world.flush_ctx(list(world.get_ctx_ids()))
    fresh = [
        world.append_context(make_context(make_step(step), opt_id=None, step=step))
        for step in range(2)
    ]

    assert [c.id for c in world.get_new_env_step_contexts()] == fresh


@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason="the per-optimizer cursor counts positions in a list that "
    + "remove_context shrinks, so contexts appended afterwards are skipped",
)
def test_optimizer_cursor_survives_remove_context(world, make_context, make_step):
    contexts = [
        make_context(make_step(step), opt_id="a", step=step) for step in range(3)
    ]

    for ctx in contexts:
        world.append_context(ctx)

    assert len(world.get_new_env_step_contexts("a")) == 3

    for ctx in contexts:
        world.remove_context(ctx)

    fresh = world.append_context(make_context(make_step(0), opt_id="a", step=0))

    assert [c.id for c in world.get_new_env_step_contexts("a")] == [fresh]
