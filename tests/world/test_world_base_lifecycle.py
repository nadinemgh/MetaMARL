"""Life cycle of a mechanism candidate stored in the ``World``.

A candidate is published by the regulator, fetched once for training, then
fetched for evaluation. The tests pin down the transition table of
``World.get_mechanism_by_id`` exactly as it is today:

- ``train`` accepts only a ``published`` entry, so a second training fetch of
  the same ``(index, seed)`` returns ``None``;
- ``eval`` accepts both a ``train`` and an ``eval`` entry, so evaluation can be
  fetched repeatedly.

The asymmetry between the two is intentional: since commit ``b330a82`` the
environment keeps the last candidate it fetched, so ``None`` means "keep what
you have". The legacy accessors ``get_mechanism`` and ``try_get_mechanism`` and
``flush`` are covered here too. The ``World`` is the plain class
behind the Ray actor, so no Ray runtime is involved.
"""

from __future__ import annotations

import pytest

from core.world.context import MechanismStatus

# Statuses a fetch in the given mode may start from (the current table).
VALID_PREDECESSORS = {
    MechanismStatus.train: {MechanismStatus.published},
    MechanismStatus.eval: {MechanismStatus.train, MechanismStatus.eval},
}


@pytest.mark.unit
@pytest.mark.parametrize("mode", [MechanismStatus.train, MechanismStatus.eval])
@pytest.mark.parametrize("start", list(MechanismStatus))
def test_get_mechanism_by_id_transition_table(
    world, make_context, make_mechanism, start, mode
):
    payload = make_mechanism(index=2, seed=5, status=start)
    world.append_context(make_context(payload))

    fetched = world.get_mechanism_by_id(2, 5, mode)

    if start in VALID_PREDECESSORS[mode]:
        assert fetched is payload
        assert payload.status == mode
    else:
        assert fetched is None
        assert payload.status == start


@pytest.mark.unit
def test_candidate_is_handed_out_once_for_training_then_reusable_for_eval(
    world, make_context, make_mechanism
):
    payload = make_mechanism(index=1, seed=10)
    world.append_context(make_context(payload))

    assert world.get_mechanism_by_id(1, 10, MechanismStatus.eval) is None
    assert payload.status == MechanismStatus.published

    assert world.get_mechanism_by_id(1, 10, MechanismStatus.train) is payload
    assert payload.status == MechanismStatus.train
    assert world.get_mechanism_by_id(1, 10, MechanismStatus.train) is None

    assert world.get_mechanism_by_id(1, 10, MechanismStatus.eval) is payload
    assert payload.status == MechanismStatus.eval
    assert world.get_mechanism_by_id(1, 10, MechanismStatus.eval) is payload
    assert world.get_mechanism_by_id(1, 10, MechanismStatus.train) is None
    assert payload.status == MechanismStatus.eval


@pytest.mark.unit
def test_get_mechanism_by_id_matches_index_and_seed(
    world, make_context, make_mechanism
):
    first = make_mechanism(index=1, seed=10)
    second = make_mechanism(index=1, seed=20)
    third = make_mechanism(index=2, seed=10)

    for payload in (first, second, third):
        world.append_context(make_context(payload))

    assert world.get_mechanism_by_id(1, 99, MechanismStatus.train) is None
    assert world.get_mechanism_by_id(5, 10, MechanismStatus.train) is None
    assert world.get_mechanism_by_id(1, 20, MechanismStatus.train) is second

    assert first.status == MechanismStatus.published
    assert third.status == MechanismStatus.published
    assert second.status == MechanismStatus.train


@pytest.mark.unit
def test_get_mechanism_by_id_returns_none_on_an_empty_registry(world):
    assert world.get_mechanism_by_id(0, 0, MechanismStatus.train) is None
    assert world.get_mechanism_by_id(0, 0, MechanismStatus.eval) is None


@pytest.mark.unit
def test_get_mechanism_by_id_prefers_the_first_registered_duplicate(
    world, make_context, make_mechanism
):
    """Two entries with the same ``(index, seed)`` are served in insertion order."""
    older = make_mechanism(index=0, seed=0)
    newer = make_mechanism(index=0, seed=0)
    world.append_context(make_context(older))
    world.append_context(make_context(newer))

    assert world.get_mechanism_by_id(0, 0, MechanismStatus.train) is older
    assert world.get_mechanism_by_id(0, 0, MechanismStatus.train) is newer
    assert world.get_mechanism_by_id(0, 0, MechanismStatus.train) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode",
    [
        MechanismStatus.init,
        MechanismStatus.published,
        MechanismStatus.assigned,
        MechanismStatus.done,
    ],
)
def test_get_mechanism_by_id_rejects_a_mode_that_is_not_train_or_eval(
    world, make_context, make_mechanism, mode
):
    payload = make_mechanism(index=0, seed=0)
    world.append_context(make_context(payload))

    with pytest.raises(TypeError):
        world.get_mechanism_by_id(0, 0, mode)

    assert payload.status == MechanismStatus.published


@pytest.mark.unit
def test_get_mechanism_claims_published_entries_in_order(
    world, make_context, make_mechanism
):
    world.append_context(
        make_context(make_mechanism(index=0, status=MechanismStatus.train))
    )
    world.append_context(make_context(make_mechanism(index=1)))
    world.append_context(make_context(make_mechanism(index=2)))

    first = world.get_mechanism()
    assert (first.index, first.status) == (1, MechanismStatus.assigned)

    second = world.get_mechanism()
    assert (second.index, second.status) == (2, MechanismStatus.assigned)

    with pytest.raises(RuntimeError, match="no available mechanisms to train"):
        world.get_mechanism()


@pytest.mark.unit
def test_try_get_mechanism_returns_none_when_nothing_is_published(
    world, make_context, make_mechanism
):
    assert world.try_get_mechanism() is None

    world.append_context(make_context(make_mechanism(index=4)))
    claimed = world.try_get_mechanism()

    assert (claimed.index, claimed.status) == (4, MechanismStatus.assigned)
    assert world.try_get_mechanism() is None


@pytest.mark.unit
def test_assigned_entry_cannot_be_fetched_by_id(world, make_context, make_mechanism):
    """The legacy claim leaves ``assigned``, which no fetch mode accepts."""
    payload = make_mechanism(index=3, seed=1)
    world.append_context(make_context(payload))
    world.try_get_mechanism()

    assert world.get_mechanism_by_id(3, 1, MechanismStatus.train) is None
    assert world.get_mechanism_by_id(3, 1, MechanismStatus.eval) is None


@pytest.mark.unit
def test_get_mechanism_by_index_is_keyed_by_context_id(
    world, make_context, make_mechanism
):
    payload = make_mechanism(index=0)
    cid = world.append_context(make_context(payload))

    assert world.get_mechanism_by_index(cid) is payload

    with pytest.raises(KeyError):
        world.get_mechanism_by_index(0)


@pytest.mark.unit
def test_flush_by_status_keeps_the_other_entries(world, make_context, make_mechanism):
    published = world.append_context(make_context(make_mechanism(index=0)))
    evaluated = world.append_context(
        make_context(make_mechanism(index=1, status=MechanismStatus.eval))
    )
    trained = world.append_context(
        make_context(make_mechanism(index=2, status=MechanismStatus.train))
    )

    world.flush(status=MechanismStatus.eval)

    assert set(world.get_mechanism_registry()) == {published, trained}
    assert world.get_ctx_ids() == {published, trained}
    assert world.get_context(evaluated) is None
    assert world.get_opt_ctx_ids("opt") == [published, trained]


@pytest.mark.unit
def test_flush_without_status_empties_every_registry_but_keeps_the_optimizer(
    world, make_context, make_mechanism
):
    for i in range(3):
        world.append_context(make_context(make_mechanism(index=i)))

    world.flush()

    assert world.get_mechanism_registry() == {}
    assert world.get_ctx_ids() == set()
    # The optimizer ID stays registered: ``build_optimizer`` reserved it and
    # ``generate_uuid`` must not hand it out again.
    assert world.get_opt_ids() == {"opt"}
    assert world.get_opt_ctx_ids("opt") == []


@pytest.mark.unit
def test_flush_keeps_every_listed_context_id_resolvable(
    world, make_context, make_mechanism, make_other
):
    """The three registries agree after a partial flush."""
    world.append_context(make_context(make_other(), opt_id="a"))
    world.append_context(
        make_context(make_mechanism(index=0, status=MechanismStatus.eval), opt_id="a")
    )
    world.append_context(make_context(make_mechanism(index=1), opt_id="b"))
    world.append_context(
        make_context(make_mechanism(index=2, status=MechanismStatus.eval), opt_id="b")
    )
    world.append_context(
        make_context(make_mechanism(index=3, status=MechanismStatus.eval), opt_id=None)
    )

    world.flush(status=MechanismStatus.eval)

    listed = [cid for opt in world.get_opt_ids() for cid in world.get_opt_ctx_ids(opt)]

    assert all(world.get_context(cid) is not None for cid in listed)
    assert set(world.get_mechanism_registry()) <= world.get_ctx_ids()
    assert len(world.get_ctx_ids()) == 2


@pytest.mark.unit
def test_flushed_evaluation_entries_are_no_longer_fetchable(
    world, make_context, make_mechanism
):
    world.append_context(make_context(make_mechanism(index=0, seed=0)))
    world.get_mechanism_by_id(0, 0, MechanismStatus.train)
    assert world.get_mechanism_by_id(0, 0, MechanismStatus.eval) is not None

    world.flush(status=MechanismStatus.eval)

    assert world.get_mechanism_by_id(0, 0, MechanismStatus.eval) is None
