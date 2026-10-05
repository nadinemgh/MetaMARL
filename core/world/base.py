"""Ray actor holding the shared state of a bilevel optimisation run.

The ``World`` is the blackboard both levels of the optimisation talk to. The
outer regulator publishes one ``MechanismContext`` per candidate and training
seed, and the inner RLlib environments (living in env-runner processes) fetch
their candidate from it at every episode reset. Regulators also append the
``done`` contexts that record the aggregated fitness of a candidate, and the
inner optimizer flushes the evaluated candidates after each evaluation pass.
Because ``World`` is a Ray actor, every method is called as
``world.<method>.remote(...)`` and its return value crosses a Ray boundary;
callers must not rely on mutating a returned object to update the World.

Three registries are kept in sync: ``_contexts`` (all contexts by
``ContextID``), ``_mechanism_registry`` (the ``MechanismContext`` payloads,
keyed by the ``ContextID`` of their context, not by mechanism index) and
``_opt_ctx_map`` (context IDs owned by each optimizer).

Examples
--------
The decorated class cannot be instantiated without a Ray runtime, so the
examples use the plain class stored in ``World.__ray_metadata__.modified_class``;
with a running actor every call below becomes ``ray.get(world.<method>.remote())``.

>>> from core.world.context import Context, MechanismContext, MechanismStatus
>>> world = World.__ray_metadata__.modified_class()
>>> payload = MechanismContext(
...     index=0,
...     env_id=None,
...     seed=7,
...     status=MechanismStatus.published,
...     mechanism={"quota": 0.3},
...     metrics=None,
... )
>>> ctx = Context(id=None, opt_id="opt", step=0, env="regulator", payload=payload)
>>> ctx_id = world.append_context(ctx)
>>> world.get_opt_ctx_ids("opt") == [ctx_id]
True
>>> world.get_mechanism_by_id(0, 7, MechanismStatus.train).mechanism
{'quota': 0.3}
>>> world.get_mechanism_by_id(0, 7, MechanismStatus.train) is None
True
>>> world.flush(status=MechanismStatus.train)
>>> world.get_mechanism_registry()
{}
"""

from __future__ import annotations

from typing import TYPE_CHECKING, KeysView, Optional

import ray

from core.types import ContextID, OptimizerID
from core.utils import generate_uuid
from core.world.context import Context, ContextSchema, MechanismContext, MechanismStatus

if TYPE_CHECKING:
    pass


@ray.remote
class World:
    """Shared runtime container for optimizer-produced contexts.

    The World is the single place where the regulator and the inner
    environments exchange mechanism candidates. It keeps three registries (all
    contexts by ID, mechanism payloads by the ID of the context that carried
    them, and the context IDs owned by each optimizer). It enforces one global
    constraint: a context whose payload type must be unique can be registered
    as a singleton. The World does not own optimizers or environments; it only
    tracks identifiers and context payloads.

    Because the class is decorated with ``@ray.remote``, instances are actors:
    arguments and return values are serialised across the actor boundary, and
    ``copy.copy`` / ``copy.deepcopy`` of an instance return the same object.

    The constructor takes no argument; every registry starts empty.

    When to use: create exactly one World per run, before the outer
    optimizer and the inner optimizer are built, and hand the actor handle to
    both so that candidates published by the regulator can be fetched by the
    inner environments and optimizer IDs stay unique.

    Examples
    --------
    >>> import ray
    >>> world = World.remote()  # doctest: +SKIP
    >>> ray.get(world.get_opt_ids.remote())  # doctest: +SKIP
    set()
    """

    def __init__(self):
        # Maps optimizer IDs to the list of context IDs they own
        self._opt_ctx_map: dict[OptimizerID, list[ContextID]] = {}

        # Maps context IDs to Context objects
        self._contexts: dict[ContextID, Context] = {}

        # Mechanism registry
        self._mechanism_registry: dict[int, MechanismContext] = {}

    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self

    # Accessors
    def get_mechanism_registry(self) -> dict[int, MechanismContext]:
        """Return the mechanism registry.

        Returns
        -------
        dict[ContextID, MechanismContext]
            Mechanism payloads keyed by the ``ContextID`` of the context that
            carried them. Despite the ``int`` annotation, keys are UUID
            strings; the mechanism's batch position lives in
            ``MechanismContext.index``.
        """

        return self._mechanism_registry

    def get_opt_registry(self) -> KeysView[OptimizerID]:
        """Return a view of the optimizer IDs known to the world.

        Returns
        -------
        KeysView[OptimizerID]
            Keys of the optimizer-to-contexts map. The ``build_optimizer``
            methods of ``OptimizerConfig`` and ``RayOptimizerConfig`` pass it to
            ``generate_uuid`` to draw a fresh, unused optimizer ID.
        """

        return self._opt_ctx_map.keys()

    def get_context(self, ctx_id: ContextID) -> Context | None:
        """Return the context stored under an ID.

        Parameters
        ----------
        ctx_id : ContextID
            Identifier of the context, as returned by ``append_context``.

        Returns
        -------
        Context or None
            The registered context, or ``None`` if the ID is unknown (including
            an ID whose context was removed).
        """

        return self._contexts.get(ctx_id, None)

    def get_opt_ctx_ids(self, opt_id: OptimizerID) -> list[ContextID]:
        """Return the IDs of the contexts registered under an optimizer.

        Parameters
        ----------
        opt_id : OptimizerID
            Identifier of the optimizer.

        Returns
        -------
        list[ContextID]
            A copy of the optimizer's ID list, in registration order; empty if
            the optimizer is unknown.
        """

        return list(self._opt_ctx_map.get(opt_id, []))

    def get_ctx_ids(self) -> set[ContextID]:
        """Return the IDs of every context in the context registry.

        Returns
        -------
        set[ContextID]
            A new set holding the keys of ``_contexts``.
        """

        return set(self._contexts.keys())

    def get_opt_ids(self) -> set[OptimizerID]:
        """Return the IDs of every optimizer known to the World.

        Returns
        -------
        set[OptimizerID]
            A new set holding the keys of the optimizer-to-contexts map.
        """

        return set(self._opt_ctx_map.keys())

    def get_mechanism(self) -> MechanismContext:
        """Claim the first published mechanism, regardless of index or seed.

        Legacy accessor: the first entry with status ``published`` is moved to
        ``assigned`` and returned. The current environments use
        ``get_mechanism_by_id`` instead, which matches index and seed. An entry
        in status ``assigned`` is not accepted by any fetch mode of
        ``get_mechanism_by_id``.

        Returns
        -------
        MechanismContext
            The claimed mechanism, now in status ``assigned``.

        Raises
        ------
        RuntimeError
            If no mechanism is in status ``published``.
        """

        for m_ctx in self._mechanism_registry.values():
            if m_ctx.status == MechanismStatus.published:
                m_ctx.status = MechanismStatus.assigned

                return m_ctx

        raise RuntimeError("no available mechanisms to train")

    # Use the contextID as mechanismID
    def get_mechanism_by_id(
        self, mechanism_id: int, seed: int, mode: MechanismStatus
    ) -> MechanismContext:
        """Fetch the mechanism for ``(mechanism_id, seed)`` and advance its status.

        Called by ``MultiAgentEnv.reset`` (``core.envs.marl_regulated``) at the
        start of every episode. The first registry entry whose ``index`` equals
        ``mechanism_id``, whose ``seed`` equals ``seed`` and whose status is a
        valid predecessor of ``mode`` is switched to ``mode`` and returned. Valid
        predecessors are ``published`` for ``mode=train`` and ``{train, eval}``
        for ``mode=eval``.

        Parameters
        ----------
        mechanism_id : int
            Batch position of the candidate (``MechanismContext.index``), not
            a registry key.
        seed : int
            Policy seed the environment was built with
            (``MechanismContext.seed``).
        mode : MechanismStatus
            Target status: ``MechanismStatus.train`` or ``MechanismStatus.eval``.

        Returns
        -------
        MechanismContext or None
            The matching mechanism on the first successful fetch. On later
            calls with the same arguments the entry is no longer in a
            predecessor status, so ``None`` is returned; environments treat
            that as "keep the mechanism you already have". The return
            annotation does not mention ``None`` but it is a routine outcome.

        Raises
        ------
        TypeError
            If ``mode`` is neither ``train`` nor ``eval`` and a registry entry
            matches ``mechanism_id`` and ``seed``: the predecessor lookup yields
            ``None`` and the ``in`` test fails. With no matching entry the call
            returns ``None`` whatever ``mode`` is, and no status is changed.
        """

        required_status = {
            MechanismStatus.train: {MechanismStatus.published},
            MechanismStatus.eval: {MechanismStatus.train, MechanismStatus.eval},
        }
        target_prev_status = required_status.get(mode)

        for m_ctx in self._mechanism_registry.values():
            if (
                mechanism_id == m_ctx.index
                and seed == m_ctx.seed
                and m_ctx.status in target_prev_status
            ):
                m_ctx.status = mode

                return m_ctx

        return None

    def try_get_mechanism(self) -> MechanismContext | None:
        """Claim the first published mechanism, or return ``None``.

        Non-raising variant of ``get_mechanism``: the first entry in status
        ``published`` is moved to ``assigned`` and returned, regardless of index
        or seed.

        Returns
        -------
        MechanismContext or None
            The claimed mechanism, or ``None`` if no entry is ``published``.
        """

        for m_ctx in self._mechanism_registry.values():
            if m_ctx.status == MechanismStatus.published:
                m_ctx.status = MechanismStatus.assigned

                return m_ctx

        return None

    def get_mechanism_by_index(self, index: int) -> MechanismContext:
        """Return the mechanism stored under registry key ``index``.

        Despite the name and the ``int`` annotation, the registry is keyed by
        the ``ContextID`` string of the publishing context, so this only works
        when passed that ID; an integer batch index raises ``KeyError``. The
        mismatch is recorded in ``TODO.md``. Use ``get_mechanism_by_id`` to look
        up a candidate by its batch position.

        Parameters
        ----------
        index : int
            Registry key. In practice the ``ContextID`` string returned by
            ``append_context``.

        Returns
        -------
        MechanismContext
            The payload stored under that key; its status is not changed.

        Raises
        ------
        KeyError
            If ``index`` is not a registry key.
        """

        return self._mechanism_registry[index]

    def _validate_ctx_schema_exists(self, schema: type[ContextSchema]) -> None:
        """Ensure a singleton ContextSchema is not already present in the world.

        The comparison is on the exact payload type (``type(payload) is
        schema``), so a subclass of ``schema`` does not count as a duplicate.
        """

        for ctx in self._contexts.values():
            if type(ctx.payload) is schema:
                raise ValueError(
                    f"Singleton Context Schema {schema.__name__} already exists in "
                    + "world."
                )

    # Mutators
    def append_context(self, ctx: Context, *, singleton: bool = False) -> ContextID:
        """Register a context, assigning it a fresh ID.

        This is the path used by ``RegulatorEnv.step`` for every candidate it
        publishes and by the regulators of the examples for the ``done``
        contexts. The context is stored in ``_contexts``; a
        ``MechanismContext`` payload is additionally indexed in the mechanism
        registry; and the ID is appended to the owning optimizer's list,
        creating that optimizer entry on the fly if needed.

        Parameters
        ----------
        ctx : Context
            Context to store. ``ctx.id`` is always overwritten with a new UUID,
            so a caller-supplied ID is not preserved; the duplicate check that
            precedes the overwrite can only fire if the caller passed an ID
            that already exists.
        singleton : bool, optional
            If ``True``, raise when another context with the same payload type
            is already registered.

        Returns
        -------
        ContextID
            The generated ID now stored in ``ctx.id``.

        Raises
        ------
        ValueError
            If ``singleton`` is requested and violated, or if a caller-supplied
            ``ctx.id`` already exists.

        Notes
        -----
        Unlike ``update_context``, this method does not require
        ``MechanismContext.env_id`` to be set, which is why regulators can
        publish candidates with ``env_id=None``. When the singleton check or
        the duplicate-ID check raises, nothing has been stored yet.
        """

        # Enforce singleton schemas if requested
        if singleton:
            self._validate_ctx_schema_exists(type(ctx.payload))

        if ctx.id is not None and ctx.id in self._contexts:
            raise ValueError(f"ContextID '{ctx.id}' already exists")

        ctx.id = generate_uuid(self._contexts)
        self._contexts[ctx.id] = ctx

        if isinstance(ctx.payload, MechanismContext):
            self._mechanism_registry[ctx.id] = ctx.payload

        # Track optimizer → context mapping
        if ctx.opt_id is not None:
            if ctx.opt_id not in self._opt_ctx_map:
                self._set_new_opt_id(ctx.opt_id)

            self._opt_ctx_map[ctx.opt_id].append(ctx.id)

        return ctx.id

    def _set_new_opt_id(self, opt_id: OptimizerID) -> OptimizerID:
        """Ensure an optimizer ID exists in the map and return it.

        Also called remotely by the ``build_optimizer`` methods to register the
        ID they drew with ``generate_uuid``.

        Parameters
        ----------
        opt_id : OptimizerID or None
            Identifier to register. ``None`` draws a fresh UUID.

        Returns
        -------
        OptimizerID
            The registered identifier. Calling twice with the same ID is a
            no-op that keeps the existing context list.
        """

        if opt_id is None:
            opt_id = generate_uuid(registry=self._opt_ctx_map.keys())

        if opt_id not in self._opt_ctx_map:
            self._opt_ctx_map[opt_id] = []

        return opt_id

    def update_context(self, ctx: Context) -> None:
        """Replace a registered context by an updated one.

        The stored context under ``ctx.id`` is replaced by ``ctx`` and, when the
        payload is a ``MechanismContext``, the mechanism registry entry is
        replaced too. A payload of another type does not remove an earlier
        mechanism entry. The context is replaced before the ``env_id`` check, so
        a call that raises for a missing ``env_id`` leaves the new context
        stored while the mechanism registry keeps the previous payload.

        Parameters
        ----------
        ctx : Context
            Context whose ``id`` is already registered.

        Raises
        ------
        KeyError
            If ``ctx.id`` is not registered.
        ValueError
            If the payload is a ``MechanismContext`` with ``env_id=None``.
        """

        if ctx.id not in self._contexts:
            raise KeyError(f"Context {ctx.id} not registered")

        self._contexts[ctx.id] = ctx

        if isinstance(ctx.payload, MechanismContext):
            if ctx.payload.env_id is None:
                raise ValueError("MechanismContext must include env_id")

            self._mechanism_registry[ctx.id] = ctx.payload

    def flush(self, status: Optional[MechanismStatus] = None) -> None:
        """Drop mechanisms from the mechanism registry.

        Parameters
        ----------
        status : MechanismStatus or None, optional
            Only remove mechanisms in this status. ``None`` removes all of
            them. ``RayOptimizer.evaluate`` calls ``flush(status=eval)`` after
            each evaluation pass so evaluated candidates are not fetched again.

        Notes
        -----
        Only ``_mechanism_registry`` is touched. The ``Context`` objects that
        carried the mechanisms stay in ``_contexts`` and in ``_opt_ctx_map``.
        """

        to_delete = []

        for ctx_id, m_ctx in self._mechanism_registry.items():
            if status is not None and m_ctx.status != status:
                continue

            to_delete.append(ctx_id)

        for ctx_id in to_delete:
            del self._mechanism_registry[ctx_id]
