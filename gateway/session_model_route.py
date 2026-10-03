"""Experimental first-message routing before native agent construction.

The plugin only chooses. The gateway owns durability, manual precedence and
failure behavior; later turns do not depend on a plugin remaining available.
"""
from __future__ import annotations

import asyncio
import re


KIND = "session_model_route"
META_KEY = "native_model_route"
_IDENTIFIER = re.compile(r"[A-Za-z0-9_./:+-]{1,160}\Z")


class SessionModelRouteError(RuntimeError):
    """A pinned route could not be safely retained."""

    def __init__(self):
        super().__init__("Session model routing unavailable. No alternate model was selected. Retry or use /model explicitly.")


def _valid_route(route):
    return (isinstance(route, dict) and route.get("phase") in {"pending", "selected"}
            and all(isinstance(route.get(k), str) and _IDENTIFIER.fullmatch(route[k])
                    for k in ("model", "provider")))



def read_route(runner, key):
    from gateway.session import SessionStore
    store = getattr(runner, "session_store", None)
    if not key or not isinstance(store, SessionStore):
        return None
    try:
        value = store.get_session_metadata(key, META_KEY)
    except Exception as exc:
        raise SessionModelRouteError() from exc
    # No marker is the unchanged stock path, including legacy test doubles.
    if value is None:
        return None
    if not _valid_route(value) or not store._routing_db_loaded:
        raise SessionModelRouteError()
    state = runner._peek_session_state(key)
    if state and state.conversation.one_turn_restore:
        return None  # Explicit /model --once uses the stock restore mechanism.
    return value


def resolve_pinned_runtime(runner, key):
    route = read_route(runner, key)
    if route is None:
        return None
    from gateway.run import _resolve_runtime_agent_kwargs_for_provider
    try:
        runtime = _resolve_runtime_agent_kwargs_for_provider(route["provider"], target_model=route["model"])
        if (runtime.get("provider") != route["provider"] or runtime.get("_fallback_notice")
                or runtime.get("model", route["model"]) != route["model"]):
            raise SessionModelRouteError()
    except Exception as exc:
        raise SessionModelRouteError() from exc
    runtime.pop("model", None)
    runner._session_state(key).conversation.model_override = {**runtime, "model": route["model"]}
    return route["model"], runtime


def bind_agent(runner, key, agent):
    """Bind the resolved route and disable fallback before the main agent executes."""
    from agent.session_model_pin import SessionModelPin
    route = read_route(runner, key)
    agent._session_model_pin = (SessionModelPin(route["model"], route["provider"],
        route.get("reason", "default_after_interrupted_classifier")) if route else None)
    if route:
        runner._apply_fallback_chain_to_agent(agent, None)


async def prepare_route(runner, event, source, entry, history, message):
    """Run only for an eligible first human turn, with already prepared text."""
    from hermes_cli.middleware import middleware_payload
    from hermes_cli.plugins import has_middleware, invoke_middleware

    if event.internal:
        return
    key = entry.session_key
    if read_route(runner, key) is not None:
        return
    if not has_middleware(KIND, platform=source.platform.value):
        return
    if not runner.session_store._routing_db_loaded:
        raise SessionModelRouteError()
    if history or entry.model_override or runner._session_model_override(key):
        return
    sid = entry.session_id

    def choose():
        model, runtime = runner._resolve_session_agent_runtime(source=source, session_key=key)
        pending = {"model": model, "provider": runtime.get("provider"), "phase": "pending"}
        if not runner.session_store.compare_and_set_model_route(key, sid, None, pending):
            return
        # Persist the default BEFORE the external call. A crash cannot retry Jev
        # or send an uncommitted Astra request after the process resumes.
        selected = {**pending, "phase": "selected", "reason": "default"}
        results = invoke_middleware(KIND, **middleware_payload(
            session_id=sid, session_key=key, platform=source.platform.value,
            model=model, provider=runtime["provider"], user_message=message))
        candidates = [r for r in results if isinstance(r, dict) and isinstance(r.get("route"), dict)]
        if len(candidates) == 1:
            proposal = candidates[0]["route"]
            if (proposal.get("provider") == pending["provider"]
                    and _valid_route({**proposal, "phase": "selected"})):
                reason = candidates[0].get("reason", "selected")
                if not isinstance(reason, str) or not _IDENTIFIER.fullmatch(reason):
                    reason = "selected"
                selected = {"model": proposal["model"], "provider": proposal["provider"],
                            "phase": "selected", "reason": reason}
        # A /model or /new arriving during classification wins this comparison.
        runner.session_store.compare_and_set_model_route(key, sid, pending, selected)

    await asyncio.to_thread(choose)
