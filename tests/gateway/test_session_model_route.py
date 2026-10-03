"""Native first-turn routing must not depend on a live plugin after selection."""
import gateway.run as run
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from gateway.session_model_route import META_KEY, prepare_route, persist_route, read_route, SessionModelRouteError
from agent.session_model_pin import SessionModelPin

SOL = "gpt-6.1-sol"
ASTRA = "gpt-6-astra"
PROVIDER = "openai-codex"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    runner = run.GatewayRunner(GatewayConfig())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="synthetic", user_id="tester")
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    runner.session_store = store
    entry = store.get_or_create_session(source)
    monkeypatch.setattr(run, "_resolve_gateway_model", lambda *_a, **_kw: SOL)
    monkeypatch.setattr(run, "_resolve_runtime_agent_kwargs", lambda: {"provider": PROVIDER, "api_key": "test", "api_mode": "codex_responses"})
    monkeypatch.setattr(run, "_resolve_runtime_agent_kwargs_for_provider", lambda provider, **kw: {"provider": provider, "api_key": "test", "api_mode": "codex_responses"})
    monkeypatch.setattr("hermes_cli.plugins.has_middleware", lambda kind: kind == "session_model_route")
    classifier = Mock(return_value=[{"route": {"model": ASTRA, "provider": PROVIDER}}])
    monkeypatch.setattr("hermes_cli.plugins.invoke_middleware", classifier)
    yield runner, source, entry, classifier
    store.close_all_db_handles()


@pytest.mark.asyncio
async def test_native_route_survives_followup_restart_and_plugin_loss(setup, monkeypatch):
    runner, source, entry, classifier = setup
    event = SimpleNamespace(internal=False)
    await prepare_route(runner, event, source, entry, [], "Successful voice transcript: please use Astra")
    assert classifier.call_args.kwargs["user_message"].startswith("Successful voice transcript")
    model, runtime = runner._resolve_session_agent_runtime(source=source, session_key=entry.session_key)
    assert (model, runtime["provider"]) == (ASTRA, PROVIDER)
    await prepare_route(runner, event, source, entry, [{"role": "user"}], "second turn")
    assert classifier.call_count == 1
    # New store and empty process state emulate a restart, not a reused object.
    restarted = SessionStore(sessions_dir=runner.session_store.sessions_dir, config=GatewayConfig())
    runner.session_store = restarted
    runner._session_state(entry.session_key).conversation.model_override = None
    monkeypatch.setattr("hermes_cli.plugins.has_middleware", lambda _: False)
    await prepare_route(runner, event, source, restarted.get_or_create_session(source), [], "retry after restart")
    assert runner._resolve_session_agent_runtime(source=source, session_key=entry.session_key)[0] == ASTRA
    assert classifier.call_count == 1
    restarted.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize("output", [[], [{"route": {"model": "bad", "provider": "other"}}], [{"route": {"model": "bad\nmodel", "provider": PROVIDER}}]])
async def test_classifier_abstention_pins_default_once(setup, output):
    runner, source, entry, classifier = setup
    classifier.return_value = output
    await prepare_route(runner, SimpleNamespace(internal=False), source, entry, [], "hello")
    await prepare_route(runner, SimpleNamespace(internal=False), source, entry, [], "retry")
    assert runner._resolve_session_agent_runtime(session_key=entry.session_key)[0] == SOL
    assert classifier.call_count == 1


@pytest.mark.asyncio
async def test_commit_failure_cannot_send_unpersisted_astra_or_reclassify(setup, monkeypatch):
    runner, source, entry, classifier = setup
    original = runner.session_store._routing_db.save_gateway_routing_entry
    writes = []
    def fail_second(*args, **kwargs):
        writes.append(1)
        if len(writes) == 2:
            raise OSError("synthetic disk failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(runner.session_store._routing_db, "save_gateway_routing_entry", fail_second)
    with pytest.raises(SessionModelRouteError):
        await prepare_route(runner, SimpleNamespace(internal=False), source, entry, [], "Astra")
    assert runner.session_store.get_model_override(entry.session_key)["model"] == SOL
    await prepare_route(runner, SimpleNamespace(internal=False), source, entry, [], "retry")
    assert classifier.call_count == 1
    assert runner._resolve_session_agent_runtime(session_key=entry.session_key)[0] == SOL


@pytest.mark.asyncio
async def test_manual_choice_during_jev_wins_even_if_same_model(setup):
    runner, source, entry, classifier = setup
    def manual(*args, **kwargs):
        runner.session_store.set_model_override(entry.session_key, {"model": SOL, "provider": PROVIDER})
        return [{"route": {"model": ASTRA, "provider": PROVIDER}}]
    classifier.side_effect = manual
    await prepare_route(runner, SimpleNamespace(internal=False), source, entry, [], "Astra")
    assert runner.session_store.get_model_override(entry.session_key)["model"] == SOL
    assert read_route(runner, entry.session_key) is None


@pytest.mark.asyncio
async def test_reset_during_jev_cannot_pin_replacement_session(setup):
    runner, source, entry, classifier = setup
    def reset(*args, **kwargs):
        runner.session_store.reset_session(entry.session_key)
        return [{"route": {"model": ASTRA, "provider": PROVIDER}}]
    classifier.side_effect = reset
    await prepare_route(runner, SimpleNamespace(internal=False), source, entry, [], "Astra")
    assert read_route(runner, entry.session_key) is None
    assert runner.session_store.get_model_override(entry.session_key) is None


@pytest.mark.asyncio
async def test_existing_internal_and_manual_sessions_do_not_classify(setup):
    runner, source, entry, classifier = setup
    await prepare_route(runner, SimpleNamespace(internal=True), source, entry, [], "internal")
    await prepare_route(runner, SimpleNamespace(internal=False), source, entry, [{"role": "user"}], "existing")
    runner.session_store.set_model_override(entry.session_key, {"model": SOL, "provider": PROVIDER})
    await prepare_route(runner, SimpleNamespace(internal=False), source, entry, [], "manual")
    classifier.assert_not_called()


@pytest.mark.asyncio
async def test_after_astra_storage_write_failure_does_not_affect_route(setup, monkeypatch):
    runner, source, entry, classifier = setup
    await prepare_route(runner, SimpleNamespace(internal=False), source, entry, [], "Astra")
    monkeypatch.setattr(runner.session_store._routing_db, "save_gateway_routing_entry", Mock(side_effect=OSError("disk")))
    await prepare_route(runner, SimpleNamespace(internal=False), source, entry, [], "again")
    assert runner._resolve_session_agent_runtime(session_key=entry.session_key)[0] == ASTRA
    classifier.assert_called_once()


@pytest.mark.asyncio
async def test_pinned_credential_failure_never_uses_default_runtime(setup, monkeypatch):
    runner, source, entry, _ = setup
    await prepare_route(runner, SimpleNamespace(internal=False), source, entry, [], "Astra")
    default = Mock(side_effect=AssertionError("must not resolve default"))
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", default)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs_for_provider", Mock(side_effect=RuntimeError("auth")))
    with pytest.raises(SessionModelRouteError):
        runner._resolve_session_agent_runtime(session_key=entry.session_key)
    default.assert_not_called()


def test_wire_guard_blocks_model_drift_and_receipt_uses_provider_response():
    pin = SessionModelPin(ASTRA, PROVIDER)
    agent = SimpleNamespace(model=ASTRA, provider=PROVIDER)
    pin.before_request(agent, {"model": ASTRA})
    pin.after_response(SimpleNamespace(model=ASTRA))
    assert "Confirmed by provider" in pin.receipt()
    with pytest.raises(RuntimeError, match="Request blocked"):
        pin.before_request(agent, {"model": SOL})
    assert pin.requests == 1
    other = SessionModelPin(ASTRA, PROVIDER)
    other.before_request(agent, {"model": ASTRA})
    other.after_response(SimpleNamespace())
    assert "unconfirmed" in other.receipt()


@pytest.mark.asyncio
async def test_compaction_repoint_keeps_pin_and_reset_drops_it(setup):
    runner, source, entry, classifier = setup
    await prepare_route(runner, SimpleNamespace(internal=False), source, entry, [], "Astra")
    moved = runner.session_store.switch_session(entry.session_key, "synthetic-compression-successor")
    assert read_route(runner, entry.session_key)["model"] == ASTRA
    await prepare_route(runner, SimpleNamespace(internal=False), source, moved, [], "compacted")
    assert classifier.call_count == 1
    reset = runner.session_store.reset_session(entry.session_key)
    runner._session_state(entry.session_key).conversation.model_override = None
    assert read_route(runner, entry.session_key) is None
    await prepare_route(runner, SimpleNamespace(internal=False), source, reset, [], "fresh")
    assert classifier.call_count == 2


@pytest.mark.asyncio
async def test_bad_routing_store_blocks_rather_than_reclassifies(setup):
    runner, source, entry, classifier = setup
    # Model the state after a failed primary load on process restart.
    runner.session_store._routing_db_loaded = False
    with pytest.raises(SessionModelRouteError):
        await prepare_route(runner, SimpleNamespace(internal=False), source, entry, [], "first")
    classifier.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_drift", [False, True])
async def test_real_native_agent_constructor_and_codex_tool_loop(setup, monkeypatch, wire_drift):
    from run_agent import AIAgent
    from gateway.run_turn_runner import TurnRunner
    from tests.agent.test_run_agent_codex_responses import (
        _patch_agent_bootstrap, _codex_tool_call_response, _codex_message_response)
    runner, source, entry, classifier = setup
    await prepare_route(runner, SimpleNamespace(internal=False), source, entry, [], "Astra")
    model, runtime = runner._resolve_session_agent_runtime(session_key=entry.session_key)
    runtime["base_url"] = "https://chatgpt.com/backend-api/codex"
    _patch_agent_bootstrap(monkeypatch)
    turn = object.__new__(TurnRunner)
    turn._runner = runner
    turn._ctx = SimpleNamespace(
        source=source, session_key=entry.session_key, session_id=entry.session_id,
        user_config={}, enabled_toolsets=["terminal"], disabled_toolsets=[],
        AIAgent=lambda **kw: AIAgent(**kw, skip_memory=True))
    agent = turn._build_fresh_agent(
        {"model": model, "runtime": runtime}, "telegram", "", 4, {}, {}, True)
    assert agent.model == ASTRA
    assert agent.provider == PROVIDER
    assert agent.api_mode == "codex_responses"
    agent._session_model_pin = SessionModelPin(model, PROVIDER)
    agent._disable_streaming = True
    responses = [_codex_tool_call_response(), _codex_message_response("done"), _codex_message_response("again")]
    for response in responses:
        response.model = ASTRA
    requests = []
    def fake_provider(kwargs):
        requests.append(kwargs)
        return responses.pop(0)
    monkeypatch.setattr(agent, "_interruptible_api_call", fake_provider)
    def tool_reply(message, messages, *_args):
        messages.extend({"role": "tool", "tool_call_id": c.id, "content": "synthetic tool result"}
                        for c in message.tool_calls)
    monkeypatch.setattr(agent, "_execute_tool_calls", tool_reply)
    if wire_drift:
        monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **kw: 0)
        def rewrite(request, **kw):
            return SimpleNamespace(payload={**request, "model": SOL}, original_payload=request, trace=[])
        monkeypatch.setattr("hermes_cli.middleware.apply_llm_request_middleware", rewrite)
    first = agent.run_conversation("synthetic command")
    if wire_drift:
        assert requests == []
        assert first.get("failed") is True
        return
    assert first["final_response"] == "done"
    second = agent.run_conversation("synthetic follow-up", conversation_history=first["messages"])
    assert second["final_response"] == "again"
    assert [request["model"] for request in requests] == [ASTRA, ASTRA, ASTRA]
    assert agent._session_model_pin.requests == 3
    assert "Confirmed by provider" in agent._session_model_pin.receipt()
    assert classifier.call_count == 1

