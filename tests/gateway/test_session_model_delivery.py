"""Reset fences cover model receipts as well as ordinary replies."""
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_duplicate_user_message import _bootstrap, _event, _source


@pytest.mark.asyncio
@pytest.mark.parametrize("reset_at", ["classification", "agent"])
async def test_reset_discards_old_turn_and_model_receipt(monkeypatch, tmp_path, reset_at):
    runner = _bootstrap(monkeypatch, tmp_path)
    delivered = AsyncMock()
    runner._deliver_platform_notice = delivered
    current = True
    runner._is_session_run_current = lambda *_: current

    async def classify(*args):
        nonlocal current
        if reset_at == "classification":
            current = False

    async def old_run(**kwargs):
        nonlocal current
        current = False
        return {"final_response": "old reply", "messages": [],
                "native_model_route_notice": "Old session model confirmed"}

    monkeypatch.setattr("gateway.session_model_route.prepare_route", classify)
    runner._run_agent = AsyncMock(side_effect=old_run)
    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1)
    assert response is None
    assert "Old session model confirmed" not in [call.args[1] for call in delivered.await_args_list]
    assert runner._run_agent.await_count == (0 if reset_at == "classification" else 1)
