"""Real plugin discovery, config, secret and native route resolution in A -> B -> A."""
from types import SimpleNamespace

import pytest

import gateway.run as run
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from gateway.session_model_route import prepare_route, read_route
from hermes_cli.plugins import get_plugin_manager
from hermes_constants import get_hermes_home
from agent.secret_scope import set_multiplex_active


@pytest.mark.asyncio
async def test_scoped_native_route_discovery_and_provider_resolution(tmp_path):
    plugin_source = '''from agent.secret_scope import get_secret
from hermes_constants import get_hermes_home
seen = []
def register(ctx):
    def select(**kw):
        seen.append((str(get_hermes_home()), get_secret("ROUTE_TEST_KEY")))
        return {"route": {"model": ctx.get_config("target"), "provider": kw["provider"]}}
    ctx.register_middleware("session_model_route", select, require_supported=True, platforms=("telegram",))
'''
    homes = {}
    for name in ("alpha", "beta"):
        home = tmp_path / name
        plugin = home / "plugins" / "synthetic-route"
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text("name: synthetic-route\nversion: 0.0.1\n", encoding="utf-8")
        (plugin / "__init__.py").write_text(plugin_source, encoding="utf-8")
        (home / "config.yaml").write_text(
            "model:\n  default: synthetic-default\n  provider: custom\n"
            "  base_url: http://127.0.0.1:9/v1\n  api_key: ${ROUTE_TEST_KEY}\n"
            f"plugins:\n  enabled: [synthetic-route]\n  entries:\n    synthetic-route:\n      settings:\n        target: model-{name}\n", encoding="utf-8")
        (home / ".env").write_text(f"ROUTE_TEST_KEY=synthetic-key-{name}\n", encoding="utf-8")
        homes[name] = home
    runner = run.GatewayRunner(GatewayConfig(multiplex_profiles=True))
    store = SessionStore(sessions_dir=tmp_path / "launch" / "sessions", config=GatewayConfig(multiplex_profiles=True))
    runner.session_store = store
    managers = {}
    set_multiplex_active(True)
    try:
        for name in ("alpha", "beta", "alpha"):
            home = homes[name]
            with run._profile_runtime_scope(home, hydrate_secrets=False):
                assert get_hermes_home() == home
                source = SessionSource(platform=Platform.TELEGRAM, chat_id="same-chat", user_id="user", profile=name)
                entry = store.get_or_create_session(source)
                await prepare_route(runner, SimpleNamespace(internal=False), source, entry, [], "Synthetic first text")
                model, runtime = runner._resolve_session_agent_runtime(source=source, session_key=entry.session_key)
                assert model == f"model-{name}"
                assert runtime["provider"] == "custom"
                assert runtime["api_key"] == f"synthetic-key-{name}"
                assert "key" not in str(read_route(runner, entry.session_key))
                managers[name] = get_plugin_manager()
                plugin = managers[name]._plugins["synthetic-route"]
                assert plugin.module.seen == [(str(home), f"synthetic-key-{name}")]
        assert managers["alpha"] is not managers["beta"]
        # No route reservation or callback on platforms the plugin did not opt into.
        with run._profile_runtime_scope(homes["alpha"], hydrate_secrets=False):
            source = SessionSource(platform=Platform.DISCORD, chat_id="same-chat", user_id="user", profile="alpha")
            entry = store.get_or_create_session(source)
            await prepare_route(runner, SimpleNamespace(internal=False), source, entry, [], "Not enabled here")
            assert read_route(runner, entry.session_key) is None
            assert store.get_model_override(entry.session_key) is None
    finally:
        set_multiplex_active(False)
        store.close_all_db_handles()
