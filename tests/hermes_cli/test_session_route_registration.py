"""Public middleware registration can require host support and restrict platforms."""
import pytest
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def test_scoped_registration_rejects_unknown_and_never_runs_on_other_platform():
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="synthetic-router"), manager)
    calls = []
    with pytest.raises(ValueError, match="Unsupported middleware"):
        ctx.register_middleware("not-supported", lambda **kw: None, require_supported=True)
    assert not manager.has_middleware("not-supported")
    handle = ctx.register_middleware("session_model_route", lambda **kw: calls.append(kw),
                                    require_supported=True, platforms=("telegram",))
    assert manager.has_middleware("session_model_route", platform="telegram")
    assert not manager.has_middleware("session_model_route", platform="discord")
    manager.invoke_middleware("session_model_route", platform="discord")
    assert not calls
    manager.invoke_middleware("session_model_route", platform="telegram")
    assert len(calls) == 1
    handle.release()
    assert not manager.has_middleware("session_model_route", platform="telegram")
