# Experimental first-message session routing

This branch adds `session_model_route` middleware to the messaging gateway. It is an experimental extension, not a released Hermes API. The plugin chooses a model; core owns the persistent session override and request checks.

## Contract

Register `ctx.register_middleware("session_model_route", callback, require_supported=True, platforms=("telegram",))`. The new optional `require_supported` flag rejects an unsupported kind before registration. Stock hosts without the keyword raise `TypeError`; plugins should report this as an unsupported host rather than fall back to request rewriting. The optional `platforms` collection filters eligibility and callback dispatch; omitted means all platforms. The example limits this consumer to Telegram, without a Telegram-specific rule in core. The callback receives `session_id`, `session_key`, `platform`, `model`, `provider`, and `user_message`, plus normal middleware schema fields. Return `{"route": {"model": "...", "provider": "..."}, "reason": "bounded_identifier"}`. The provider must remain the same. No credentials are passed to the callback. A trusted plugin must enforce its own model allowlist and redact any text it sends externally.

The callback runs after successful inbound preparation, including STT, and before the main agent is constructed. Only an external turn on a registered platform, with empty history and no manual override, is eligible. SessionStore atomically reserves the default route in the primary routing database before invoking middleware. A process interruption cannot repeat classification. Multiple conflicting candidates, invalid output, or a middleware exception keep the reserved default. Failed persistence aborts before main-agent execution.

Later turns resolve the pinned provider normally, including its credentials and model capabilities. They do not invoke the classifier. Configured in-agent fallback models are disabled for these turns. A check immediately before provider execution rejects a changed model or provider, including request middleware changes. The response receipt distinguishes requested and provider-returned model IDs. A receipt does not claim that a turn completed successfully.

Manual `/model` clears the automatic pin even for the same model. `/model --once` uses the existing temporary override and restore mechanism. `/new` drops the pin. Session repoints preserve it, including compression repoints.

## Verification boundary

Tests use a real local SessionStore and real AIAgent/Codex request construction with synthetic provider responses. They cover once-only classification, restart, plugin loss, manual/reset races, compaction repoints, failed persistence, failed provider resolution and rejected request-model changes. An A → B → A multiplex test discovers a real temporary plugin, reads separate synthetic credential files, and resolves native runtimes without a network call. Reset tests cover both classification and pre-delivery boundaries. Native `/model --once`, a throwing request middleware and a configured fallback on provider failure are exercised. No live gateway, real recording, live credential store, or remote inference is involved. Full fleet compatibility and live provider behavior still require separate validation before deployment.
