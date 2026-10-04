"""Enforce and report an opt-in native session route at the provider boundary."""
from dataclasses import dataclass, field
import re


@dataclass
class SessionModelPin:
    model: str
    provider: str
    reason: str = "selected"
    requests: int = 0
    response_models: set[str] = field(default_factory=set)
    unconfirmed: bool = False

    def before_request(self, agent, request):
        if (agent.model != self.model or agent.provider != self.provider
                or request.get("model") != self.model):
            raise RuntimeError("Pinned session model changed. Request blocked; use /model to change it explicitly.")
        self.requests += 1

    def after_response(self, response):
        model = getattr(response, "model", None)
        if isinstance(model, str) and re.fullmatch(r"[A-Za-z0-9_./:+-]{1,160}", model):
            self.response_models.add(model[:160])
        else:
            self.unconfirmed = True

    def receipt(self):
        if not self.requests:
            return f"Selected: {self.model}. No provider request confirmed."
        if self.response_models == {self.model} and not self.unconfirmed:
            return f"Model: {self.model}. Confirmed by provider response. Routing: {self.reason}."
        if self.response_models:
            names = ", ".join(sorted(self.response_models))
            return f"Requested: {self.model}. Provider reported: {names}. Model identity differs or is incomplete."
        return f"Requested: {self.model}. Provider model unconfirmed."
