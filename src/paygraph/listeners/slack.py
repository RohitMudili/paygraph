"""HTTP listener that resolves Slack approval responses automatically.

Without this listener, callers must build their own HTTP server to receive
Slack interaction payloads and call ``wallet.complete_spend()``. The listener
exposes a Slack-compatible endpoint (``POST /paygraph/slack/callback``),
verifies the Slack request signature, and routes the response to the wallet
that owns the pending ``request_id``.

**Slack app setup required.** ``SlackApprovalGateway`` posts approval requests
via an incoming webhook (plain text). Incoming webhooks cannot receive
interaction callbacks, so to use this listener you must configure a separate
Slack app with *Interactivity* enabled and its Request URL pointed at the
endpoint this listener exposes. The Slack app's Block Kit message should
include Approve/Deny buttons whose ``action_id`` is ``"approve"`` or
``"deny"`` and whose ``value`` is the ``request_id`` from
``HumanApprovalRequired``.

Example::

    from fastapi import FastAPI
    from paygraph import AgentWallet, SpendPolicy, SlackApprovalGateway
    from paygraph.gateways.mock import MockGateway
    from paygraph.listeners import SlackListener

    wallet = AgentWallet(
        gateways=SlackApprovalGateway(
            webhook_url="https://hooks.slack.com/...",
            inner_gateway=MockGateway(auto_approve=True),
        ),
        policy=SpendPolicy(require_human_approval_above=20.0),
    )

    listener = SlackListener(signing_secret="...")
    listener.register(wallet)

    # Standalone:
    #   uvicorn myapp:app --host 0.0.0.0 --port 8080
    app = listener.app()

    # Or embed in an existing FastAPI app:
    #   parent = FastAPI()
    #   listener.mount(parent, path="/paygraph/slack/callback")
"""

import hashlib
import hmac
import json
import time
from typing import TYPE_CHECKING, Optional

from paygraph.exceptions import SpendDeniedError, UnknownApprovalError

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import FastAPI

    from paygraph.wallet import AgentWallet


DEFAULT_CALLBACK_PATH = "/paygraph/slack/callback"
SIGNATURE_TOLERANCE_SECONDS = 60 * 5  # Slack's recommended replay window


class SlackListener:
    """FastAPI listener that resolves Slack interaction payloads.

    Verifies Slack's signed requests (HMAC-SHA256 over ``v0:<timestamp>:<body>``)
    and dispatches the human's Approve/Deny click to the right wallet.

    Routing: the listener iterates through registered wallets and asks each
    ``SlackApprovalGateway`` whether it owns the incoming ``request_id``. The
    first gateway that recognises the id receives ``complete_spend()``.
    """

    def __init__(self, signing_secret: str) -> None:
        """Initialise the listener.

        Args:
            signing_secret: The Slack app's signing secret. Used to verify
                that incoming requests originated from Slack and not a forger.
                Find this in the Slack app's *Basic Information* page.
        """
        if not signing_secret:
            raise ValueError("signing_secret is required for signature verification.")
        self.signing_secret = signing_secret
        self._wallets: list = []

    def register(self, wallet: "AgentWallet") -> None:
        """Register a wallet whose pending approvals this listener resolves.

        Call once per wallet at startup. The listener will route any incoming
        approval whose ``request_id`` is in one of this wallet's
        ``SlackApprovalGateway._pending`` stores.
        """
        if wallet not in self._wallets:
            self._wallets.append(wallet)

    def verify_signature(
        self, timestamp: str, body: bytes, slack_signature: str
    ) -> bool:
        """Verify Slack's request signature and timestamp.

        Implements the spec at https://api.slack.com/authentication/verifying-requests-from-slack:
        sigbase = ``f"v0:{timestamp}:{body}"`` → HMAC-SHA256 with signing secret →
        ``f"v0={hexdigest}"`` compared with ``X-Slack-Signature``.

        Args:
            timestamp: The ``X-Slack-Request-Timestamp`` header value.
            body: Raw request body bytes (must be the unparsed bytes Slack sent).
            slack_signature: The ``X-Slack-Signature`` header value.

        Returns:
            True iff the timestamp is within the replay window AND the
            signature matches.
        """
        if not timestamp or not slack_signature:
            return False
        try:
            ts = int(timestamp)
        except ValueError:
            return False
        if abs(time.time() - ts) > SIGNATURE_TOLERANCE_SECONDS:
            return False

        sigbase = b"v0:" + timestamp.encode("utf-8") + b":" + body
        digest = hmac.new(
            self.signing_secret.encode("utf-8"), sigbase, hashlib.sha256
        ).hexdigest()
        expected = "v0=" + digest
        return hmac.compare_digest(expected, slack_signature)

    def _find_owner(self, request_id: str) -> Optional[tuple]:
        """Return ``(wallet, gateway_name)`` for the gateway that owns ``request_id``."""
        for wallet in self._wallets:
            found = wallet.find_pending_approval(request_id)
            if found is not None:
                gateway_name, _ = found
                return wallet, gateway_name
        return None

    def handle_payload(self, payload: dict) -> dict:
        """Resolve a parsed Slack ``block_actions`` interaction payload.

        Expects a Block Kit payload (the modern format — see
        https://docs.slack.dev/reference/interaction-payloads/block_actions-payload)::

            {
                "type": "block_actions",
                "actions": [
                    {
                        "action_id": "approve",   # or "deny"
                        "value": "<request_id>",
                        "type": "button",
                        ...
                    }
                ],
                ...
            }

        Returns a JSON-serialisable dict suitable for the HTTP response body.
        Slack only requires a ``200 OK`` for acknowledgement; the body is
        useful for tests and for callers who want to surface the result.
        """
        actions = payload.get("actions") or []
        if not actions:
            return {"ok": False, "error": "missing_actions"}

        action = actions[0]
        decision = (action.get("action_id") or "").lower()
        request_id = action.get("value")
        if decision not in {"approve", "deny"}:
            return {"ok": False, "error": f"unknown action_id: {decision!r}"}
        if not request_id:
            return {"ok": False, "error": "missing_request_id"}

        owner = self._find_owner(request_id)
        if owner is None:
            return {
                "ok": False,
                "error": "unknown_request_id",
                "request_id": request_id,
            }

        wallet, gateway_name = owner
        approved = decision == "approve"
        try:
            wallet.complete_spend(request_id, approved=approved, gateway=gateway_name)
        except SpendDeniedError as e:
            return {"ok": True, "approved": False, "reason": str(e)}
        except UnknownApprovalError as e:
            return {"ok": False, "error": "unknown_request_id", "reason": str(e)}
        return {"ok": True, "approved": True}

    def app(self, path: str = DEFAULT_CALLBACK_PATH) -> "FastAPI":
        """Return a standalone FastAPI app exposing the callback endpoint."""
        try:
            from fastapi import FastAPI
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "SlackListener requires the 'slack' extra. "
                "Install with: pip install paygraph[slack]"
            ) from e

        app = FastAPI(title="paygraph-slack-listener")
        self.mount(app, path=path)
        return app

    def mount(self, app: "FastAPI", path: str = DEFAULT_CALLBACK_PATH) -> None:
        """Mount the callback route onto an existing FastAPI app."""
        try:
            from fastapi import Request
            from fastapi.responses import JSONResponse
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "SlackListener requires the 'slack' extra. "
                "Install with: pip install paygraph[slack]"
            ) from e

        @app.post(path)
        async def slack_callback(request: Request) -> JSONResponse:
            body = await request.body()
            timestamp = request.headers.get("x-slack-request-timestamp", "")
            signature = request.headers.get("x-slack-signature", "")
            if not self.verify_signature(timestamp, body, signature):
                return JSONResponse({"ok": False, "error": "invalid_signature"}, status_code=401)

            # Slack posts interactive payloads as form-encoded with a single
            # 'payload' field containing JSON.
            form = await request.form()
            raw_payload = form.get("payload")
            if raw_payload is None:
                return JSONResponse({"ok": False, "error": "missing_payload"}, status_code=400)
            try:
                payload = json.loads(raw_payload)
            except json.JSONDecodeError:
                return JSONResponse({"ok": False, "error": "invalid_json"}, status_code=400)

            # Slack expects a 200 within 3s and retries on non-2xx. Always
            # ack with 200; the body conveys the outcome for tests/observers.
            return JSONResponse(self.handle_payload(payload), status_code=200)
