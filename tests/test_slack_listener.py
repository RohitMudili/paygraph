import hashlib
import hmac
import json
import tempfile
import time
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from paygraph import (
    AgentWallet,
    HumanApprovalRequired,
    SlackApprovalGateway,
    SlackListener,
    SpendPolicy,
)
from paygraph.gateways.mock import MockGateway

SIGNING_SECRET = "test-signing-secret"


def _sign(body: bytes, timestamp: str, secret: str = SIGNING_SECRET) -> str:
    """Produce a Slack-style signature for the given body+timestamp."""
    sigbase = b"v0:" + timestamp.encode() + b":" + body
    digest = hmac.new(secret.encode(), sigbase, hashlib.sha256).hexdigest()
    return "v0=" + digest


def _make_wallet(threshold: float = 20.0) -> tuple[AgentWallet, SlackApprovalGateway, str]:
    f = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    f.close()
    gateway = SlackApprovalGateway(
        webhook_url="https://hooks.slack.com/test",
        inner_gateway=MockGateway(auto_approve=True),
    )
    wallet = AgentWallet(
        gateways=gateway,
        policy=SpendPolicy(require_human_approval_above=threshold),
        log_path=f.name,
        verbose=False,
    )
    return wallet, gateway, f.name


def _read_audit(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _issue_pending(wallet: AgentWallet, amount: float = 50.0) -> str:
    with patch("httpx.post"):
        with pytest.raises(HumanApprovalRequired) as exc_info:
            wallet.request_spend(amount, "Anthropic", "need tokens")
    return exc_info.value.request_id


def _block_actions(request_id: str, decision: str = "approve") -> dict:
    """Build a Block Kit ``block_actions`` payload like Slack would send."""
    return {
        "type": "block_actions",
        "actions": [
            {
                "action_id": decision,
                "value": request_id,
                "type": "button",
                "action_ts": "1548426417.840180",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------


class TestSignatureVerification:
    def test_valid_signature_passes(self):
        listener = SlackListener(signing_secret=SIGNING_SECRET)
        body = b"payload=%7B%7D"
        ts = str(int(time.time()))
        sig = _sign(body, ts)
        assert listener.verify_signature(ts, body, sig) is True

    def test_tampered_body_fails(self):
        listener = SlackListener(signing_secret=SIGNING_SECRET)
        ts = str(int(time.time()))
        sig = _sign(b"original", ts)
        assert listener.verify_signature(ts, b"tampered", sig) is False

    def test_wrong_secret_fails(self):
        listener = SlackListener(signing_secret=SIGNING_SECRET)
        body = b"payload=x"
        ts = str(int(time.time()))
        sig = _sign(body, ts, secret="other-secret")
        assert listener.verify_signature(ts, body, sig) is False

    def test_expired_timestamp_fails(self):
        listener = SlackListener(signing_secret=SIGNING_SECRET)
        body = b"payload=x"
        ts = str(int(time.time()) - 10 * 60)  # 10 min old, outside 5 min window
        sig = _sign(body, ts)
        assert listener.verify_signature(ts, body, sig) is False

    def test_future_timestamp_fails(self):
        listener = SlackListener(signing_secret=SIGNING_SECRET)
        body = b"payload=x"
        ts = str(int(time.time()) + 10 * 60)
        sig = _sign(body, ts)
        assert listener.verify_signature(ts, body, sig) is False

    def test_missing_headers_fail(self):
        listener = SlackListener(signing_secret=SIGNING_SECRET)
        assert listener.verify_signature("", b"x", "v0=abc") is False
        assert listener.verify_signature("123", b"x", "") is False

    def test_non_numeric_timestamp_fails(self):
        listener = SlackListener(signing_secret=SIGNING_SECRET)
        assert listener.verify_signature("not-a-number", b"x", "v0=abc") is False

    def test_empty_secret_rejected_at_construction(self):
        with pytest.raises(ValueError, match="signing_secret"):
            SlackListener(signing_secret="")


# ---------------------------------------------------------------------------
# Payload handling (logic only, no HTTP)
# ---------------------------------------------------------------------------


class TestHandlePayload:
    def test_approve_calls_complete_spend_and_audits_approval(self):
        wallet, _, audit_path = _make_wallet()
        request_id = _issue_pending(wallet)
        listener = SlackListener(signing_secret=SIGNING_SECRET)
        listener.register(wallet)

        result = listener.handle_payload(_block_actions(request_id, "approve"))
        assert result == {"ok": True, "approved": True}
        records = _read_audit(audit_path)
        approved = [r for r in records if r["policy_result"] == "approved"]
        assert len(approved) == 1
        assert approved[0]["vendor"] == "Anthropic"
        assert approved[0]["amount"] == 50.0

    def test_deny_records_denial_in_audit(self):
        wallet, _, audit_path = _make_wallet()
        request_id = _issue_pending(wallet)
        listener = SlackListener(signing_secret=SIGNING_SECRET)
        listener.register(wallet)

        result = listener.handle_payload(_block_actions(request_id, "deny"))
        assert result["ok"] is True
        assert result["approved"] is False
        assert "Human denied" in result["reason"]
        denied = [r for r in _read_audit(audit_path) if r["policy_result"] == "denied"]
        assert len(denied) == 1
        assert "Human denied" in denied[0]["denial_reason"]

    def test_unknown_request_id_returns_error(self):
        wallet, _, _ = _make_wallet()
        listener = SlackListener(signing_secret=SIGNING_SECRET)
        listener.register(wallet)

        result = listener.handle_payload(_block_actions("ffffffffffffffff", "approve"))
        assert result["ok"] is False
        assert result["error"] == "unknown_request_id"

    def test_missing_request_id_returns_error(self):
        """A button with no ``value`` (request_id) cannot be routed."""
        listener = SlackListener(signing_secret=SIGNING_SECRET)
        result = listener.handle_payload(
            {
                "type": "block_actions",
                "actions": [{"action_id": "approve", "type": "button"}],
            }
        )
        assert result["ok"] is False
        assert result["error"] == "missing_request_id"

    def test_missing_actions_returns_error(self):
        listener = SlackListener(signing_secret=SIGNING_SECRET)
        result = listener.handle_payload({"type": "block_actions"})
        assert result["ok"] is False
        assert result["error"] == "missing_actions"

    def test_unknown_action_id_returns_error(self):
        wallet, _, _ = _make_wallet()
        request_id = _issue_pending(wallet)
        listener = SlackListener(signing_secret=SIGNING_SECRET)
        listener.register(wallet)

        result = listener.handle_payload(_block_actions(request_id, "maybe"))
        assert result["ok"] is False
        assert "unknown action_id" in result["error"]

    def test_routes_to_correct_wallet_when_multiple_registered(self):
        wallet_a, _, audit_a = _make_wallet()
        wallet_b, _, audit_b = _make_wallet()
        request_id_b = _issue_pending(wallet_b)

        listener = SlackListener(signing_secret=SIGNING_SECRET)
        listener.register(wallet_a)
        listener.register(wallet_b)

        result = listener.handle_payload(_block_actions(request_id_b, "approve"))
        assert result == {"ok": True, "approved": True}
        # Only wallet_b's audit should record the approval.
        assert any(r["policy_result"] == "approved" for r in _read_audit(audit_b))
        assert not any(r["policy_result"] == "approved" for r in _read_audit(audit_a))


# ---------------------------------------------------------------------------
# Full HTTP integration (FastAPI TestClient)
# ---------------------------------------------------------------------------


def _post_slack(client: TestClient, payload: dict, secret: str = SIGNING_SECRET) -> object:
    """Form-encode a Slack interaction payload and sign it like Slack would."""
    from urllib.parse import urlencode

    body = urlencode({"payload": json.dumps(payload)}).encode()
    ts = str(int(time.time()))
    sig = _sign(body, ts, secret)
    return client.post(
        "/paygraph/slack/callback",
        content=body,
        headers={
            "X-Slack-Request-Timestamp": ts,
            "X-Slack-Signature": sig,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )


class TestHttpEndpoint:
    def test_full_loop_approve(self):
        wallet, _, audit_path = _make_wallet()
        request_id = _issue_pending(wallet)
        listener = SlackListener(signing_secret=SIGNING_SECRET)
        listener.register(wallet)
        client = TestClient(listener.app())

        resp = _post_slack(client, _block_actions(request_id, "approve"))
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "approved": True}
        approved = [
            r for r in _read_audit(audit_path) if r["policy_result"] == "approved"
        ]
        assert len(approved) == 1

    def test_invalid_signature_rejected(self):
        wallet, _, _ = _make_wallet()
        request_id = _issue_pending(wallet)
        listener = SlackListener(signing_secret=SIGNING_SECRET)
        listener.register(wallet)
        client = TestClient(listener.app())

        resp = _post_slack(
            client, _block_actions(request_id, "approve"), secret="forger-secret"
        )
        assert resp.status_code == 401
        assert resp.json() == {"ok": False, "error": "invalid_signature"}

    def test_can_mount_on_existing_app(self):
        wallet, _, _ = _make_wallet()
        request_id = _issue_pending(wallet)
        listener = SlackListener(signing_secret=SIGNING_SECRET)
        listener.register(wallet)

        parent = FastAPI()

        @parent.get("/health")
        def health():
            return {"status": "ok"}

        listener.mount(parent)
        client = TestClient(parent)
        assert client.get("/health").json() == {"status": "ok"}

        resp = _post_slack(client, _block_actions(request_id, "approve"))
        assert resp.status_code == 200

    def test_unknown_request_id_returns_200_per_slack_contract(self):
        """Slack retries on non-2xx, so unknown request_id still acks 200.
        The body conveys the error."""
        wallet, _, _ = _make_wallet()
        listener = SlackListener(signing_secret=SIGNING_SECRET)
        listener.register(wallet)
        client = TestClient(listener.app())

        resp = _post_slack(client, _block_actions("ffffffffffffffff", "approve"))
        assert resp.status_code == 200
        assert resp.json()["ok"] is False
        assert resp.json()["error"] == "unknown_request_id"
