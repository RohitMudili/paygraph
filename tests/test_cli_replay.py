import json
import tempfile
from datetime import datetime, timezone

import pytest

from paygraph.audit import AuditLogger, AuditRecord
from paygraph.cli import run_replay


def _make_audit(amount: float = 40.0) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    f.close()
    record = AuditRecord(
        timestamp=datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc).isoformat(),
        agent_id="default",
        amount=amount,
        vendor="Anthropic",
        justification="tokens",
        policy_result="approved",
        denial_reason=None,
        checks_passed=[],
        gateway_ref="ref",
        gateway_type="mock",
        policy_snapshot={"max_transaction": 50.0, "daily_budget": 200.0},
    )
    AuditLogger(log_path=f.name, verbose=False).log(record)
    return f.name


def _make_policy_json(max_transaction: float = 10.0) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    json.dump({"max_transaction": max_transaction, "daily_budget": 100.0}, f)
    f.close()
    return f.name


class TestReplayCli:
    def test_runs_and_returns_zero_on_success(self, capsys):
        audit = _make_audit(amount=40.0)
        policy = _make_policy_json(max_transaction=10.0)

        rc = run_replay(audit, policy, all_rows=False, as_json=False)
        captured = capsys.readouterr()

        assert rc == 0
        assert "Policy replay report" in captured.out
        assert "Approved -> Denied:    1" in captured.out

    def test_json_output_is_parseable(self, capsys):
        audit = _make_audit(amount=40.0)
        policy = _make_policy_json(max_transaction=10.0)

        rc = run_replay(audit, policy, all_rows=False, as_json=True)
        captured = capsys.readouterr()

        assert rc == 0
        payload = json.loads(captured.out)
        assert payload["total"] == 1
        assert payload["flipped_to_denied"] == 1
        assert payload["candidate_policy"]["max_transaction"] == 10.0

    def test_missing_audit_log_returns_nonzero(self, capsys):
        policy = _make_policy_json()
        rc = run_replay(
            "C:/nonexistent/audit.jsonl", policy, all_rows=False, as_json=False
        )
        captured = capsys.readouterr()
        assert rc == 1
        assert "audit log not found" in captured.err

    def test_missing_policy_returns_nonzero(self, capsys):
        audit = _make_audit()
        rc = run_replay(
            audit, "C:/nonexistent/policy.json", all_rows=False, as_json=False
        )
        captured = capsys.readouterr()
        assert rc == 1
        assert "policy file not found" in captured.err


class TestReplaySubcommandArgParsing:
    def test_replay_requires_policy_flag(self, monkeypatch):
        from paygraph import cli as cli_mod

        monkeypatch.setattr("sys.argv", ["paygraph", "replay", "audit.jsonl"])
        with pytest.raises(SystemExit):
            cli_mod.main()
