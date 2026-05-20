import json
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone

import pytest

from paygraph.audit import AuditLogger, AuditRecord
from paygraph.policy import SpendPolicy
from paygraph.simulator import (
    FLIPPED_TO_DENIED,
    UNCHANGED,
    PolicySimulator,
    load_policy_json,
)


def _make_audit_path() -> str:
    f = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    f.close()
    return f.name


def _write_record(
    path: str,
    *,
    timestamp: str,
    amount: float,
    vendor: str = "Anthropic",
    justification: str | None = "needs tokens",
    policy_result: str = "approved",
    denial_reason: str | None = None,
    policy_snapshot: dict | None = None,
) -> None:
    record = AuditRecord(
        timestamp=timestamp,
        agent_id="default",
        amount=amount,
        vendor=vendor,
        justification=justification,
        policy_result=policy_result,
        denial_reason=denial_reason,
        checks_passed=[],
        gateway_ref=None,
        gateway_type=None,
        policy_snapshot=policy_snapshot,
    )
    logger = AuditLogger(log_path=path, verbose=False)
    logger.log(record)


def _ts(hour: int = 12, minute: int = 0, day: int = 15) -> str:
    return datetime(2026, 5, day, hour, minute, tzinfo=timezone.utc).isoformat()


class TestReplayBasics:
    def test_unchanged_policy_yields_all_unchanged(self):
        path = _make_audit_path()
        for i in range(3):
            _write_record(path, timestamp=_ts(hour=10 + i), amount=5.0)

        sim = PolicySimulator(SpendPolicy(max_transaction=50.0, daily_budget=200.0))
        report = sim.replay_file(path)

        assert report.total == 3
        assert report.unchanged == 3
        assert report.flipped_to_denied == 0
        assert all(o.delta == UNCHANGED for o in report.outcomes)

    def test_tightening_amount_cap_flips_approved_to_denied(self):
        path = _make_audit_path()
        _write_record(path, timestamp=_ts(hour=9), amount=40.0)
        _write_record(path, timestamp=_ts(hour=10), amount=3.0)

        sim = PolicySimulator(SpendPolicy(max_transaction=10.0, daily_budget=200.0))
        report = sim.replay_file(path)

        assert report.flipped_to_denied == 1
        assert report.unchanged == 1
        flipped = next(o for o in report.outcomes if o.delta == FLIPPED_TO_DENIED)
        assert flipped.amount == 40.0
        assert "exceeds" in flipped.new_denial_reason.lower()

    def test_loosening_amount_cap_keeps_already_approved_unchanged(self):
        path = _make_audit_path()
        _write_record(path, timestamp=_ts(hour=9), amount=20.0)

        sim = PolicySimulator(SpendPolicy(max_transaction=100.0, daily_budget=500.0))
        report = sim.replay_file(path)

        assert report.unchanged == 1
        assert report.flipped_to_denied == 0
        assert report.flipped_to_approved == 0

    def test_only_approved_filter_does_not_count_denied_rows_toward_budget(self):
        path = _make_audit_path()
        # An originally-denied row that fits within the candidate budget
        # individually — but if it (incorrectly) counted toward the budget,
        # the next legitimate row would be denied.
        _write_record(
            path,
            timestamp=_ts(hour=9),
            amount=60.0,
            policy_result="denied",
            denial_reason="Vendor 'BadCo' is blocked",
            vendor="BadCo",
        )
        _write_record(path, timestamp=_ts(hour=10), amount=60.0)

        # Candidate: no vendor blocklist, daily_budget=100. Originally-denied
        # $60 (BadCo) would still be denied by the candidate's amount budget
        # if it leaked into _daily_spend. only_approved=True must keep it out.
        sim = PolicySimulator(SpendPolicy(max_transaction=200.0, daily_budget=100.0))
        report = sim.replay_file(path)

        # The 10:00 Anthropic $60 must approve cleanly because the prior
        # denied $60 didn't consume budget.
        ten_oclock = next(o for o in report.outcomes if "10:00" in o.timestamp)
        assert ten_oclock.new_result == "approved"

    def test_all_flag_reevaluates_denied_rows_and_affects_budget(self):
        path = _make_audit_path()
        _write_record(
            path,
            timestamp=_ts(hour=9),
            amount=80.0,
            policy_result="denied",
            denial_reason="Amount $80.00 exceeds limit of $50.00",
        )
        _write_record(path, timestamp=_ts(hour=10), amount=30.0)

        # With --all and a daily_budget of 100, the resurrected $80 + the $30
        # now blow the daily budget on the second row.
        sim = PolicySimulator(SpendPolicy(max_transaction=1000.0, daily_budget=100.0))
        report = sim.replay_file(path, only_approved=False)

        assert report.flipped_to_approved >= 1
        flipped_to_denied = [o for o in report.outcomes if o.delta == FLIPPED_TO_DENIED]
        assert len(flipped_to_denied) == 1
        assert "budget" in flipped_to_denied[0].new_denial_reason.lower()


class TestCumulativeBudget:
    def test_daily_budget_reconstructed_in_chronological_order(self):
        path = _make_audit_path()
        # Five $25 spends on the same day = $125 total.
        for i in range(5):
            _write_record(path, timestamp=_ts(hour=8 + i), amount=25.0)

        # Candidate caps daily at $60 → first two approve, third onward denied.
        sim = PolicySimulator(SpendPolicy(max_transaction=50.0, daily_budget=60.0))
        report = sim.replay_file(path)

        approved = [o for o in report.outcomes if o.new_result == "approved"]
        denied = [o for o in report.outcomes if o.new_result == "denied"]
        assert len(approved) == 2
        assert len(denied) == 3
        assert all("budget" in o.new_denial_reason.lower() for o in denied)

    def test_records_out_of_order_are_sorted_by_timestamp(self):
        path = _make_audit_path()
        # Write out of order on purpose.
        _write_record(path, timestamp=_ts(hour=14), amount=25.0)
        _write_record(path, timestamp=_ts(hour=9), amount=25.0)
        _write_record(path, timestamp=_ts(hour=11), amount=25.0)

        sim = PolicySimulator(SpendPolicy(max_transaction=50.0, daily_budget=60.0))
        report = sim.replay_file(path)

        # First two chronologically should approve ($50 of $60), the 14:00 one denies.
        outcomes_by_ts = sorted(report.outcomes, key=lambda o: o.timestamp)
        assert outcomes_by_ts[0].new_result == "approved"
        assert outcomes_by_ts[1].new_result == "approved"
        assert outcomes_by_ts[2].new_result == "denied"

    def test_different_days_reset_daily_budget(self):
        path = _make_audit_path()
        _write_record(path, timestamp=_ts(day=15, hour=10), amount=50.0)
        _write_record(path, timestamp=_ts(day=16, hour=10), amount=50.0)

        sim = PolicySimulator(SpendPolicy(max_transaction=100.0, daily_budget=60.0))
        report = sim.replay_file(path)

        assert all(o.new_result == "approved" for o in report.outcomes)


class TestSnapshotForwardCompat:
    def test_rows_without_policy_snapshot_still_replay(self):
        path = _make_audit_path()
        # Manually write a row in the legacy format (no policy_snapshot field).
        legacy = {
            "timestamp": _ts(hour=10),
            "agent_id": "default",
            "amount": 5.0,
            "vendor": "Anthropic",
            "justification": "tokens",
            "policy_result": "approved",
            "denial_reason": None,
            "checks_passed": [],
            "gateway_ref": "ref",
            "gateway_type": "mock",
        }
        with open(path, "w") as f:
            f.write(json.dumps(legacy) + "\n")

        sim = PolicySimulator(SpendPolicy(max_transaction=10.0, daily_budget=100.0))
        report = sim.replay_file(path)

        assert report.total == 1
        assert report.unchanged == 1


class TestReportShape:
    def test_summary_counts_match_outcomes(self):
        path = _make_audit_path()
        _write_record(path, timestamp=_ts(hour=9), amount=40.0)
        _write_record(path, timestamp=_ts(hour=10), amount=3.0)
        _write_record(
            path,
            timestamp=_ts(hour=11),
            amount=2.0,
            policy_result="denied",
            denial_reason="Justification is required but was not provided",
            justification=None,
        )

        sim = PolicySimulator(SpendPolicy(max_transaction=10.0, daily_budget=200.0))
        report = sim.replay_file(path)

        assert report.total == 3
        assert (
            report.unchanged
            + report.flipped_to_denied
            + report.flipped_to_approved
            + report.denial_reason_changed
            == report.total
        )

    def test_report_serializable_to_json(self):
        path = _make_audit_path()
        _write_record(path, timestamp=_ts(hour=9), amount=5.0)

        sim = PolicySimulator(SpendPolicy(max_transaction=10.0))
        report = sim.replay_file(path)

        # Round-trip through JSON to confirm everything is serializable.
        blob = json.dumps(asdict(report))
        parsed = json.loads(blob)
        assert parsed["total"] == 1
        assert parsed["candidate_policy"]["max_transaction"] == 10.0

    def test_summary_string_is_human_readable(self):
        path = _make_audit_path()
        _write_record(path, timestamp=_ts(hour=9), amount=40.0)

        sim = PolicySimulator(SpendPolicy(max_transaction=10.0))
        report = sim.replay_file(path)
        summary = report.summary()
        assert "Total records evaluated: 1" in summary
        assert "Approved -> Denied:    1" in summary


class TestLoadPolicyJson:
    def test_loads_known_fields(self):
        f = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        json.dump({"max_transaction": 7.5, "daily_budget": 99.0}, f)
        f.close()

        policy = load_policy_json(f.name)
        assert policy.max_transaction == 7.5
        assert policy.daily_budget == 99.0

    def test_ignores_unknown_fields_for_forward_compat(self):
        f = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        json.dump({"max_transaction": 5.0, "future_field_v3": "ignored"}, f)
        f.close()

        policy = load_policy_json(f.name)
        assert policy.max_transaction == 5.0


class TestReplayWithRealAuditLog:
    """End-to-end: spin up a wallet, generate real audit rows, replay them."""

    def test_round_trip_against_unchanged_policy(self):
        from paygraph.gateways.mock import MockGateway
        from paygraph.wallet import AgentWallet

        audit_path = _make_audit_path()
        policy = SpendPolicy(max_transaction=50.0, daily_budget=200.0)
        wallet = AgentWallet(
            gateways=MockGateway(auto_approve=True),
            policy=policy,
            log_path=audit_path,
            verbose=False,
        )
        for amt in (5.0, 12.0, 7.0):
            wallet.request_spend(amt, "Anthropic", "tokens")
        with pytest.raises(Exception):
            wallet.request_spend(500.0, "Anthropic", "too much")

        sim = PolicySimulator(policy)
        report = sim.replay_file(audit_path)

        assert report.total == 4
        assert report.unchanged == 4

    def test_policy_snapshot_is_persisted_in_real_audit_log(self):
        from paygraph.gateways.mock import MockGateway
        from paygraph.wallet import AgentWallet

        audit_path = _make_audit_path()
        wallet = AgentWallet(
            gateways=MockGateway(auto_approve=True),
            policy=SpendPolicy(max_transaction=42.0, daily_budget=99.0),
            log_path=audit_path,
            verbose=False,
        )
        wallet.request_spend(5.0, "Anthropic", "tokens")

        with open(audit_path) as f:
            row = json.loads(f.readline())

        assert row["policy_snapshot"] is not None
        assert row["policy_snapshot"]["max_transaction"] == 42.0
        assert row["policy_snapshot"]["daily_budget"] == 99.0
