"""Policy simulator — replay an audit JSONL log against a candidate SpendPolicy.

The simulator answers the question *"what would have happened if this new
policy had been live last week?"* by reading the existing audit log,
re-evaluating each historical request against a candidate ``SpendPolicy``,
and reporting the delta per record (unchanged, flipped to approved, flipped
to denied, or denial-reason changed).

Cumulative state (daily / hourly / weekly / monthly budgets) is reconstructed
by replaying records in chronological order through a fresh ``PolicyEngine``,
using the engine's existing ``now=`` kwarg so the wall clock is irrelevant.

Limitations:
- Budget reconstruction only counts rows that were originally approved (and
  still approve under the candidate policy). Originally-denied rows never
  consumed budget, so re-counting them would inflate cumulative totals.
  Pass ``only_approved=False`` to also re-check denied rows for reporting,
  without affecting the budget reconstruction.
- ``mcc_filter`` is a no-op in the live engine today, so it's also a no-op
  during replay.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime

from paygraph.policy import PolicyEngine, PolicyResult, SpendPolicy

UNCHANGED = "unchanged"
FLIPPED_TO_DENIED = "approved->denied"
FLIPPED_TO_APPROVED = "denied->approved"
DENIAL_REASON_CHANGED = "denial_reason_changed"

_ORIGINAL_APPROVED_RESULTS = {"approved", "pending_approval"}


@dataclass
class ReplayOutcome:
    """One audit record replayed against the candidate policy."""

    timestamp: str
    amount: float
    vendor: str
    justification: str | None
    original_result: str
    original_denial_reason: str | None
    new_result: str
    new_denial_reason: str | None
    delta: str


@dataclass
class ReplayReport:
    """Aggregate result of replaying an audit log against a candidate policy."""

    candidate_policy: dict
    total: int
    unchanged: int
    flipped_to_denied: int
    flipped_to_approved: int
    denial_reason_changed: int
    outcomes: list[ReplayOutcome] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable table for CLI output."""
        lines = [
            "Policy replay report",
            "=" * 50,
            f"Total records evaluated: {self.total}",
            f"  Unchanged:             {self.unchanged}",
            f"  Approved -> Denied:    {self.flipped_to_denied}",
            f"  Denied   -> Approved:  {self.flipped_to_approved}",
            f"  Denial reason changed: {self.denial_reason_changed}",
            "",
        ]
        flipped = [o for o in self.outcomes if o.delta != UNCHANGED]
        if flipped:
            lines.append("Changed outcomes:")
            lines.append("-" * 50)
            for o in flipped:
                lines.append(
                    f"  [{o.delta}] {o.timestamp}  ${o.amount:.2f} -> {o.vendor}"
                )
                if o.new_denial_reason and o.new_denial_reason != o.original_denial_reason:
                    lines.append(f"      new reason: {o.new_denial_reason}")
        return "\n".join(lines)


class PolicySimulator:
    """Replay an audit JSONL log against a candidate ``SpendPolicy``.

    Example:
        ```python
        from paygraph import PolicySimulator, SpendPolicy

        sim = PolicySimulator(SpendPolicy(max_transaction=10.0))
        report = sim.replay_file("paygraph_audit.jsonl")
        print(report.summary())
        ```
    """

    def __init__(self, candidate_policy: SpendPolicy) -> None:
        """Initialize the simulator.

        Args:
            candidate_policy: The ``SpendPolicy`` to evaluate historical
                requests against.
        """
        self.candidate_policy = candidate_policy

    def replay_file(
        self,
        audit_path: str,
        *,
        only_approved: bool = True,
    ) -> ReplayReport:
        """Replay records from a JSONL audit file.

        Args:
            audit_path: Path to a paygraph audit JSONL log file.
            only_approved: If True (default), only previously-approved rows
                consume reconstructed budget. If False, originally-denied
                rows are also counted toward cumulative totals during replay
                (rarely what you want — original denied spends never hit
                the budget).

        Returns:
            A ``ReplayReport`` with per-record outcomes and aggregate counts.
        """
        with open(audit_path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        return self.replay(records, only_approved=only_approved)

    def replay(
        self,
        records: list[dict],
        *,
        only_approved: bool = True,
    ) -> ReplayReport:
        """Replay an in-memory list of audit records (raw dicts).

        Records are sorted by ``timestamp`` so cumulative state is
        reconstructed in chronological order, independent of the order they
        were passed in.
        """
        records = sorted(records, key=lambda r: r["timestamp"])
        engine = PolicyEngine(self.candidate_policy)
        outcomes: list[ReplayOutcome] = []

        for row in records:
            ts = _parse_timestamp(row["timestamp"])
            new_result = engine.evaluate(
                amount=row["amount"],
                vendor=row["vendor"],
                justification=row.get("justification"),
                now=ts,
            )
            originally_approved = row["policy_result"] in _ORIGINAL_APPROVED_RESULTS
            should_commit = new_result.approved and (
                originally_approved or not only_approved
            )
            if should_commit:
                engine.commit_spend(row["amount"], now=ts)

            outcomes.append(_build_outcome(row, new_result))

        return _summarize(outcomes, self.candidate_policy)


def _parse_timestamp(value: str) -> datetime:
    # AuditRecord.now() writes datetime.now(timezone.utc).isoformat(), which
    # fromisoformat handles natively on 3.11+. The "Z" suffix some upstream
    # tools emit is not produced by our logger but we accept it defensively.
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def _build_outcome(row: dict, new: PolicyResult) -> ReplayOutcome:
    original_result = row["policy_result"]
    original_denial_reason = row.get("denial_reason")
    new_result_str = "approved" if new.approved else "denied"
    originally_approved = original_result in _ORIGINAL_APPROVED_RESULTS

    if originally_approved and not new.approved:
        delta = FLIPPED_TO_DENIED
    elif (not originally_approved) and new.approved:
        delta = FLIPPED_TO_APPROVED
    elif (not originally_approved) and (not new.approved) and (
        new.denial_reason != original_denial_reason
    ):
        delta = DENIAL_REASON_CHANGED
    else:
        delta = UNCHANGED

    return ReplayOutcome(
        timestamp=row["timestamp"],
        amount=row["amount"],
        vendor=row["vendor"],
        justification=row.get("justification"),
        original_result=original_result,
        original_denial_reason=original_denial_reason,
        new_result=new_result_str,
        new_denial_reason=new.denial_reason,
        delta=delta,
    )


def _summarize(
    outcomes: list[ReplayOutcome], candidate_policy: SpendPolicy
) -> ReplayReport:
    return ReplayReport(
        candidate_policy=asdict(candidate_policy),
        total=len(outcomes),
        unchanged=sum(1 for o in outcomes if o.delta == UNCHANGED),
        flipped_to_denied=sum(1 for o in outcomes if o.delta == FLIPPED_TO_DENIED),
        flipped_to_approved=sum(
            1 for o in outcomes if o.delta == FLIPPED_TO_APPROVED
        ),
        denial_reason_changed=sum(
            1 for o in outcomes if o.delta == DENIAL_REASON_CHANGED
        ),
        outcomes=outcomes,
    )


def load_policy_json(path: str) -> SpendPolicy:
    """Load a ``SpendPolicy`` from a JSON file.

    Unknown fields are ignored so a snapshot written by a newer paygraph
    version can still be loaded by an older one.
    """
    with open(path) as f:
        data = json.load(f)
    allowed = {f.name for f in SpendPolicy.__dataclass_fields__.values()}
    return SpendPolicy(**{k: v for k, v in data.items() if k in allowed})
