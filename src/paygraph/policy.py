from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta


@dataclass
class SpendPolicy:
    """Configuration for spend governance rules.

    Attributes:
        max_transaction: Maximum dollar amount allowed per transaction.
        daily_budget: Maximum total dollar amount allowed per calendar day.
        allowed_vendors: If set, only vendors matching these names are
            permitted (case-insensitive substring match).
        blocked_vendors: If set, vendors matching these names are always
            blocked (case-insensitive substring match).
        allowed_mccs: Merchant Category Code allowlist (reserved for future use).
        require_justification: Whether a justification string is required
            for every spend request.
        hourly_budget: Maximum total dollar amount allowed per hour.
            If None, no hourly limit is enforced.
        weekly_budget: Maximum total dollar amount allowed per week.
            If None, no weekly limit is enforced.
        monthly_budget: Maximum total dollar amount allowed per month.
            If None, no monthly limit is enforced.
        require_human_approval_above: If set, spends above this dollar amount
            require human approval via Slack before the gateway is called.
    """

    max_transaction: float = 50.0
    daily_budget: float = 200.0
    allowed_vendors: list[str] | None = None
    blocked_vendors: list[str] | None = None
    allowed_mccs: list[int] | None = None
    require_justification: bool = True
    hourly_budget: float | None = None
    weekly_budget: float | None = None
    monthly_budget: float | None = None
    require_human_approval_above: float | None = None


@dataclass
class PolicyResult:
    """Result of a policy evaluation.

    Attributes:
        approved: Whether the spend request passed all policy checks.
        denial_reason: Human-readable reason if the request was denied.
        checks_passed: Names of policy checks that passed before denial
            (or all checks if approved).
    """

    approved: bool
    denial_reason: str | None = None
    checks_passed: list[str] = field(default_factory=list)


class PolicyEngine:
    """Stateful engine that evaluates spend requests against policy rules.

    Tracks cumulative spend in memory per period. Counters reset automatically
    at period boundaries (hourly on hour change, daily on date change,
    weekly on Monday-start week boundary, monthly on month boundary).
    """

    def __init__(self, policy: SpendPolicy) -> None:
        """Initialize the engine with a spend policy.

        Args:
            policy: The ``SpendPolicy`` defining governance rules.
        """
        self.policy = policy
        self._daily_spend: float = 0.0
        self._current_date: date = date.today()

        self._hourly_spend: float = 0.0
        self._hourly_start: datetime | None = None

        self._weekly_spend: float = 0.0
        self._weekly_start: datetime | None = None

        self._monthly_spend: float = 0.0
        self._monthly_start: datetime | None = None

    def _reset_daily_if_needed(self) -> None:
        today = date.today()
        if today != self._current_date:
            self._daily_spend = 0.0
            self._current_date = today

    def _reset_hourly_if_needed(self, now: datetime) -> None:
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        if self._hourly_start != hour_start:
            self._hourly_spend = 0.0
            self._hourly_start = hour_start

    def _reset_weekly_if_needed(self, now: datetime) -> None:
        monday = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        if self._weekly_start != monday:
            self._weekly_spend = 0.0
            self._weekly_start = monday

    def _reset_monthly_if_needed(self, now: datetime) -> None:
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if self._monthly_start != month_start:
            self._monthly_spend = 0.0
            self._monthly_start = month_start

    def evaluate(
        self,
        amount: float,
        vendor: str,
        justification: str | None = None,
        on_check: Callable[[str, bool], None] | None = None,
        *,
        now: datetime | None = None,
    ) -> PolicyResult:
        """Evaluate a spend request against all policy rules.

        Checks are run in order: positive_amount, amount_cap, vendor_allowlist,
        vendor_blocklist, mcc_filter, hourly_budget (if configured),
        weekly_budget (if configured), monthly_budget (if configured),
        daily_budget, justification.
        Evaluation stops at the first failure.

        Args:
            amount: Dollar amount of the spend request.
            vendor: Name of the vendor or service.
            justification: Reason for the spend (required if
                ``policy.require_justification`` is True).
            on_check: Optional callback invoked after each check with
                ``(check_name, passed)``.
            now: Override current time (for deterministic testing).

        Returns:
            A ``PolicyResult`` indicating approval or denial.
        """
        if now is None:
            now = datetime.now()
        self._reset_daily_if_needed()
        self._reset_hourly_if_needed(now)
        self._reset_weekly_if_needed(now)
        self._reset_monthly_if_needed(now)
        checks_passed: list[str] = []

        def _pass(name: str) -> None:
            checks_passed.append(name)
            if on_check:
                on_check(name, True)

        def _fail(name: str, reason: str) -> PolicyResult:
            if on_check:
                on_check(name, False)
            return PolicyResult(
                approved=False,
                denial_reason=reason,
                checks_passed=checks_passed,
            )

        # 0. Positive amount check
        if amount <= 0:
            return _fail(
                "positive_amount",
                f"Amount must be positive (got ${amount:.2f})",
            )
        _pass("positive_amount")

        # 1. Amount cap
        if amount > self.policy.max_transaction:
            return _fail(
                "amount_cap",
                f"Amount ${amount:.2f} exceeds limit of ${self.policy.max_transaction:.2f}",
            )
        _pass("amount_cap")

        # 2. Vendor allowlist / blocklist
        vendor_lower = vendor.lower()
        if self.policy.allowed_vendors is not None:
            if not any(v.lower() in vendor_lower for v in self.policy.allowed_vendors):
                return _fail(
                    "vendor_allowlist", f"Vendor '{vendor}' is not in the allowed list"
                )
        _pass("vendor_allowlist")

        if self.policy.blocked_vendors is not None:
            if any(v.lower() in vendor_lower for v in self.policy.blocked_vendors):
                return _fail("vendor_blocklist", f"Vendor '{vendor}' is blocked")
        _pass("vendor_blocklist")

        # 3. MCC filter (stubbed — no MCC in spend request yet)
        _pass("mcc_filter")

        # 4. Time-based budget checks — only enforced and reported when configured
        if self.policy.hourly_budget is not None:
            if self._hourly_spend + amount > self.policy.hourly_budget:
                return _fail(
                    "hourly_budget",
                    f"Hourly budget exhausted (${self._hourly_spend:.2f} / ${self.policy.hourly_budget:.2f})",
                )
            _pass("hourly_budget")

        if self.policy.weekly_budget is not None:
            if self._weekly_spend + amount > self.policy.weekly_budget:
                return _fail(
                    "weekly_budget",
                    f"Weekly budget exhausted (${self._weekly_spend:.2f} / ${self.policy.weekly_budget:.2f})",
                )
            _pass("weekly_budget")

        if self.policy.monthly_budget is not None:
            if self._monthly_spend + amount > self.policy.monthly_budget:
                return _fail(
                    "monthly_budget",
                    f"Monthly budget exhausted (${self._monthly_spend:.2f} / ${self.policy.monthly_budget:.2f})",
                )
            _pass("monthly_budget")

        # 5. Daily budget (existing logic)
        if self._daily_spend + amount > self.policy.daily_budget:
            return _fail(
                "daily_budget",
                f"Daily budget exhausted (${self._daily_spend:.2f} / ${self.policy.daily_budget:.2f})",
            )
        _pass("daily_budget")

        # 6. Justification present
        if self.policy.require_justification and not justification:
            return _fail(
                "justification", "Justification is required but was not provided"
            )
        _pass("justification")

        return PolicyResult(approved=True, checks_passed=checks_passed)

    def commit_spend(self, amount: float, *, now: datetime | None = None) -> None:
        """Permanently record a spend against all budget counters.

        Must be called only after a successful gateway transaction so that a
        gateway failure does not silently consume the agent's budget.

        Args:
            amount: Dollar amount that was successfully spent.
            now: Override current time (for deterministic testing).
        """
        if now is None:
            now = datetime.now()
        self._reset_daily_if_needed()
        self._daily_spend += amount
        if self.policy.hourly_budget is not None:
            self._reset_hourly_if_needed(now)
            self._hourly_spend += amount
        if self.policy.weekly_budget is not None:
            self._reset_weekly_if_needed(now)
            self._weekly_spend += amount
        if self.policy.monthly_budget is not None:
            self._reset_monthly_if_needed(now)
            self._monthly_spend += amount
