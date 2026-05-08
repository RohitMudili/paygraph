"""Tests for time-based spending policies (hourly, weekly, monthly)."""

from datetime import datetime

from paygraph.policy import PolicyEngine, SpendPolicy


class TestHourlyBudget:
    def test_enforced_within_hour(self):
        engine = PolicyEngine(SpendPolicy(max_transaction=100.0, daily_budget=1000.0, hourly_budget=50.0))
        t = datetime(2024, 1, 15, 14, 30, 0)
        result = engine.evaluate(30.0, "vendor", "reason", now=t)
        assert result.approved
        assert "hourly_budget" in result.checks_passed
        engine.commit_spend(30.0, now=t)

        result = engine.evaluate(25.0, "vendor", "reason", now=t.replace(minute=45))
        assert not result.approved
        assert "Hourly budget exhausted" in result.denial_reason

    def test_resets_on_new_hour(self):
        engine = PolicyEngine(SpendPolicy(max_transaction=100.0, daily_budget=1000.0, hourly_budget=100.0))
        t = datetime(2024, 1, 15, 14, 59, 0)
        engine.evaluate(80.0, "vendor", "reason", now=t)
        engine.commit_spend(80.0, now=t)

        result = engine.evaluate(90.0, "vendor", "reason", now=datetime(2024, 1, 15, 15, 1, 0))
        assert result.approved

    def test_zero_budget_blocks_all(self):
        engine = PolicyEngine(SpendPolicy(max_transaction=100.0, daily_budget=1000.0, hourly_budget=0.0))
        result = engine.evaluate(1.0, "vendor", "reason", now=datetime(2024, 1, 15, 14, 30, 0))
        assert not result.approved
        assert "Hourly budget exhausted" in result.denial_reason

    def test_not_in_checks_passed_when_unconfigured(self):
        engine = PolicyEngine(SpendPolicy(max_transaction=100.0, daily_budget=1000.0))
        result = engine.evaluate(50.0, "vendor", "reason")
        assert result.approved
        assert "hourly_budget" not in result.checks_passed


class TestWeeklyBudget:
    def test_enforced_across_days(self):
        engine = PolicyEngine(SpendPolicy(max_transaction=200.0, daily_budget=1000.0, weekly_budget=200.0))
        monday = datetime(2024, 1, 15, 14, 0, 0)
        result = engine.evaluate(150.0, "vendor", "reason", now=monday)
        assert result.approved
        assert "weekly_budget" in result.checks_passed
        engine.commit_spend(150.0, now=monday)

        wednesday = datetime(2024, 1, 17, 10, 0, 0)
        result = engine.evaluate(75.0, "vendor", "reason", now=wednesday)
        assert not result.approved
        assert "Weekly budget exhausted" in result.denial_reason

    def test_resets_on_monday(self):
        engine = PolicyEngine(SpendPolicy(max_transaction=200.0, daily_budget=1000.0, weekly_budget=200.0))
        monday = datetime(2024, 1, 15, 14, 0, 0)
        engine.evaluate(150.0, "vendor", "reason", now=monday)
        engine.commit_spend(150.0, now=monday)

        next_monday = datetime(2024, 1, 22, 10, 0, 0)
        result = engine.evaluate(150.0, "vendor", "reason", now=next_monday)
        assert result.approved

    def test_not_in_checks_passed_when_unconfigured(self):
        engine = PolicyEngine(SpendPolicy(max_transaction=100.0, daily_budget=1000.0))
        result = engine.evaluate(50.0, "vendor", "reason")
        assert result.approved
        assert "weekly_budget" not in result.checks_passed


class TestMonthlyBudget:
    def test_enforced_across_days(self):
        engine = PolicyEngine(SpendPolicy(max_transaction=1500.0, daily_budget=2000.0, monthly_budget=1500.0))
        t = datetime(2024, 1, 15, 14, 0, 0)
        result = engine.evaluate(1200.0, "vendor", "reason", now=t)
        assert result.approved
        assert "monthly_budget" in result.checks_passed
        engine.commit_spend(1200.0, now=t)

        later = datetime(2024, 1, 25, 10, 0, 0)
        result = engine.evaluate(400.0, "vendor", "reason", now=later)
        assert not result.approved
        assert "Monthly budget exhausted" in result.denial_reason

    def test_resets_on_new_month(self):
        engine = PolicyEngine(SpendPolicy(max_transaction=1500.0, daily_budget=5000.0, monthly_budget=1500.0))
        t = datetime(2024, 1, 15, 14, 0, 0)
        engine.evaluate(1200.0, "vendor", "reason", now=t)
        engine.commit_spend(1200.0, now=t)

        next_month = datetime(2024, 2, 1, 10, 0, 0)
        result = engine.evaluate(1200.0, "vendor", "reason", now=next_month)
        assert result.approved

    def test_not_in_checks_passed_when_unconfigured(self):
        engine = PolicyEngine(SpendPolicy(max_transaction=100.0, daily_budget=1000.0))
        result = engine.evaluate(50.0, "vendor", "reason")
        assert result.approved
        assert "monthly_budget" not in result.checks_passed


class TestMultipleTimeBudgets:
    def test_hourly_blocks_before_weekly(self):
        engine = PolicyEngine(SpendPolicy(
            max_transaction=200.0,
            daily_budget=1000.0,
            hourly_budget=100.0,
            weekly_budget=500.0,
        ))
        t = datetime(2024, 1, 15, 14, 30, 0)
        engine.evaluate(80.0, "vendor", "reason", now=t)
        engine.commit_spend(80.0, now=t)

        result = engine.evaluate(30.0, "vendor", "reason", now=t)
        assert not result.approved
        assert "Hourly budget exhausted" in result.denial_reason

    def test_all_time_budgets_in_checks_passed(self):
        engine = PolicyEngine(SpendPolicy(
            max_transaction=100.0,
            daily_budget=1000.0,
            hourly_budget=100.0,
            weekly_budget=500.0,
            monthly_budget=2000.0,
        ))
        t = datetime(2024, 1, 15, 14, 30, 0)
        result = engine.evaluate(50.0, "vendor", "reason", now=t)
        assert result.approved
        assert "hourly_budget" in result.checks_passed
        assert "weekly_budget" in result.checks_passed
        assert "monthly_budget" in result.checks_passed

    def test_weekly_blocks_after_hourly_resets(self):
        engine = PolicyEngine(SpendPolicy(
            max_transaction=200.0,
            daily_budget=2000.0,
            hourly_budget=200.0,
            weekly_budget=300.0,
        ))
        monday_h1 = datetime(2024, 1, 15, 10, 0, 0)
        engine.evaluate(200.0, "vendor", "reason", now=monday_h1)
        engine.commit_spend(200.0, now=monday_h1)

        # New hour — hourly resets, but weekly does not
        monday_h2 = datetime(2024, 1, 15, 11, 0, 0)
        result = engine.evaluate(150.0, "vendor", "reason", now=monday_h2)
        assert not result.approved
        assert "Weekly budget exhausted" in result.denial_reason
