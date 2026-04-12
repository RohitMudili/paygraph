import secrets

import httpx

from paygraph.exceptions import HumanApprovalRequired, SpendDeniedError
from paygraph.gateways.base import BaseGateway, VirtualCard


class SlackApprovalGateway(BaseGateway):
    """Gateway that fires a Slack webhook for human approval before spending.

    Wraps an inner gateway (e.g. ``MockGateway``, ``StripeCardGateway``).
    When ``request_approval()`` is called, it posts an Approve/Deny message
    to Slack and raises ``HumanApprovalRequired``. The caller must store the
    ``request_id`` and later call ``complete_spend()`` once the human responds.

    Example::

        from paygraph import AgentWallet, SpendPolicy
        from paygraph.gateways.slack import SlackApprovalGateway
        from paygraph.gateways.mock import MockGateway
        from paygraph.exceptions import HumanApprovalRequired

        gateway = SlackApprovalGateway(
            webhook_url="https://hooks.slack.com/services/...",
            inner_gateway=MockGateway(auto_approve=True),
        )
        wallet = AgentWallet(
            gateway=gateway,
            policy=SpendPolicy(require_human_approval_above=20.0),
        )

        try:
            wallet.request_spend(50.0, "Anthropic", "need tokens")
        except HumanApprovalRequired as e:
            # Store e.request_id — resume later when human responds
            result = wallet.complete_spend(e.request_id, approved=True)
    """

    def __init__(self, webhook_url: str, inner_gateway: BaseGateway) -> None:
        """Initialise the Slack approval gateway.

        Args:
            webhook_url: Slack incoming webhook URL to post approval requests to.
            inner_gateway: Gateway used to execute the spend once approved.
        """
        self.webhook_url = webhook_url
        self.inner_gateway = inner_gateway
        self._pending: dict[str, dict] = {}

    def request_approval(self, amount_cents: int, vendor: str, memo: str) -> None:
        """Post an approval request to Slack and raise ``HumanApprovalRequired``.

        Args:
            amount_cents: Spend amount in cents.
            vendor: Vendor name.
            memo: Justification for the spend.

        Raises:
            HumanApprovalRequired: Always — after successfully posting to Slack.
            GatewayError: If the Slack webhook POST fails.
        """
        request_id = secrets.token_hex(8)
        amount_dollars = amount_cents / 100

        payload = {
            "text": (
                f"*PayGraph Approval Required*\n"
                f"Amount: *${amount_dollars:.2f}*\n"
                f"Vendor: *{vendor}*\n"
                f"Justification: {memo}\n"
                f"Request ID: `{request_id}`"
            ),
            "attachments": [
                {
                    "fallback": "Approve or Deny this spend request",
                    "callback_id": request_id,
                    "actions": [
                        {
                            "name": "approve",
                            "text": "Approve",
                            "type": "button",
                            "value": "approve",
                            "style": "primary",
                        },
                        {
                            "name": "deny",
                            "text": "Deny",
                            "type": "button",
                            "value": "deny",
                            "style": "danger",
                        },
                    ],
                }
            ],
        }

        httpx.post(self.webhook_url, json=payload, timeout=10)
        self._pending[request_id] = {
            "amount_cents": amount_cents,
            "vendor": vendor,
            "memo": memo,
        }
        raise HumanApprovalRequired(request_id, amount_dollars, vendor)

    def execute_spend(self, amount_cents: int, vendor: str, memo: str) -> VirtualCard:
        """Execute a spend directly via the inner gateway (below-threshold path).

        Args:
            amount_cents: Spend amount in cents.
            vendor: Vendor name.
            memo: Justification for the spend.

        Returns:
            A ``VirtualCard`` from the inner gateway.
        """
        return self.inner_gateway.execute_spend(amount_cents, vendor, memo)

    def complete_spend(self, request_id: str, approved: bool) -> VirtualCard:
        """Resume a pending spend after a human has responded in Slack.

        Args:
            request_id: The ``request_id`` from the ``HumanApprovalRequired``
                exception raised by ``request_approval()``.
            approved: ``True`` if the human approved, ``False`` if denied.

        Returns:
            A ``VirtualCard`` from the inner gateway if approved.

        Raises:
            SpendDeniedError: If ``approved`` is ``False``.
            KeyError: If ``request_id`` is unknown or already completed.
        """
        pending = self._pending.pop(request_id)
        if not approved:
            raise SpendDeniedError(
                f"Human denied spend of ${pending['amount_cents'] / 100:.2f} "
                f"for {pending['vendor']}"
            )
        return self.inner_gateway.execute_spend(
            pending["amount_cents"], pending["vendor"], pending["memo"]
        )

    def revoke(self, gateway_ref: str) -> bool:
        """Revoke a card via the inner gateway.

        Args:
            gateway_ref: The ``gateway_ref`` of the card to revoke.

        Returns:
            ``True`` if the card was revoked, ``False`` otherwise.
        """
        return self.inner_gateway.revoke(gateway_ref)
