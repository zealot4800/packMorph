"""Counter-based policy and holding-time schedule for the FlowShift simulation."""

from __future__ import annotations

import hashlib
import hmac
import math


class FlowShiftSchedule:
    DOMAIN = b"flowshift-policy-schedule-v1"

    def __init__(self, seed: str, policy_count: int,
                 minimum_hold: float, maximum_hold: float):
        if (not seed or policy_count < 1 or not math.isfinite(minimum_hold)
                or not math.isfinite(maximum_hold) or minimum_hold <= 0
                or maximum_hold < minimum_hold):
            raise ValueError("Invalid FlowShift schedule settings")
        self.key = seed.encode("utf-8")
        self.policy_count = policy_count
        self.minimum_hold = minimum_hold
        self.maximum_hold = maximum_hold

    def plan(self, interval: int) -> tuple[int, float]:
        if not 0 <= interval < 1 << 64:
            raise ValueError("FlowShift interval is outside the PRG counter range")
        digest = hmac.new(self.key, self.DOMAIN + interval.to_bytes(8, "big"),
                          hashlib.sha256).digest()
        u = int.from_bytes(digest[:8], "big")
        v = int.from_bytes(digest[8:16], "big") / (1 << 64)
        return (u % self.policy_count,
                self.minimum_hold + v * (self.maximum_hold - self.minimum_hold))
