"""Reversible packet-size and volume transformations."""

from __future__ import annotations

import hashlib
import os
import random

from .policy import EpochPlan, FlowConfig
from .protocol import Frame, FrameType


class TransformationPlanner:
    def __init__(self, config: FlowConfig):
        self.config = config

    def _rng(self, epoch: int, object_id: int) -> random.Random:
        material = f"{self.config.seed}:{epoch}:{object_id}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
        return random.Random(value)

    def transform(
        self,
        data: bytes,
        object_id: int,
        plan: EpochPlan,
    ) -> list[Frame]:
        if not data:
            raise ValueError("cannot transform an empty application object")
        rng = self._rng(plan.index, object_id)
        target_size = rng.choice(plan.policy.target_sizes)
        fragment_count = (len(data) + target_size - 1) // target_size
        frames: list[Frame] = []
        for index in range(fragment_count):
            fragment = data[index * target_size : (index + 1) * target_size]
            padding = os.urandom(target_size - len(fragment))
            frames.append(
                Frame(
                    frame_type=FrameType.DATA,
                    epoch=plan.index,
                    object_id=object_id,
                    fragment_index=index,
                    fragment_count=fragment_count,
                    valid_length=len(fragment),
                    payload=fragment + padding,
                    policy_index=plan.policy_index,
                )
            )
        return frames

    @staticmethod
    def dummy(plan: EpochPlan, target_size: int) -> Frame:
        return Frame(
            frame_type=FrameType.DUMMY,
            epoch=plan.index,
            valid_length=0,
            payload=os.urandom(target_size),
            policy_index=plan.policy_index,
        )
