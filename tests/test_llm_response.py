#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartInterruptionFrame,
)
from pipecat.processors.aggregators.llm_response import LLMFullResponseAggregator
from pipecat.tests.utils import SleepFrame, run_test


class TestLLMFullResponseAggregator(unittest.IsolatedAsyncioTestCase):
    async def test_empty(self):
        completion_ok = False

        aggregator = LLMFullResponseAggregator()

        @aggregator.event_handler("on_completion")
        async def on_completion(aggregator, completion):
            nonlocal completion_ok
            completion_ok = completion == ""

        frames_to_send = [LLMFullResponseStartFrame(), LLMFullResponseEndFrame()]
        expected_down_frames = [LLMFullResponseStartFrame, LLMFullResponseEndFrame]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert completion_ok

    async def test_simple(self):
        completion_ok = False

        aggregator = LLMFullResponseAggregator()

        @aggregator.event_handler("on_completion")
        async def on_completion(aggregator, completion):
            nonlocal completion_ok
            completion_ok = completion == "Hello from Pipecat!"

        frames_to_send = [
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello from Pipecat!"),
            LLMFullResponseEndFrame(),
        ]
        expected_down_frames = [LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert completion_ok

    async def test_multiple(self):
        completion_ok = False

        aggregator = LLMFullResponseAggregator()

        @aggregator.event_handler("on_completion")
        async def on_completion(aggregator, completion):
            nonlocal completion_ok
            completion_ok = completion == "Hello from Pipecat!"

        frames_to_send = [
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello "),
            LLMTextFrame("from "),
            LLMTextFrame("Pipecat!"),
            LLMFullResponseEndFrame(),
        ]
        expected_down_frames = [
            LLMFullResponseStartFrame,
            LLMTextFrame,
            LLMTextFrame,
            LLMTextFrame,
            LLMFullResponseEndFrame,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert completion_ok

    async def test_interruption(self):
        completion_ok = False

        aggregator = LLMFullResponseAggregator()

        @aggregator.event_handler("on_completion")
        async def on_completion(aggregator, completion):
            nonlocal completion_ok
            completion_ok = completion == "Hello there!"

        frames_to_send = [
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello "),
            SleepFrame(),
            StartInterruptionFrame(),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello "),
            LLMTextFrame("there!"),
            LLMFullResponseEndFrame(),
        ]
        expected_down_frames = [
            LLMFullResponseStartFrame,
            LLMTextFrame,
            StartInterruptionFrame,
            LLMFullResponseStartFrame,
            LLMTextFrame,
            LLMTextFrame,
            LLMFullResponseEndFrame,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert completion_ok
