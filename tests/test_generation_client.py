"""
Tests for Groq generation client hardening.

All provider interactions are mocked. No network access, no API key, no model calls.
"""

import unittest
from unittest.mock import MagicMock, patch

from legalrag.generation import client as client_module
from legalrag.generation.client import GenerationError


class TestGenerationClientHardening(unittest.TestCase):
    def _make_client(self, **kwargs):
        fake_groq_cls = MagicMock()
        with patch.object(client_module, "Groq", fake_groq_cls):
            client = client_module.LegalGenerationClient(
                api_key="gsk_unit_test_key_123456",
                model_name="llama-test-model",
                **kwargs,
            )
        return client, fake_groq_cls

    def test_explicit_timeout_and_sdk_retries_disabled(self):
        client, groq_cls = self._make_client(request_timeout=12.5, max_retries=2)
        groq_cls.assert_called_once()
        _, kwargs = groq_cls.call_args
        self.assertEqual(kwargs["timeout"], 12.5)
        self.assertEqual(kwargs["max_retries"], 0)
        self.assertEqual(client.request_timeout, 12.5)

    def test_request_timeout_passed_per_call(self):
        client, _ = self._make_client(request_timeout=7.0)
        response = MagicMock()
        response.usage = None
        choice = MagicMock()
        choice.message.content = "grounded answer"
        choice.finish_reason = "stop"
        response.choices = [choice]
        client.client.chat.completions.create.return_value = response

        result = client.generate("system", "user")
        self.assertTrue(result.is_success)
        _, kwargs = client.client.chat.completions.create.call_args
        self.assertEqual(kwargs["timeout"], 7.0)
        self.assertEqual(kwargs["model"], "llama-test-model")
        self.assertEqual(kwargs["temperature"], 0.0)

    def test_bounded_retries_then_controlled_502_error(self):
        client, _ = self._make_client(max_retries=2)
        client._is_retryable = lambda exc: True
        create = client.client.chat.completions.create
        create.side_effect = RuntimeError("provider boom with gsk_unit_test_key_123456")

        with patch.object(client_module.time, "sleep") as sleeper:
            with self.assertRaises(GenerationError) as ctx:
                client.generate("system", "user")

        self.assertEqual(create.call_count, 3)  # 1 initial + 2 bounded retries
        self.assertEqual(sleeper.call_count, 2)
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertNotIn("gsk_unit_test_key_123456", str(ctx.exception))

    def test_non_retryable_failure_fails_fast(self):
        client, _ = self._make_client(max_retries=3)
        create = client.client.chat.completions.create
        create.side_effect = RuntimeError("non-transient failure")

        with patch.object(client_module.time, "sleep") as sleeper:
            with self.assertRaises(GenerationError):
                client.generate("system", "user")

        self.assertEqual(create.call_count, 1)
        self.assertEqual(sleeper.call_count, 0)

    def test_empty_response_raises_controlled_error(self):
        client, _ = self._make_client()
        response = MagicMock()
        response.usage = None
        choice = MagicMock()
        choice.message.content = "   "
        choice.finish_reason = "stop"
        response.choices = [choice]
        client.client.chat.completions.create.return_value = response

        with self.assertRaises(GenerationError) as ctx:
            client.generate("system", "user")
        self.assertEqual(ctx.exception.status_code, 502)

    def test_is_retryable_classifies_generic_exception_as_false(self):
        client, _ = self._make_client()
        self.assertFalse(client._is_retryable(RuntimeError("x")))


if __name__ == "__main__":
    unittest.main()
