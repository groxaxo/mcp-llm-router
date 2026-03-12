import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_llm_router.server import agent_llm_request, sessions


class TestDeepSeekRouting(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session_id = "test-session"
        sessions[self.session_id] = {"session_id": self.session_id, "events": []}

    @patch("mcp_llm_router.server.brain_client.chat", new_callable=AsyncMock)
    @patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "sk-dummy-openai", "DEEPSEEK_API_KEY": "sk-dummy-deepseek"},
    )
    async def test_deepseek_routing_automatic(self, mock_chat):
        mock_chat.return_value = {
            "choices": [{"message": {"content": "Hello from DeepSeek"}}],
            "usage": {},
        }

        result = await agent_llm_request(
            session_id=self.session_id, prompt="Hello", model="deepseek-chat"
        )

        self.assertTrue(result["success"])
        mock_chat.assert_awaited_once()
        messages, config = mock_chat.await_args.args
        self.assertEqual(messages[-1]["content"], "Hello")
        self.assertEqual(config.model, "deepseek-chat")
        self.assertEqual(config.api_key_env, "OPENAI_API_KEY")

    @patch("mcp_llm_router.server.brain_client.chat", new_callable=AsyncMock)
    @patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "sk-dummy-openai", "DEEPSEEK_API_KEY": "sk-dummy-deepseek"},
    )
    async def test_normal_routing(self, mock_chat):
        mock_chat.return_value = {
            "choices": [{"message": {"content": "Hello from OpenAI"}}],
            "usage": {},
        }

        await agent_llm_request(
            session_id=self.session_id, prompt="Hello", model="gpt-4"
        )

        mock_chat.assert_awaited_once()
        _messages, config = mock_chat.await_args.args
        self.assertEqual(config.model, "gpt-4")
        self.assertEqual(config.api_key_env, "OPENAI_API_KEY")


if __name__ == "__main__":
    unittest.main()
