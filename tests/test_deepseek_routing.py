import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import os
import sys

# Add the project directory to sys.path
sys.path.append("/home/op/mcp-llm-router")

# Mock FastMCP before importing server
mock_fastmcp = MagicMock()
mock_mcp_instance = MagicMock()
# The decorator should return the original function
# For async functions, we can return the function itself, as we will await it in tests
mock_mcp_instance.tool.return_value = lambda x: x
mock_fastmcp.FastMCP.return_value = mock_mcp_instance
sys.modules["fastmcp"] = mock_fastmcp

mock_mcp_pkg = MagicMock()
sys.modules["mcp"] = mock_mcp_pkg
sys.modules["mcp.client"] = MagicMock()
sys.modules["mcp.client.stdio"] = MagicMock()

# Now import the server
from mcp_llm_router.server import agent_llm_request, sessions, _get_api_key


class TestDeepSeekRouting(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create a dummy session
        self.session_id = "test-session"
        sessions[self.session_id] = {"session_id": self.session_id, "events": []}

    @patch("builtins.open", new_callable=MagicMock)
    @patch("os.path.exists")
    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}, clear=False)
    def test_get_api_key_fallback(self, mock_exists, mock_open):
        mock_exists.return_value = True
        # Simulate .bashrc content
        bashrc_content = 'export DEEPSEEK_API_KEY="sk-bashrc-key"\n'
        mock_open.return_value.__enter__.return_value.read.return_value = bashrc_content

        key = _get_api_key("DEEPSEEK_API_KEY")
        self.assertEqual(key, "sk-bashrc-key")

    @patch("mcp_llm_router.server.httpx.AsyncClient")
    @patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "sk-dummy-openai", "DEEPSEEK_API_KEY": "sk-dummy-deepseek"},
    )
    async def test_deepseek_routing_automatic(self, mock_client_cls):
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from DeepSeek"}}],
            "usage": {},
        }
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        # Call the function with a deepseek model
        result = await agent_llm_request(
            session_id=self.session_id, prompt="Hello", model="deepseek-chat"
        )

        # Assertions
        self.assertTrue(result["success"])

        # Check that the correct URL was called
        mock_client.post.assert_called_once()
        args, kwargs = mock_client.post.call_args

        url = args[0]
        headers = kwargs["headers"]

        self.assertEqual(url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer sk-dummy-deepseek")

    @patch("mcp_llm_router.server.httpx.AsyncClient")
    @patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "sk-dummy-openai", "DEEPSEEK_API_KEY": "sk-dummy-deepseek"},
    )
    async def test_normal_routing(self, mock_client_cls):
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from OpenAI"}}],
            "usage": {},
        }
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        # Call with normal model
        await agent_llm_request(
            session_id=self.session_id, prompt="Hello", model="gpt-4"
        )

        # Verify OpenAI URL and Key
        args, kwargs = mock_client.post.call_args
        self.assertEqual(args[0], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sk-dummy-openai")


if __name__ == "__main__":
    unittest.main()
