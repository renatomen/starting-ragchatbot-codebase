"""
Tests for AIGenerator in ai_generator.py

Key test areas:
1. generate_response() - direct text response (no tools)
2. generate_response() - tool_use triggering
3. _execute_tool_loop() - tool execution flow
4. Sequential tool calling - up to 2 rounds
5. Conversation history handling
"""

import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from ai_generator import AIGenerator


class TestAIGeneratorInit:
    """Tests for AIGenerator initialization"""

    def test_init_creates_client(self):
        """Test that initialization creates Anthropic client"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            # Act
            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

            # Assert
            mock_anthropic.assert_called_once_with(api_key="test-key")
            assert generator.model == "claude-sonnet-4-20250514"

    def test_init_sets_base_params(self):
        """Test that base parameters are set correctly"""
        with patch('ai_generator.anthropic.Anthropic'):
            # Act
            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

            # Assert
            assert generator.base_params["model"] == "claude-sonnet-4-20250514"
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800


class TestGenerateResponseDirect:
    """Tests for generate_response() returning direct text (no tool use)"""

    def test_generate_response_returns_text(self, mock_text_response):
        """Test that direct text response is returned correctly"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            # Arrange
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_text_response
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

            # Act
            result = generator.generate_response(query="What is Python?")

            # Assert
            assert result == "This is a direct answer about course materials."

    def test_generate_response_without_tools(self, mock_text_response):
        """Test that API is called without tools when none provided"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            # Arrange
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_text_response
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="test-model")

            # Act
            generator.generate_response(query="test query")

            # Assert
            call_kwargs = mock_client.messages.create.call_args[1]
            assert "tools" not in call_kwargs

    def test_generate_response_with_conversation_history(self, mock_text_response):
        """Test that conversation history is included in system prompt"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            # Arrange
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_text_response
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="test-model")
            history = "User: Previous question\nAssistant: Previous answer"

            # Act
            generator.generate_response(query="test", conversation_history=history)

            # Assert
            call_kwargs = mock_client.messages.create.call_args[1]
            assert "Previous conversation" in call_kwargs["system"]
            assert history in call_kwargs["system"]

    def test_generate_response_without_history(self, mock_text_response):
        """Test that system prompt works without conversation history"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            # Arrange
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_text_response
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="test-model")

            # Act
            generator.generate_response(query="test", conversation_history=None)

            # Assert
            call_kwargs = mock_client.messages.create.call_args[1]
            assert "Previous conversation" not in call_kwargs["system"]


class TestGenerateResponseWithToolUse:
    """Tests for generate_response() when Claude requests tool use"""

    def test_generate_response_triggers_tool_execution(
        self, mock_tool_use_response, mock_final_response_after_tool, mock_tool_manager
    ):
        """Test that tool use response triggers _handle_tool_execution"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            # Arrange
            mock_client = Mock()
            # First call returns tool_use, second call returns final response
            mock_client.messages.create.side_effect = [
                mock_tool_use_response,
                mock_final_response_after_tool
            ]
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="test-model")
            tools = mock_tool_manager.get_tool_definitions()

            # Act
            result = generator.generate_response(
                query="search for ML content",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Assert
            assert result == "Based on the course materials, machine learning is a field of AI."
            mock_tool_manager.execute_tool.assert_called_once()

    def test_generate_response_includes_tools_in_api_call(self, mock_text_response, mock_tool_manager):
        """Test that tools are included in API call when provided"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            # Arrange
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_text_response
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="test-model")
            tools = mock_tool_manager.get_tool_definitions()

            # Act
            generator.generate_response(query="test", tools=tools, tool_manager=mock_tool_manager)

            # Assert
            call_kwargs = mock_client.messages.create.call_args[1]
            assert "tools" in call_kwargs
            assert call_kwargs["tool_choice"] == {"type": "auto"}


class TestExecuteToolLoop:
    """Tests for _execute_tool_loop() method"""

    def test_execute_tool_loop_calls_tool_manager(
        self, mock_tool_use_response, mock_final_response_after_tool, mock_tool_manager
    ):
        """Test that tool manager is called with correct parameters"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            # Arrange
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_final_response_after_tool
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="test-model")
            api_params = {
                "messages": [{"role": "user", "content": "test query"}],
                "system": "system prompt"
            }
            tools = mock_tool_manager.get_tool_definitions()

            # Act
            generator._execute_tool_loop(
                mock_tool_use_response,
                api_params,
                mock_tool_manager,
                tools
            )

            # Assert
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="machine learning basics"
            )

    def test_execute_tool_loop_sends_tool_results_to_claude(
        self, mock_tool_use_response, mock_final_response_after_tool, mock_tool_manager
    ):
        """Test that tool results are sent back to Claude"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            # Arrange
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_final_response_after_tool
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="test-model")
            api_params = {
                "messages": [{"role": "user", "content": "test query"}],
                "system": "system prompt"
            }
            tools = mock_tool_manager.get_tool_definitions()

            # Act
            generator._execute_tool_loop(
                mock_tool_use_response,
                api_params,
                mock_tool_manager,
                tools
            )

            # Assert
            # Check that the API call includes tool results
            call_kwargs = mock_client.messages.create.call_args[1]
            messages = call_kwargs["messages"]

            # Should have: user message, assistant tool_use, user tool_result
            assert len(messages) == 3
            assert messages[2]["role"] == "user"
            assert messages[2]["content"][0]["type"] == "tool_result"

    def test_execute_tool_loop_returns_final_response(
        self, mock_tool_use_response, mock_final_response_after_tool, mock_tool_manager
    ):
        """Test that final Claude response is returned"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            # Arrange
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_final_response_after_tool
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="test-model")
            api_params = {
                "messages": [{"role": "user", "content": "test"}],
                "system": "system"
            }
            tools = mock_tool_manager.get_tool_definitions()

            # Act
            result = generator._execute_tool_loop(
                mock_tool_use_response,
                api_params,
                mock_tool_manager,
                tools
            )

            # Assert
            assert result == "Based on the course materials, machine learning is a field of AI."

    def test_execute_tool_loop_handles_multiple_tool_calls_in_one_response(
        self, mock_final_response_after_tool, mock_tool_manager
    ):
        """Test handling multiple tool calls in single response"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            # Arrange
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_final_response_after_tool
            mock_anthropic.return_value = mock_client

            # Create response with multiple tool uses
            mock_multi_tool_response = Mock()
            mock_multi_tool_response.stop_reason = "tool_use"

            tool_block_1 = Mock()
            tool_block_1.type = "tool_use"
            tool_block_1.id = "tool_1"
            tool_block_1.name = "search_course_content"
            tool_block_1.input = {"query": "topic 1"}

            tool_block_2 = Mock()
            tool_block_2.type = "tool_use"
            tool_block_2.id = "tool_2"
            tool_block_2.name = "search_course_content"
            tool_block_2.input = {"query": "topic 2"}

            mock_multi_tool_response.content = [tool_block_1, tool_block_2]

            generator = AIGenerator(api_key="test-key", model="test-model")
            api_params = {
                "messages": [{"role": "user", "content": "test"}],
                "system": "system"
            }
            tools = mock_tool_manager.get_tool_definitions()

            # Act
            generator._execute_tool_loop(
                mock_multi_tool_response,
                api_params,
                mock_tool_manager,
                tools
            )

            # Assert
            assert mock_tool_manager.execute_tool.call_count == 2


class TestSequentialToolCalling:
    """Tests for sequential tool calling (up to 2 rounds)"""

    def test_single_round_tool_use_returns_after_one_round(
        self, mock_tool_use_response, mock_final_response_after_tool, mock_tool_manager
    ):
        """Test that a single tool call works as before"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            # Arrange
            mock_client = Mock()
            mock_client.messages.create.side_effect = [
                mock_tool_use_response,
                mock_final_response_after_tool
            ]
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="test-model")
            tools = mock_tool_manager.get_tool_definitions()

            # Act
            result = generator.generate_response(
                query="search for ML content",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Assert - 2 API calls: initial + after tool execution
            assert mock_client.messages.create.call_count == 2
            assert result == "Based on the course materials, machine learning is a field of AI."
            mock_tool_manager.execute_tool.assert_called_once()

    def test_two_sequential_tool_calls_work(
        self, mock_tool_use_response, mock_second_tool_use_response,
        mock_final_response_after_tool, mock_tool_manager
    ):
        """Test that Claude can make 2 sequential tool calls"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            # Arrange
            mock_client = Mock()
            # Round 1: tool_use -> Round 2: tool_use -> Final: text
            mock_client.messages.create.side_effect = [
                mock_tool_use_response,       # Initial call - Claude wants tool
                mock_second_tool_use_response, # After first tool - Claude wants another
                mock_final_response_after_tool # After second tool - Claude gives answer
            ]
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="test-model")
            tools = mock_tool_manager.get_tool_definitions()

            # Act
            result = generator.generate_response(
                query="compare ML topics",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Assert - 3 API calls total
            assert mock_client.messages.create.call_count == 3
            assert mock_tool_manager.execute_tool.call_count == 2
            assert result == "Based on the course materials, machine learning is a field of AI."

    def test_max_rounds_enforced(
        self, mock_tool_use_response, mock_second_tool_use_response,
        mock_final_response_after_tool, mock_tool_manager
    ):
        """Test that tool calling stops after MAX_TOOL_ROUNDS"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            # Arrange
            mock_client = Mock()
            # Claude keeps requesting tools, but we should stop after 2 rounds
            mock_client.messages.create.side_effect = [
                mock_tool_use_response,        # Initial - tool use
                mock_tool_use_response,        # Round 1 - tool use
                mock_final_response_after_tool # Round 2 - forced to stop (no tools passed)
            ]
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="test-model")
            tools = mock_tool_manager.get_tool_definitions()

            # Act
            result = generator.generate_response(
                query="complex query",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Assert - should make 3 API calls (initial + 2 rounds)
            assert mock_client.messages.create.call_count == 3
            # Should execute tools twice (once per round)
            assert mock_tool_manager.execute_tool.call_count == 2

    def test_tools_included_in_first_round_api_call(
        self, mock_tool_use_response, mock_second_tool_use_response,
        mock_final_response_after_tool, mock_tool_manager
    ):
        """Test that tools are included in intermediate API calls"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            # Arrange
            mock_client = Mock()
            mock_client.messages.create.side_effect = [
                mock_tool_use_response,
                mock_second_tool_use_response,
                mock_final_response_after_tool
            ]
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="test-model")
            tools = mock_tool_manager.get_tool_definitions()

            # Act
            generator.generate_response(
                query="test",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Assert - check the second API call (first round) includes tools
            calls = mock_client.messages.create.call_args_list
            second_call_kwargs = calls[1][1]
            assert "tools" in second_call_kwargs
            assert second_call_kwargs["tool_choice"] == {"type": "auto"}

    def test_tools_disabled_on_final_round(
        self, mock_tool_use_response, mock_second_tool_use_response,
        mock_final_response_after_tool, mock_tool_manager
    ):
        """Test that final API call (round == MAX) excludes tools"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            # Arrange
            mock_client = Mock()
            mock_client.messages.create.side_effect = [
                mock_tool_use_response,
                mock_tool_use_response,  # Round 1 returns tool_use
                mock_final_response_after_tool
            ]
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="test-model")
            tools = mock_tool_manager.get_tool_definitions()

            # Act
            generator.generate_response(
                query="test",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Assert - third call (final round) should NOT have tools
            calls = mock_client.messages.create.call_args_list
            third_call_kwargs = calls[2][1]
            assert "tools" not in third_call_kwargs

    def test_tool_error_returns_gracefully(self, mock_tool_use_response, mock_tool_manager):
        """Test that tool errors are handled gracefully"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            # Arrange
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_tool_use_response
            mock_anthropic.return_value = mock_client

            # Make tool execution raise an error
            mock_tool_manager.execute_tool.side_effect = Exception("Tool failed!")

            generator = AIGenerator(api_key="test-key", model="test-model")
            tools = mock_tool_manager.get_tool_definitions()

            # Act
            result = generator.generate_response(
                query="test",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Assert - should return error message, not crash
            assert "Tool error" in result
            assert "Tool failed!" in result

    def test_message_accumulation_across_rounds(
        self, mock_tool_use_response, mock_second_tool_use_response,
        mock_final_response_after_tool, mock_tool_manager
    ):
        """Test that messages build correctly across rounds"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            # Arrange
            mock_client = Mock()
            mock_client.messages.create.side_effect = [
                mock_tool_use_response,
                mock_second_tool_use_response,
                mock_final_response_after_tool
            ]
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="test-model")
            tools = mock_tool_manager.get_tool_definitions()

            # Act
            generator.generate_response(
                query="test query",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Assert - final API call should have accumulated messages
            calls = mock_client.messages.create.call_args_list
            final_call_kwargs = calls[2][1]
            messages = final_call_kwargs["messages"]

            # Should have: user, assistant (tool_use 1), user (tool_result 1),
            #              assistant (tool_use 2), user (tool_result 2)
            assert len(messages) == 5
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"
            assert messages[2]["role"] == "user"
            assert messages[3]["role"] == "assistant"
            assert messages[4]["role"] == "user"


class TestGenerateResponseEdgeCases:
    """Edge case tests for generate_response"""

    def test_tool_use_without_tool_manager(self, mock_tool_use_response):
        """Test that tool_use without tool_manager returns response content"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            # Arrange
            mock_client = Mock()
            # Return a response that has text content even with tool_use stop_reason
            text_block = Mock()
            text_block.type = "text"
            text_block.text = "I would use a tool but cannot"
            mock_tool_use_response.content = [text_block]
            mock_client.messages.create.return_value = mock_tool_use_response
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="test-model")

            # Act
            result = generator.generate_response(
                query="test",
                tools=[{"name": "test_tool"}],
                tool_manager=None  # No tool manager
            )

            # Assert
            assert result == "I would use a tool but cannot"

    def test_generate_response_with_empty_tools_list(self, mock_text_response):
        """Test behavior with empty tools list - treated as no tools"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            # Arrange
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_text_response
            mock_anthropic.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="test-model")

            # Act
            result = generator.generate_response(query="test", tools=[])

            # Assert
            # Empty list is falsy in Python, so no tools are added to API params
            # This is correct behavior - no point sending empty tools to API
            call_kwargs = mock_client.messages.create.call_args[1]
            assert "tools" not in call_kwargs
