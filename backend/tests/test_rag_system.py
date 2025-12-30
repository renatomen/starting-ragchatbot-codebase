"""
Tests for RAGSystem in rag_system.py

Key test areas:
1. query() - orchestration flow
2. query() - session handling
3. query() - source retrieval and reset
4. Integration between components
"""

import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


class TestRAGSystemInit:
    """Tests for RAGSystem initialization"""

    def test_init_creates_all_components(self, mock_config):
        """Test that all components are initialized"""
        with patch('rag_system.DocumentProcessor') as mock_dp, \
             patch('rag_system.VectorStore') as mock_vs, \
             patch('rag_system.AIGenerator') as mock_ai, \
             patch('rag_system.SessionManager') as mock_sm, \
             patch('rag_system.ToolManager') as mock_tm, \
             patch('rag_system.CourseSearchTool') as mock_cst, \
             patch('rag_system.CourseOutlineTool') as mock_cot:

            from rag_system import RAGSystem

            # Act
            system = RAGSystem(mock_config)

            # Assert
            mock_dp.assert_called_once()
            mock_vs.assert_called_once()
            mock_ai.assert_called_once()
            mock_sm.assert_called_once()

    def test_init_registers_tools(self, mock_config):
        """Test that search tools are registered with ToolManager"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager') as mock_tm, \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):

            from rag_system import RAGSystem
            mock_manager = Mock()
            mock_tm.return_value = mock_manager

            # Act
            system = RAGSystem(mock_config)

            # Assert
            assert mock_manager.register_tool.call_count == 2  # search + outline


class TestRAGSystemQuery:
    """Tests for RAGSystem.query() method"""

    @pytest.fixture
    def rag_system_with_mocks(self, mock_config):
        """Create RAGSystem with mocked dependencies"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as mock_ai_class, \
             patch('rag_system.SessionManager') as mock_sm_class, \
             patch('rag_system.ToolManager') as mock_tm_class, \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):

            from rag_system import RAGSystem

            # Set up mocks
            mock_ai = Mock()
            mock_ai.generate_response.return_value = "Test response about ML"
            mock_ai_class.return_value = mock_ai

            mock_sm = Mock()
            mock_sm.get_conversation_history.return_value = None
            mock_sm_class.return_value = mock_sm

            mock_tm = Mock()
            mock_tm.get_tool_definitions.return_value = []
            mock_tm.get_last_sources.return_value = ["Source 1", "Source 2"]
            mock_tm_class.return_value = mock_tm

            system = RAGSystem(mock_config)
            system.ai_generator = mock_ai
            system.session_manager = mock_sm
            system.tool_manager = mock_tm

            yield system, mock_ai, mock_sm, mock_tm

    def test_query_returns_response_and_sources(self, rag_system_with_mocks):
        """Test that query returns tuple of (response, sources)"""
        system, mock_ai, mock_sm, mock_tm = rag_system_with_mocks

        # Act
        response, sources = system.query("What is machine learning?")

        # Assert
        assert response == "Test response about ML"
        assert sources == ["Source 1", "Source 2"]

    def test_query_calls_ai_generator(self, rag_system_with_mocks):
        """Test that query calls AIGenerator.generate_response"""
        system, mock_ai, mock_sm, mock_tm = rag_system_with_mocks

        # Act
        system.query("test question")

        # Assert
        mock_ai.generate_response.assert_called_once()
        call_kwargs = mock_ai.generate_response.call_args[1]
        assert "test question" in call_kwargs["query"]
        assert call_kwargs["tool_manager"] == mock_tm

    def test_query_retrieves_conversation_history_with_session(self, rag_system_with_mocks):
        """Test that conversation history is retrieved when session_id provided"""
        system, mock_ai, mock_sm, mock_tm = rag_system_with_mocks
        mock_sm.get_conversation_history.return_value = "Previous conversation"

        # Act
        system.query("follow-up question", session_id="session_123")

        # Assert
        mock_sm.get_conversation_history.assert_called_with("session_123")
        call_kwargs = mock_ai.generate_response.call_args[1]
        assert call_kwargs["conversation_history"] == "Previous conversation"

    def test_query_no_history_without_session(self, rag_system_with_mocks):
        """Test that no history is retrieved without session_id"""
        system, mock_ai, mock_sm, mock_tm = rag_system_with_mocks

        # Act
        system.query("question without session")

        # Assert
        call_kwargs = mock_ai.generate_response.call_args[1]
        assert call_kwargs["conversation_history"] is None

    def test_query_adds_exchange_to_session(self, rag_system_with_mocks):
        """Test that query/response exchange is saved to session"""
        system, mock_ai, mock_sm, mock_tm = rag_system_with_mocks

        # Act
        system.query("user question", session_id="session_456")

        # Assert
        mock_sm.add_exchange.assert_called_once_with(
            "session_456",
            "user question",
            "Test response about ML"
        )

    def test_query_does_not_save_without_session(self, rag_system_with_mocks):
        """Test that exchange is not saved when no session_id"""
        system, mock_ai, mock_sm, mock_tm = rag_system_with_mocks

        # Act
        system.query("question without session", session_id=None)

        # Assert
        mock_sm.add_exchange.assert_not_called()

    def test_query_gets_sources_from_tool_manager(self, rag_system_with_mocks):
        """Test that sources are retrieved from ToolManager"""
        system, mock_ai, mock_sm, mock_tm = rag_system_with_mocks
        mock_tm.get_last_sources.return_value = ["Course A - Lesson 1"]

        # Act
        response, sources = system.query("test")

        # Assert
        mock_tm.get_last_sources.assert_called_once()
        assert sources == ["Course A - Lesson 1"]

    def test_query_resets_sources_after_retrieval(self, rag_system_with_mocks):
        """Test that sources are reset after being retrieved"""
        system, mock_ai, mock_sm, mock_tm = rag_system_with_mocks

        # Act
        system.query("test")

        # Assert
        mock_tm.reset_sources.assert_called_once()

    def test_query_creates_proper_prompt(self, rag_system_with_mocks):
        """Test that query creates proper prompt for AI"""
        system, mock_ai, mock_sm, mock_tm = rag_system_with_mocks

        # Act
        system.query("What is deep learning?")

        # Assert
        call_kwargs = mock_ai.generate_response.call_args[1]
        assert "Answer this question about course materials" in call_kwargs["query"]
        assert "What is deep learning?" in call_kwargs["query"]

    def test_query_passes_tools_to_ai_generator(self, rag_system_with_mocks):
        """Test that tool definitions are passed to AI generator"""
        system, mock_ai, mock_sm, mock_tm = rag_system_with_mocks
        mock_tm.get_tool_definitions.return_value = [{"name": "search_tool"}]

        # Act
        system.query("test")

        # Assert
        call_kwargs = mock_ai.generate_response.call_args[1]
        assert call_kwargs["tools"] == [{"name": "search_tool"}]


class TestRAGSystemQueryIntegration:
    """Integration-style tests for query flow"""

    def test_full_query_flow_without_tool_use(self, mock_config):
        """Test complete query flow when AI responds directly"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as mock_ai_class, \
             patch('rag_system.SessionManager') as mock_sm_class, \
             patch('rag_system.ToolManager') as mock_tm_class, \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):

            from rag_system import RAGSystem

            # Set up for direct response (no tool use)
            mock_ai = Mock()
            mock_ai.generate_response.return_value = "Direct answer without tools"
            mock_ai_class.return_value = mock_ai

            mock_sm = Mock()
            mock_sm.get_conversation_history.return_value = None
            mock_sm_class.return_value = mock_sm

            mock_tm = Mock()
            mock_tm.get_tool_definitions.return_value = []
            mock_tm.get_last_sources.return_value = []  # No sources when no tool used
            mock_tm_class.return_value = mock_tm

            system = RAGSystem(mock_config)

            # Act
            response, sources = system.query("What is 2+2?")

            # Assert
            assert response == "Direct answer without tools"
            assert sources == []

    def test_full_query_flow_with_tool_use(self, mock_config):
        """Test complete query flow when AI uses search tool"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as mock_ai_class, \
             patch('rag_system.SessionManager') as mock_sm_class, \
             patch('rag_system.ToolManager') as mock_tm_class, \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):

            from rag_system import RAGSystem

            # Set up for tool-assisted response
            mock_ai = Mock()
            mock_ai.generate_response.return_value = "Answer based on course content"
            mock_ai_class.return_value = mock_ai

            mock_sm = Mock()
            mock_sm.get_conversation_history.return_value = None
            mock_sm_class.return_value = mock_sm

            mock_tm = Mock()
            mock_tm.get_tool_definitions.return_value = [{"name": "search_course_content"}]
            mock_tm.get_last_sources.return_value = [
                '<a href="http://example.com">ML Course - Lesson 1</a>'
            ]
            mock_tm_class.return_value = mock_tm

            system = RAGSystem(mock_config)

            # Act
            response, sources = system.query("Explain neural networks")

            # Assert
            assert response == "Answer based on course content"
            assert len(sources) == 1
            assert "ML Course" in sources[0]

    def test_query_with_session_continuity(self, mock_config):
        """Test that session maintains continuity across queries"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as mock_ai_class, \
             patch('rag_system.SessionManager') as mock_sm_class, \
             patch('rag_system.ToolManager') as mock_tm_class, \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):

            from rag_system import RAGSystem

            mock_ai = Mock()
            mock_ai.generate_response.return_value = "Response with context"
            mock_ai_class.return_value = mock_ai

            mock_sm = Mock()
            # Simulate that second query has history from first
            mock_sm.get_conversation_history.side_effect = [
                None,  # First query - no history
                "User: First question\nAssistant: First response"  # Second query has history
            ]
            mock_sm_class.return_value = mock_sm

            mock_tm = Mock()
            mock_tm.get_tool_definitions.return_value = []
            mock_tm.get_last_sources.return_value = []
            mock_tm_class.return_value = mock_tm

            system = RAGSystem(mock_config)

            # Act - First query
            system.query("First question", session_id="session_1")

            # Act - Second query with same session
            system.query("Follow-up question", session_id="session_1")

            # Assert - Second call should have history
            second_call_kwargs = mock_ai.generate_response.call_args_list[1][1]
            assert second_call_kwargs["conversation_history"] == "User: First question\nAssistant: First response"
