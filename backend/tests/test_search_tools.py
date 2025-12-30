"""
Tests for CourseSearchTool and ToolManager in search_tools.py

Key test areas:
1. CourseSearchTool.execute() - search returns results
2. CourseSearchTool.execute() - empty results handling
3. CourseSearchTool.execute() - error cases
4. CourseSearchTool.execute() - filtering by course/lesson
5. CourseSearchTool._format_results() - output formatting
6. ToolManager - tool registration and execution
"""

import pytest
from unittest.mock import Mock
import sys
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute() method"""

    def test_execute_returns_formatted_results(self, mock_vector_store, sample_search_results):
        """Test that execute returns properly formatted results when search succeeds"""
        # Arrange
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="machine learning")

        # Assert
        mock_vector_store.search.assert_called_once_with(
            query="machine learning",
            course_name=None,
            lesson_number=None
        )
        assert "[ML Fundamentals" in result
        assert "machine learning basics" in result
        assert tool.last_sources  # Sources should be populated

    def test_execute_with_empty_results(self, mock_vector_store, empty_search_results):
        """Test that execute returns appropriate message when no results found"""
        # Arrange
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="nonexistent topic")

        # Assert
        assert "No relevant content found" in result

    def test_execute_with_empty_results_and_course_filter(self, mock_vector_store, empty_search_results):
        """Test empty results message includes course filter info"""
        # Arrange
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="topic", course_name="Python Basics")

        # Assert
        assert "No relevant content found" in result
        assert "in course 'Python Basics'" in result

    def test_execute_with_empty_results_and_lesson_filter(self, mock_vector_store, empty_search_results):
        """Test empty results message includes lesson filter info"""
        # Arrange
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="topic", lesson_number=3)

        # Assert
        assert "No relevant content found" in result
        assert "in lesson 3" in result

    def test_execute_with_error_result(self, mock_vector_store, error_search_results):
        """Test that execute returns error message when search fails"""
        # Arrange
        mock_vector_store.search.return_value = error_search_results
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="some query")

        # Assert
        assert "No course found matching" in result

    def test_execute_with_course_filter(self, mock_vector_store, sample_search_results):
        """Test that course_name filter is passed to vector store"""
        # Arrange
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)

        # Act
        tool.execute(query="neural networks", course_name="ML Fundamentals")

        # Assert
        mock_vector_store.search.assert_called_once_with(
            query="neural networks",
            course_name="ML Fundamentals",
            lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store, sample_search_results):
        """Test that lesson_number filter is passed to vector store"""
        # Arrange
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)

        # Act
        tool.execute(query="introduction", lesson_number=1)

        # Assert
        mock_vector_store.search.assert_called_once_with(
            query="introduction",
            course_name=None,
            lesson_number=1
        )

    def test_execute_with_both_filters(self, mock_vector_store, sample_search_results):
        """Test that both filters are passed correctly"""
        # Arrange
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)

        # Act
        tool.execute(query="basics", course_name="ML Course", lesson_number=2)

        # Assert
        mock_vector_store.search.assert_called_once_with(
            query="basics",
            course_name="ML Course",
            lesson_number=2
        )


class TestCourseSearchToolFormatResults:
    """Tests for CourseSearchTool._format_results() method"""

    def test_format_results_includes_course_title(self, mock_vector_store, sample_search_results):
        """Test that formatted results include course title in header"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool._format_results(sample_search_results)

        # Assert
        assert "[ML Fundamentals" in result

    def test_format_results_includes_lesson_number(self, mock_vector_store, sample_search_results):
        """Test that formatted results include lesson number"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool._format_results(sample_search_results)

        # Assert
        assert "Lesson 1" in result
        assert "Lesson 2" in result

    def test_format_results_includes_document_content(self, mock_vector_store, sample_search_results):
        """Test that formatted results include actual document content"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool._format_results(sample_search_results)

        # Assert
        assert "machine learning basics" in result
        assert "Deep learning uses neural networks" in result

    def test_format_results_populates_last_sources(self, mock_vector_store, sample_search_results):
        """Test that formatting populates last_sources for UI"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)

        # Act
        tool._format_results(sample_search_results)

        # Assert
        assert len(tool.last_sources) == 2
        assert "ML Fundamentals - Lesson 1" in tool.last_sources[0]

    def test_format_results_with_lesson_link(self, mock_vector_store, sample_search_results):
        """Test that lesson links are included when available"""
        # Arrange
        mock_vector_store.get_lesson_link.return_value = "https://example.com/ml/lesson1"
        tool = CourseSearchTool(mock_vector_store)

        # Act
        tool._format_results(sample_search_results)

        # Assert
        # Check that sources contain clickable links
        assert any("href=" in source for source in tool.last_sources)

    def test_format_results_without_lesson_link(self, mock_vector_store, sample_search_results):
        """Test handling when lesson link is not available"""
        # Arrange
        mock_vector_store.get_lesson_link.return_value = None
        tool = CourseSearchTool(mock_vector_store)

        # Act
        tool._format_results(sample_search_results)

        # Assert
        # Sources should still be populated but without links
        assert len(tool.last_sources) == 2
        assert "href=" not in tool.last_sources[0]


class TestToolManager:
    """Tests for ToolManager class"""

    def test_register_tool(self, mock_vector_store):
        """Test that tools can be registered"""
        # Arrange
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        # Act
        manager.register_tool(tool)

        # Assert
        assert "search_course_content" in manager.tools

    def test_get_tool_definitions(self, mock_vector_store):
        """Test that tool definitions are returned correctly"""
        # Arrange
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Act
        definitions = manager.get_tool_definitions()

        # Assert
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"
        assert "input_schema" in definitions[0]

    def test_execute_tool_success(self, mock_vector_store, sample_search_results):
        """Test successful tool execution through manager"""
        # Arrange
        mock_vector_store.search.return_value = sample_search_results
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Act
        result = manager.execute_tool("search_course_content", query="test")

        # Assert
        assert "[ML Fundamentals" in result

    def test_execute_tool_not_found(self):
        """Test error handling when tool not found"""
        # Arrange
        manager = ToolManager()

        # Act
        result = manager.execute_tool("nonexistent_tool", query="test")

        # Assert
        assert "not found" in result

    def test_get_last_sources(self, mock_vector_store, sample_search_results):
        """Test retrieving sources from last search"""
        # Arrange
        mock_vector_store.search.return_value = sample_search_results
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="test")

        # Act
        sources = manager.get_last_sources()

        # Assert
        assert len(sources) > 0

    def test_reset_sources(self, mock_vector_store, sample_search_results):
        """Test resetting sources after retrieval"""
        # Arrange
        mock_vector_store.search.return_value = sample_search_results
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="test")

        # Act
        manager.reset_sources()
        sources = manager.get_last_sources()

        # Assert
        assert sources == []


class TestCourseOutlineTool:
    """Tests for CourseOutlineTool class"""

    def test_execute_returns_course_outline(self, mock_vector_store):
        """Test that execute returns formatted course outline"""
        # Arrange
        tool = CourseOutlineTool(mock_vector_store)

        # Act
        result = tool.execute(course_name="ML Fundamentals")

        # Assert
        assert "Course:" in result
        assert "ML Fundamentals" in result
        assert "Lessons" in result

    def test_execute_course_not_found(self, mock_vector_store):
        """Test error handling when course not found"""
        # Arrange
        mock_vector_store.course_catalog.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        tool = CourseOutlineTool(mock_vector_store)

        # Act
        result = tool.execute(course_name="NonExistent")

        # Assert
        assert "No course found" in result

    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is correct"""
        # Arrange
        tool = CourseOutlineTool(mock_vector_store)

        # Act
        definition = tool.get_tool_definition()

        # Assert
        assert definition["name"] == "get_course_outline"
        assert "course_name" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["required"]
