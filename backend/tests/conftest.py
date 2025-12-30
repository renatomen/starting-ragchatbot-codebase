"""
Shared fixtures for RAG chatbot backend tests.

Provides mock objects for:
- VectorStore and SearchResults
- Anthropic API responses
- ToolManager
- Config and SessionManager
"""

import pytest
from unittest.mock import Mock
import sys
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from vector_store import SearchResults


# ============== SearchResults Fixtures ==============

@pytest.fixture
def sample_search_results():
    """Create sample SearchResults with course content"""
    return SearchResults(
        documents=[
            "This is content about machine learning basics.",
            "Deep learning uses neural networks for complex patterns."
        ],
        metadata=[
            {"course_title": "ML Fundamentals", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "ML Fundamentals", "lesson_number": 2, "chunk_index": 5}
        ],
        distances=[0.25, 0.35],
        error=None
    )


@pytest.fixture
def empty_search_results():
    """Create empty SearchResults"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None
    )


@pytest.fixture
def error_search_results():
    """Create SearchResults with an error"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="No course found matching 'NonExistent Course'"
    )


# ============== VectorStore Mock ==============

@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore"""
    mock = Mock()

    # Default: return empty results
    mock.search.return_value = SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None
    )

    # Mock get_lesson_link
    mock.get_lesson_link.return_value = "https://example.com/lesson/1"

    # Mock course_catalog for CourseOutlineTool
    mock.course_catalog = Mock()
    mock.course_catalog.query.return_value = {
        'documents': [['ML Fundamentals']],
        'metadatas': [[{
            'title': 'ML Fundamentals',
            'course_link': 'https://example.com/ml',
            'lessons_json': '[{"lesson_number": 1, "lesson_title": "Introduction"}]'
        }]],
        'distances': [[0.1]]
    }

    return mock


# ============== Anthropic API Mock Fixtures ==============

@pytest.fixture
def mock_text_response():
    """Mock a simple text response from Claude (no tool use)"""
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"

    text_block = Mock()
    text_block.type = "text"
    text_block.text = "This is a direct answer about course materials."

    mock_response.content = [text_block]
    return mock_response


@pytest.fixture
def mock_tool_use_response():
    """Mock a response where Claude wants to use a tool"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"

    # Tool use block
    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.id = "tool_use_123"
    tool_block.name = "search_course_content"
    tool_block.input = {"query": "machine learning basics"}

    mock_response.content = [tool_block]
    return mock_response


@pytest.fixture
def mock_final_response_after_tool():
    """Mock Claude's final response after receiving tool results"""
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"

    text_block = Mock()
    text_block.type = "text"
    text_block.text = "Based on the course materials, machine learning is a field of AI."

    mock_response.content = [text_block]
    return mock_response


@pytest.fixture
def mock_second_tool_use_response():
    """Mock a second round tool_use response (e.g., after first search)"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"

    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.id = "tool_use_456"
    tool_block.name = "search_course_content"
    tool_block.input = {"query": "neural networks"}

    mock_response.content = [tool_block]
    return mock_response


# ============== ToolManager Mock ==============

@pytest.fixture
def mock_tool_manager():
    """Create a mock ToolManager"""
    mock = Mock()
    mock.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    ]
    mock.execute_tool.return_value = "[ML Fundamentals - Lesson 1]\nMachine learning basics content..."
    mock.get_last_sources.return_value = ["ML Fundamentals - Lesson 1"]
    mock.reset_sources.return_value = None
    return mock


# ============== Config Mock ==============

@pytest.fixture
def mock_config():
    """Create a mock Config object"""
    mock = Mock()
    mock.CHUNK_SIZE = 800
    mock.CHUNK_OVERLAP = 100
    mock.CHROMA_PATH = "./test_chroma_db"
    mock.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    mock.MAX_RESULTS = 5
    mock.MAX_HISTORY = 2
    mock.ANTHROPIC_API_KEY = "test-api-key"
    mock.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    return mock


# ============== SessionManager Mock ==============

@pytest.fixture
def mock_session_manager():
    """Create a mock SessionManager"""
    mock = Mock()
    mock.get_conversation_history.return_value = None
    mock.add_exchange.return_value = None
    return mock
