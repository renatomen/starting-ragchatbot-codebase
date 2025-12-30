"""
API endpoint tests for the Course Materials RAG System.

Tests cover:
- POST /api/query - Query processing with session management
- GET /api/courses - Course statistics retrieval
- GET / - Root endpoint
- Error handling for various failure scenarios
"""

import pytest
from unittest.mock import Mock


class TestQueryEndpoint:
    """Tests for POST /api/query endpoint"""

    def test_query_with_new_session(self, test_client, mock_rag_system):
        """Test query creates new session when none provided"""
        response = test_client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_with_existing_session(self, test_client, mock_rag_system):
        """Test query uses provided session ID"""
        response = test_client.post(
            "/api/query",
            json={"query": "Tell me more", "session_id": "existing-session-456"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing-session-456"
        mock_rag_system.session_manager.create_session.assert_not_called()

    def test_query_returns_sources(self, test_client, mock_rag_system):
        """Test query response includes sources from RAG system"""
        response = test_client.post(
            "/api/query",
            json={"query": "What courses cover neural networks?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 2
        assert "ML Fundamentals - Lesson 1" in data["sources"]

    def test_query_calls_rag_system(self, test_client, mock_rag_system):
        """Test query endpoint calls RAGSystem.query with correct arguments"""
        test_query = "How do I implement gradient descent?"
        test_session = "my-session-789"

        response = test_client.post(
            "/api/query",
            json={"query": test_query, "session_id": test_session}
        )

        assert response.status_code == 200
        mock_rag_system.query.assert_called_once_with(test_query, test_session)

    def test_query_missing_query_field(self, test_client):
        """Test query endpoint returns 422 when query field is missing"""
        response = test_client.post(
            "/api/query",
            json={"session_id": "some-session"}
        )

        assert response.status_code == 422

    def test_query_empty_query(self, test_client, mock_rag_system):
        """Test query endpoint handles empty query string"""
        response = test_client.post(
            "/api/query",
            json={"query": ""}
        )

        # Empty string is valid per the schema, behavior depends on RAG system
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once()

    def test_query_rag_system_error(self, test_client, mock_rag_system):
        """Test query endpoint returns 500 when RAGSystem raises exception"""
        mock_rag_system.query.side_effect = Exception("Database connection failed")

        response = test_client.post(
            "/api/query",
            json={"query": "What is AI?"}
        )

        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]


class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint"""

    def test_get_courses_success(self, test_client, mock_rag_system):
        """Test courses endpoint returns course statistics"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "ML Fundamentals" in data["course_titles"]

    def test_get_courses_calls_analytics(self, test_client, mock_rag_system):
        """Test courses endpoint calls get_course_analytics"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        mock_rag_system.get_course_analytics.assert_called_once()

    def test_get_courses_empty_catalog(self, test_client, mock_rag_system):
        """Test courses endpoint handles empty course catalog"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_get_courses_error(self, test_client, mock_rag_system):
        """Test courses endpoint returns 500 on analytics error"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Vector store unavailable")

        response = test_client.get("/api/courses")

        assert response.status_code == 500
        assert "Vector store unavailable" in response.json()["detail"]


class TestRootEndpoint:
    """Tests for GET / endpoint"""

    def test_root_returns_message(self, test_client):
        """Test root endpoint returns API info message"""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "RAG System" in data["message"]


class TestRequestValidation:
    """Tests for request validation and edge cases"""

    def test_query_invalid_json(self, test_client):
        """Test query endpoint rejects invalid JSON"""
        response = test_client.post(
            "/api/query",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_query_wrong_content_type(self, test_client):
        """Test query endpoint requires JSON content type"""
        response = test_client.post(
            "/api/query",
            data="query=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        assert response.status_code == 422

    def test_query_extra_fields_ignored(self, test_client, mock_rag_system):
        """Test query endpoint ignores extra fields in request"""
        response = test_client.post(
            "/api/query",
            json={
                "query": "Test query",
                "extra_field": "should be ignored",
                "another_field": 123
            }
        )

        assert response.status_code == 200
        mock_rag_system.query.assert_called_once()

    def test_query_null_session_id(self, test_client, mock_rag_system):
        """Test query endpoint handles null session_id"""
        response = test_client.post(
            "/api/query",
            json={"query": "Test", "session_id": None}
        )

        assert response.status_code == 200
        # Should create new session when session_id is None
        mock_rag_system.session_manager.create_session.assert_called_once()


class TestResponseFormat:
    """Tests for response format compliance"""

    def test_query_response_schema(self, test_client):
        """Test query response matches expected schema"""
        response = test_client.post(
            "/api/query",
            json={"query": "Test"}
        )

        data = response.json()
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

    def test_courses_response_schema(self, test_client):
        """Test courses response matches expected schema"""
        response = test_client.get("/api/courses")

        data = response.json()
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert all(isinstance(title, str) for title in data["course_titles"])
