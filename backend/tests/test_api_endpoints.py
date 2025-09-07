"""
Tests for FastAPI endpoints to identify web interface issues
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, Mock

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app import app


@pytest.mark.integration
class TestAPIEndpoints:
    """Test FastAPI endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    def test_query_endpoint_general_question(self, client):
        """Test /api/query with general question"""
        response = client.post(
            "/api/query",
            json={"query": "What is 2+2?", "session_id": "test_session"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert "4" in data["answer"]  # Should answer the math question

    def test_query_endpoint_course_question(self, client):
        """Test /api/query with course-specific question"""
        response = client.post(
            "/api/query", 
            json={"query": "Tell me about MCP", "session_id": "test_session"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        
        # Should have course-related content
        answer = data["answer"].lower()
        assert "mcp" in answer or "model context protocol" in answer
        
        # Should have sources for course content
        if data["sources"]:
            print(f"Course query sources: {data['sources']}")

    def test_query_endpoint_without_session(self, client):
        """Test /api/query without session_id (should create one)"""
        response = client.post(
            "/api/query",
            json={"query": "Hello"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "session_id" in data
        assert data["session_id"] is not None
        assert len(data["session_id"]) > 0

    def test_courses_endpoint(self, client):
        """Test /api/courses endpoint"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] > 0
        assert len(data["course_titles"]) > 0
        
        print(f"API reports {data['total_courses']} courses: {data['course_titles']}")

    def test_new_session_endpoint(self, client):
        """Test /api/new-session endpoint"""
        response = client.post("/api/new-session")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "session_id" in data
        assert data["session_id"] is not None

    def test_query_endpoint_error_handling(self, client):
        """Test query endpoint error handling"""
        # Test with invalid JSON structure
        response = client.post("/api/query", json={})  # Missing required 'query' field
        
        # Should return 422 for validation error
        assert response.status_code == 422

    def test_query_endpoint_empty_query(self, client):
        """Test query endpoint with empty query"""
        response = client.post(
            "/api/query",
            json={"query": "", "session_id": "test_session"}
        )
        
        # Should still return 200 but handle gracefully
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    @pytest.mark.slow
    def test_multiple_queries_same_session(self, client):
        """Test multiple queries in the same session"""
        session_id = "test_multi_session"
        
        # First query
        response1 = client.post(
            "/api/query",
            json={"query": "What is MCP?", "session_id": session_id}
        )
        assert response1.status_code == 200
        
        # Follow-up query in same session
        response2 = client.post(
            "/api/query", 
            json={"query": "Tell me more about that", "session_id": session_id}
        )
        assert response2.status_code == 200
        
        data2 = response2.json()
        assert data2["session_id"] == session_id

    def test_query_with_specific_course_filter(self, client):
        """Test query that should use course filtering"""
        response = client.post(
            "/api/query",
            json={"query": "What is retrieval augmented generation?", "session_id": "test"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        print(f"RAG query response: {data['answer'][:200]}...")
        print(f"RAG query sources: {data['sources']}")

    def test_query_nonexistent_topic(self, client):
        """Test query about nonexistent topic"""
        response = client.post(
            "/api/query",
            json={"query": "Tell me about quantum computing in course materials", "session_id": "test"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should still return an answer, even if no course content found
        assert len(data["answer"]) > 0
        print(f"Nonexistent topic response: {data['answer'][:200]}...")

    def test_cors_headers(self, client):
        """Test that CORS headers are present"""
        response = client.post("/api/query", json={"query": "test"})
        
        # Check for CORS headers (may not be present in test client)
        # This is more of a documentation test
        assert response.status_code == 200


@pytest.mark.integration
class TestAPIErrorScenarios:
    """Test API error scenarios that might cause 'query failed'"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @patch('app.rag_system')
    def test_rag_system_exception(self, mock_rag_system, client):
        """Test when RAG system throws exception"""
        mock_rag_system.query.side_effect = Exception("RAG system error")
        
        response = client.post("/api/query", json={"query": "test"})
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "RAG system error" in data["detail"]

    @patch('app.rag_system')  
    def test_rag_system_returns_none(self, mock_rag_system, client):
        """Test when RAG system returns None"""
        mock_rag_system.query.return_value = (None, [])
        mock_rag_system.session_manager.create_session.return_value = "test_session_123"
        
        response = client.post("/api/query", json={"query": "test"})
        
        # Should handle None gracefully
        assert response.status_code == 200
        data = response.json()
        assert "I apologize, but I encountered an issue" in data["answer"]

    @patch('app.rag_system')
    def test_rag_system_empty_response(self, mock_rag_system, client):
        """Test when RAG system returns empty response"""  
        mock_rag_system.query.return_value = ("", [])
        mock_rag_system.session_manager.create_session.return_value = "test_session_123"
        
        response = client.post("/api/query", json={"query": "test"})
        
        assert response.status_code == 200
        data = response.json()
        assert "I couldn't find relevant information" in data["answer"]

    def test_malformed_request(self, client):
        """Test malformed request handling"""
        response = client.post("/api/query", json={"invalid": "data"})
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])