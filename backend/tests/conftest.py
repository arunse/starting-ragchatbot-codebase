"""
Pytest configuration and shared fixtures for RAG system tests
"""
import pytest
import tempfile
import os
import shutil
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any
import json

# Add the backend directory to Python path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = Config()
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = tempfile.mkdtemp()  # Use temp directory for tests
    return config


@pytest.fixture
def sample_courses():
    """Sample course data for testing"""
    return [
        Course(
            title="Introduction to MCP",
            instructor="Claude AI",
            course_link="https://example.com/mcp-intro",
            lessons=[
                Lesson(lesson_number=1, title="What is MCP?", lesson_link="https://example.com/lesson1"),
                Lesson(lesson_number=2, title="Setting up MCP", lesson_link="https://example.com/lesson2")
            ]
        ),
        Course(
            title="Advanced Web Development",
            instructor="John Doe",
            course_link="https://example.com/web-dev",
            lessons=[
                Lesson(lesson_number=1, title="JavaScript Fundamentals", lesson_link="https://example.com/js-fund"),
                Lesson(lesson_number=2, title="React Components", lesson_link="https://example.com/react")
            ]
        )
    ]


@pytest.fixture
def sample_course_chunks(sample_courses):
    """Sample course chunks for testing"""
    chunks = []
    for course in sample_courses:
        for i, lesson in enumerate(course.lessons):
            chunk = CourseChunk(
                course_title=course.title,
                lesson_number=lesson.lesson_number,
                content=f"This is content for {lesson.title}. It contains educational material about the topic.",
                chunk_index=i
            )
            chunks.append(chunk)
    return chunks


@pytest.fixture
def mock_search_results():
    """Mock search results for testing"""
    return SearchResults(
        documents=["This is sample course content about MCP.", "Another piece of content about web development."],
        metadata=[
            {"course_title": "Introduction to MCP", "lesson_number": 1},
            {"course_title": "Advanced Web Development", "lesson_number": 2}
        ],
        distances=[0.1, 0.2]
    )


@pytest.fixture
def empty_search_results():
    """Empty search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def error_search_results():
    """Error search results for testing"""
    return SearchResults.empty("Database connection failed")


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response"""
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "This is a sample response from Claude"
    mock_response.stop_reason = "end_turn"
    return mock_response


@pytest.fixture
def mock_anthropic_tool_response():
    """Mock Anthropic API response with tool use"""
    mock_response = Mock()
    
    # Mock text content
    mock_text_block = Mock()
    mock_text_block.type = "text"
    mock_text_block.text = "Let me search for that information."
    
    # Mock tool use content
    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.id = "tool_123"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.input = {"query": "test query"}
    
    mock_response.content = [mock_text_block, mock_tool_block]
    mock_response.stop_reason = "tool_use"
    return mock_response


@pytest.fixture
def mock_vector_store(mock_search_results):
    """Mock vector store for testing"""
    mock_store = Mock()
    mock_store.search.return_value = mock_search_results
    mock_store.get_lesson_link.return_value = "https://example.com/lesson1"
    mock_store.get_course_count.return_value = 2
    mock_store.get_existing_course_titles.return_value = ["Introduction to MCP", "Advanced Web Development"]
    return mock_store


@pytest.fixture
def failing_vector_store(error_search_results):
    """Mock vector store that fails for testing error handling"""
    mock_store = Mock()
    mock_store.search.return_value = error_search_results
    return mock_store


@pytest.fixture
def mock_tool_manager():
    """Mock tool manager for testing"""
    mock_manager = Mock()
    mock_manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course content",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }
        }
    ]
    mock_manager.execute_tool.return_value = "Mock search results"
    mock_manager.get_last_sources.return_value = []
    return mock_manager


@pytest.fixture
def temp_chroma_path():
    """Temporary directory for ChromaDB testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_sentence_transformer():
    """Mock sentence transformer for testing"""
    with patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction') as mock:
        yield mock


@pytest.fixture
def sample_course_document_text():
    """Sample course document text for testing document processing"""
    return """Course Title: Test Course
Course Link: https://example.com/test-course
Course Instructor: Test Instructor

Lesson 1: Introduction to Testing
Lesson Link: https://example.com/lesson1

This is the content for lesson 1. It covers the basics of testing and provides examples of how to write good tests.

Lesson 2: Advanced Testing Techniques  
Lesson Link: https://example.com/lesson2

This is the content for lesson 2. It covers more advanced testing concepts like mocking and integration testing.
"""


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test"""
    yield
    # Any cleanup code would go here


# Pytest configuration
def pytest_configure(config):
    """Configure pytest settings"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )