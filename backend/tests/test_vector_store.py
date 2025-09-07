"""
Unit and integration tests for VectorStore
"""
import pytest
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


@pytest.mark.unit
class TestSearchResults:
    """Test SearchResults data class"""

    def test_from_chroma(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'course': 'test'}, {'course': 'test2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'course': 'test'}, {'course': 'test2'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None

    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [],
            'metadatas': [],
            'distances': []
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.is_empty()

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        results = SearchResults.empty("Connection failed")
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "Connection failed"
        assert results.is_empty()

    def test_is_empty(self):
        """Test is_empty method"""
        empty_results = SearchResults([], [], [])
        non_empty_results = SearchResults(['doc'], [{}], [0.1])
        
        assert empty_results.is_empty()
        assert not non_empty_results.is_empty()


@pytest.mark.unit
class TestVectorStore:
    """Test VectorStore functionality with mocks"""

    @pytest.fixture
    def mock_chroma_client(self):
        """Mock ChromaDB client"""
        with patch('vector_store.chromadb.PersistentClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # Mock collections
            mock_catalog = Mock()
            mock_content = Mock()
            mock_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]
            
            yield mock_instance, mock_catalog, mock_content

    @pytest.fixture
    def mock_embedding_function(self):
        """Mock embedding function"""
        with patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction') as mock_embed:
            yield mock_embed

    def test_initialization(self, mock_chroma_client, mock_embedding_function, temp_chroma_path):
        """Test VectorStore initialization"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client
        
        vector_store = VectorStore(temp_chroma_path, "test-model", max_results=3)
        
        # Verify client creation
        assert mock_instance.get_or_create_collection.call_count == 2
        assert vector_store.max_results == 3
        assert vector_store.course_catalog == mock_catalog
        assert vector_store.course_content == mock_content

    def test_search_without_filters(self, mock_chroma_client, mock_embedding_function, temp_chroma_path):
        """Test search without course name or lesson filters"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client
        
        # Setup mock query results
        mock_results = {
            'documents': [['result doc']],
            'metadatas': [[{'course_title': 'Test Course'}]],
            'distances': [[0.1]]
        }
        mock_content.query.return_value = mock_results
        
        vector_store = VectorStore(temp_chroma_path, "test-model")
        results = vector_store.search("test query")
        
        # Verify query was called correctly
        mock_content.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,  # default max_results
            where=None
        )
        
        assert len(results.documents) == 1
        assert results.documents[0] == 'result doc'

    def test_search_with_course_name(self, mock_chroma_client, mock_embedding_function, temp_chroma_path):
        """Test search with course name filter"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client
        
        # Mock course resolution
        catalog_results = {
            'documents': [['Test Course']],
            'metadatas': [[{'title': 'Test Course'}]]
        }
        mock_catalog.query.return_value = catalog_results
        
        # Mock content search
        content_results = {
            'documents': [['filtered content']],
            'metadatas': [[{'course_title': 'Test Course'}]],
            'distances': [[0.2]]
        }
        mock_content.query.return_value = content_results
        
        vector_store = VectorStore(temp_chroma_path, "test-model")
        results = vector_store.search("query", course_name="Test")
        
        # Verify course resolution was called
        mock_catalog.query.assert_called_once_with(
            query_texts=["Test"],
            n_results=1
        )
        
        # Verify content search with filter
        mock_content.query.assert_called_once_with(
            query_texts=["query"],
            n_results=5,
            where={"course_title": "Test Course"}
        )

    def test_search_with_lesson_number(self, mock_chroma_client, mock_embedding_function, temp_chroma_path):
        """Test search with lesson number filter"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client
        
        content_results = {
            'documents': [['lesson content']],
            'metadatas': [[{'lesson_number': 2}]],
            'distances': [[0.3]]
        }
        mock_content.query.return_value = content_results
        
        vector_store = VectorStore(temp_chroma_path, "test-model")
        results = vector_store.search("query", lesson_number=2)
        
        mock_content.query.assert_called_once_with(
            query_texts=["query"],
            n_results=5,
            where={"lesson_number": 2}
        )

    def test_search_with_both_filters(self, mock_chroma_client, mock_embedding_function, temp_chroma_path):
        """Test search with both course name and lesson number filters"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client
        
        # Mock course resolution
        catalog_results = {
            'documents': [['MCP Course']],
            'metadatas': [[{'title': 'MCP Course'}]]
        }
        mock_catalog.query.return_value = catalog_results
        
        content_results = {
            'documents': [['specific content']],
            'metadatas': [[{'course_title': 'MCP Course', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }
        mock_content.query.return_value = content_results
        
        vector_store = VectorStore(temp_chroma_path, "test-model")
        results = vector_store.search("query", course_name="MCP", lesson_number=1)
        
        expected_filter = {"$and": [
            {"course_title": "MCP Course"},
            {"lesson_number": 1}
        ]}
        
        mock_content.query.assert_called_once_with(
            query_texts=["query"],
            n_results=5,
            where=expected_filter
        )

    def test_search_course_not_found(self, mock_chroma_client, mock_embedding_function, temp_chroma_path):
        """Test search when course name cannot be resolved"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client
        
        # Mock failed course resolution
        mock_catalog.query.return_value = {'documents': [[]], 'metadatas': [[]]}
        
        vector_store = VectorStore(temp_chroma_path, "test-model")
        results = vector_store.search("query", course_name="NonexistentCourse")
        
        assert results.error == "No course found matching 'NonexistentCourse'"
        assert results.is_empty()

    def test_search_error_handling(self, mock_chroma_client, mock_embedding_function, temp_chroma_path):
        """Test search error handling"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client
        
        # Mock ChromaDB error
        mock_content.query.side_effect = Exception("Database error")
        
        vector_store = VectorStore(temp_chroma_path, "test-model")
        results = vector_store.search("query")
        
        assert "Search error: Database error" in results.error
        assert results.is_empty()

    def test_resolve_course_name_success(self, mock_chroma_client, mock_embedding_function, temp_chroma_path):
        """Test successful course name resolution"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client
        
        catalog_results = {
            'documents': [['MCP Course']],
            'metadatas': [[{'title': 'MCP Course'}]]
        }
        mock_catalog.query.return_value = catalog_results
        
        vector_store = VectorStore(temp_chroma_path, "test-model")
        resolved = vector_store._resolve_course_name("MCP")
        
        assert resolved == "MCP Course"

    def test_resolve_course_name_failure(self, mock_chroma_client, mock_embedding_function, temp_chroma_path):
        """Test failed course name resolution"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client
        
        # Mock empty results
        catalog_results = {'documents': [[]], 'metadatas': [[]]}
        mock_catalog.query.return_value = catalog_results
        
        vector_store = VectorStore(temp_chroma_path, "test-model")
        resolved = vector_store._resolve_course_name("NonexistentCourse")
        
        assert resolved is None

    def test_build_filter_combinations(self, mock_chroma_client, mock_embedding_function, temp_chroma_path):
        """Test different filter combinations"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client
        vector_store = VectorStore(temp_chroma_path, "test-model")
        
        # No filters
        assert vector_store._build_filter(None, None) is None
        
        # Course only
        course_filter = vector_store._build_filter("Test Course", None)
        assert course_filter == {"course_title": "Test Course"}
        
        # Lesson only
        lesson_filter = vector_store._build_filter(None, 1)
        assert lesson_filter == {"lesson_number": 1}
        
        # Both filters
        both_filter = vector_store._build_filter("Test Course", 1)
        expected = {"$and": [
            {"course_title": "Test Course"},
            {"lesson_number": 1}
        ]}
        assert both_filter == expected

    def test_add_course_metadata(self, mock_chroma_client, mock_embedding_function, temp_chroma_path, sample_courses):
        """Test adding course metadata"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client
        
        vector_store = VectorStore(temp_chroma_path, "test-model")
        course = sample_courses[0]  # Introduction to MCP
        
        vector_store.add_course_metadata(course)
        
        # Verify add was called with correct parameters
        mock_catalog.add.assert_called_once()
        call_args = mock_catalog.add.call_args[1]
        
        assert call_args["documents"] == ["Introduction to MCP"]
        assert call_args["ids"] == ["Introduction to MCP"]
        
        metadata = call_args["metadatas"][0]
        assert metadata["title"] == "Introduction to MCP"
        assert metadata["instructor"] == "Claude AI"
        assert "lessons_json" in metadata
        assert metadata["lesson_count"] == 2

    def test_add_course_content(self, mock_chroma_client, mock_embedding_function, temp_chroma_path, sample_course_chunks):
        """Test adding course content chunks"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client
        
        vector_store = VectorStore(temp_chroma_path, "test-model")
        
        vector_store.add_course_content(sample_course_chunks)
        
        # Verify add was called with correct parameters
        mock_content.add.assert_called_once()
        call_args = mock_content.add.call_args[1]
        
        assert len(call_args["documents"]) == len(sample_course_chunks)
        assert len(call_args["metadatas"]) == len(sample_course_chunks)
        assert len(call_args["ids"]) == len(sample_course_chunks)
        
        # Check first chunk
        first_metadata = call_args["metadatas"][0]
        assert first_metadata["course_title"] == "Introduction to MCP"
        assert first_metadata["lesson_number"] == 1
        assert first_metadata["chunk_index"] == 0

    def test_get_existing_course_titles(self, mock_chroma_client, mock_embedding_function, temp_chroma_path):
        """Test getting existing course titles"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client
        
        mock_catalog.get.return_value = {
            'ids': ['Course 1', 'Course 2', 'Course 3']
        }
        
        vector_store = VectorStore(temp_chroma_path, "test-model")
        titles = vector_store.get_existing_course_titles()
        
        assert titles == ['Course 1', 'Course 2', 'Course 3']

    def test_get_course_count(self, mock_chroma_client, mock_embedding_function, temp_chroma_path):
        """Test getting course count"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client
        
        mock_catalog.get.return_value = {
            'ids': ['Course 1', 'Course 2']
        }
        
        vector_store = VectorStore(temp_chroma_path, "test-model")
        count = vector_store.get_course_count()
        
        assert count == 2

    def test_get_lesson_link(self, mock_chroma_client, mock_embedding_function, temp_chroma_path):
        """Test getting lesson link"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client
        
        lessons_json = '[{"lesson_number": 1, "lesson_link": "https://example.com/lesson1"}]'
        mock_catalog.get.return_value = {
            'metadatas': [{'lessons_json': lessons_json}]
        }
        
        vector_store = VectorStore(temp_chroma_path, "test-model")
        link = vector_store.get_lesson_link("Test Course", 1)
        
        assert link == "https://example.com/lesson1"

    def test_clear_all_data(self, mock_chroma_client, mock_embedding_function, temp_chroma_path):
        """Test clearing all data"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client
        
        vector_store = VectorStore(temp_chroma_path, "test-model")
        vector_store.clear_all_data()
        
        # Verify collections were deleted
        assert mock_instance.delete_collection.call_count == 2
        mock_instance.delete_collection.assert_any_call("course_catalog")
        mock_instance.delete_collection.assert_any_call("course_content")


@pytest.mark.integration
class TestVectorStoreIntegration:
    """Integration tests with real ChromaDB"""

    def test_real_chromadb_operations(self, temp_chroma_path, sample_courses, sample_course_chunks):
        """Test actual ChromaDB operations"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2")
        
        # Add course metadata
        course = sample_courses[0]
        vector_store.add_course_metadata(course)
        
        # Add course content
        chunks = sample_course_chunks[:2]  # Just first 2 chunks
        vector_store.add_course_content(chunks)
        
        # Test course count
        assert vector_store.get_course_count() == 1
        
        # Test course titles
        titles = vector_store.get_existing_course_titles()
        assert "Introduction to MCP" in titles
        
        # Test search
        results = vector_store.search("MCP")
        assert not results.is_empty()
        assert len(results.documents) > 0
        
        # Test course name resolution
        resolved = vector_store._resolve_course_name("Introduction")
        assert resolved == "Introduction to MCP"

    def test_real_search_with_filters(self, temp_chroma_path, sample_courses, sample_course_chunks):
        """Test real search with course and lesson filters"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2")
        
        # Add multiple courses
        for course in sample_courses:
            vector_store.add_course_metadata(course)
        
        vector_store.add_course_content(sample_course_chunks)
        
        # Test course-specific search
        results = vector_store.search("content", course_name="Introduction")
        assert not results.is_empty()
        
        # All results should be from the MCP course
        for metadata in results.metadata:
            assert "MCP" in metadata.get("course_title", "")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])