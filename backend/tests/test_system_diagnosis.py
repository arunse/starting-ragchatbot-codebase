"""
System diagnostic tests to identify basic configuration and data issues
"""
import pytest
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from vector_store import VectorStore
from rag_system import RAGSystem
import chromadb


@pytest.mark.diagnostic
class TestSystemDiagnosis:
    """Diagnostic tests to identify system issues"""

    def test_api_key_configuration(self):
        """Test if Anthropic API key is configured"""
        assert config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY is not set in config"
        assert config.ANTHROPIC_API_KEY != "", "ANTHROPIC_API_KEY is empty"
        assert len(config.ANTHROPIC_API_KEY) > 10, "ANTHROPIC_API_KEY appears to be invalid (too short)"

    def test_env_file_exists(self):
        """Test if .env file exists and is readable"""
        env_path = Path(__file__).parent.parent.parent / ".env"
        assert env_path.exists(), f".env file not found at {env_path}"
        assert env_path.is_file(), f".env path exists but is not a file: {env_path}"

    def test_documents_folder_exists(self):
        """Test if course documents folder exists and has files"""
        docs_path = Path(__file__).parent.parent.parent / "docs"
        assert docs_path.exists(), f"Documents folder not found at {docs_path}"
        
        doc_files = list(docs_path.glob("*.txt")) + list(docs_path.glob("*.pdf")) + list(docs_path.glob("*.docx"))
        assert len(doc_files) > 0, f"No course documents found in {docs_path}"
        print(f"Found {len(doc_files)} document files: {[f.name for f in doc_files]}")

    def test_chromadb_exists_and_has_data(self):
        """Test if ChromaDB exists and contains course data"""
        chroma_path = Path(__file__).parent.parent / config.CHROMA_PATH
        print(f"Checking ChromaDB at: {chroma_path}")
        
        if not chroma_path.exists():
            pytest.fail(f"ChromaDB path does not exist: {chroma_path}")

        # Try to connect to ChromaDB
        try:
            client = chromadb.PersistentClient(path=str(chroma_path))
            collections = client.list_collections()
            collection_names = [c.name for c in collections]
            print(f"Found ChromaDB collections: {collection_names}")
            
            # Check for expected collections
            assert "course_catalog" in collection_names, "Missing 'course_catalog' collection"
            assert "course_content" in collection_names, "Missing 'course_content' collection"
            
            # Check if collections have data
            catalog_collection = client.get_collection("course_catalog")
            content_collection = client.get_collection("course_content")
            
            catalog_count = catalog_collection.count()
            content_count = content_collection.count()
            
            print(f"Course catalog has {catalog_count} entries")
            print(f"Course content has {content_count} entries")
            
            assert catalog_count > 0, "Course catalog collection is empty"
            assert content_count > 0, "Course content collection is empty"
            
        except Exception as e:
            pytest.fail(f"Failed to connect to or query ChromaDB: {str(e)}")

    def test_vector_store_initialization(self):
        """Test if VectorStore can be initialized"""
        try:
            vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
            assert vector_store is not None, "VectorStore initialization failed"
            
            # Test basic functionality
            course_count = vector_store.get_course_count()
            course_titles = vector_store.get_existing_course_titles()
            
            print(f"VectorStore reports {course_count} courses")
            print(f"Course titles: {course_titles}")
            
            assert course_count > 0, "VectorStore reports no courses"
            assert len(course_titles) > 0, "VectorStore reports no course titles"
            
        except Exception as e:
            pytest.fail(f"VectorStore initialization failed: {str(e)}")

    def test_sentence_transformer_model(self):
        """Test if sentence transformer model can be loaded"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(config.EMBEDDING_MODEL)
            
            # Test basic encoding
            test_text = "This is a test sentence for embedding"
            embedding = model.encode(test_text)
            
            assert embedding is not None, "Failed to generate embedding"
            assert len(embedding) > 0, "Empty embedding generated"
            print(f"Successfully generated embedding of size {len(embedding)}")
            
        except Exception as e:
            pytest.fail(f"Sentence transformer model failed: {str(e)}")

    def test_rag_system_initialization(self):
        """Test if RAGSystem can be initialized without errors"""
        try:
            rag_system = RAGSystem(config)
            assert rag_system is not None, "RAGSystem initialization failed"
            
            # Test basic analytics
            analytics = rag_system.get_course_analytics()
            assert analytics is not None, "Failed to get course analytics"
            assert "total_courses" in analytics, "Missing total_courses in analytics"
            assert "course_titles" in analytics, "Missing course_titles in analytics"
            
            print(f"RAGSystem analytics: {analytics}")
            
            total_courses = analytics["total_courses"]
            assert total_courses > 0, f"RAGSystem reports {total_courses} courses (expected > 0)"
            
        except Exception as e:
            pytest.fail(f"RAGSystem initialization failed: {str(e)}")

    def test_basic_vector_search(self):
        """Test if basic vector search works"""
        try:
            vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
            
            # Try a basic search
            results = vector_store.search("introduction")
            
            assert results is not None, "Search returned None"
            assert not results.error, f"Search returned error: {results.error}"
            
            print(f"Basic search returned {len(results.documents)} documents")
            if results.documents:
                print(f"First result: {results.documents[0][:100]}...")
                print(f"First metadata: {results.metadata[0]}")
            
        except Exception as e:
            pytest.fail(f"Basic vector search failed: {str(e)}")

    def test_anthropic_api_connectivity(self):
        """Test if we can connect to Anthropic API (basic connectivity test)"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            
            # Just test client creation - we won't make actual API calls in diagnostics
            # to avoid costs, but we can verify the client initializes properly
            assert client is not None, "Anthropic client initialization failed"
            print("Anthropic client initialized successfully")
            
        except Exception as e:
            pytest.fail(f"Anthropic API client initialization failed: {str(e)}")

    @pytest.mark.slow
    def test_full_query_pipeline(self):
        """Test the complete query pipeline with a simple question"""
        try:
            rag_system = RAGSystem(config)
            
            # Test with a general question (should not use tools)
            response, sources = rag_system.query("What is 2+2?", session_id="test_session")
            
            assert response is not None, "Query returned None response"
            assert isinstance(response, str), f"Response is not string, got {type(response)}"
            assert len(response) > 0, "Response is empty"
            
            print(f"General query response: {response}")
            print(f"Sources: {sources}")
            
            # Test with a course-specific question (should use tools)
            course_response, course_sources = rag_system.query(
                "Tell me about MCP introduction", 
                session_id="test_session"
            )
            
            assert course_response is not None, "Course query returned None response"
            assert isinstance(course_response, str), f"Course response is not string, got {type(course_response)}"
            assert len(course_response) > 0, "Course response is empty"
            
            print(f"Course query response: {course_response}")
            print(f"Course sources: {course_sources}")
            
        except Exception as e:
            pytest.fail(f"Full query pipeline test failed: {str(e)}")


if __name__ == "__main__":
    # Run diagnostic tests when executed directly
    pytest.main([__file__, "-v", "-s"])