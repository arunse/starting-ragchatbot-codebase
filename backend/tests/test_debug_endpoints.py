"""
Debug endpoints for troubleshooting RAG system issues
"""
import pytest
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app import app


@pytest.mark.diagnostic
class TestDebugEndpoints:
    """Debug tests to help troubleshoot user issues"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    def test_debug_specific_user_queries(self, client):
        """Test specific queries that might be causing 'query failed' issues"""
        
        # Test queries that users commonly have trouble with
        problematic_queries = [
            "How do I use this system?",
            "What courses are available?", 
            "Tell me about the first lesson",
            "Help me understand embeddings",
            "What is MCP introduction?",
            "How do I build AI applications?",
            "",  # Empty query
            "xyzabc123",  # Nonsense query
            "Tell me about quantum physics",  # Topic not in courses
        ]
        
        print("\n=== DEBUG: Testing Problematic Queries ===")
        
        for query in problematic_queries:
            print(f"\nTesting query: '{query}'")
            
            response = client.post(
                "/api/query",
                json={"query": query, "session_id": "debug_session"}
            )
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"][:100] + "..." if len(data["answer"]) > 100 else data["answer"]
                print(f"Answer preview: {answer}")
                print(f"Sources count: {len(data.get('sources', []))}")
                
                if data.get('sources'):
                    print(f"First source: {data['sources'][0]}")
                    
            else:
                print(f"ERROR: {response.text}")

    def test_debug_session_behavior(self, client):
        """Test session behavior that might cause issues"""
        print("\n=== DEBUG: Testing Session Behavior ===")
        
        # Test without session
        response1 = client.post("/api/query", json={"query": "What is MCP?"})
        print(f"No session provided - Status: {response1.status_code}")
        if response1.status_code == 200:
            session1 = response1.json().get("session_id")
            print(f"Created session: {session1}")
        
        # Test with explicit session
        response2 = client.post("/api/query", json={"query": "Tell me more", "session_id": "explicit_session"})
        print(f"Explicit session - Status: {response2.status_code}")
        if response2.status_code == 200:
            session2 = response2.json().get("session_id")
            print(f"Used session: {session2}")

    def test_debug_response_format(self, client):
        """Debug response format issues"""
        print("\n=== DEBUG: Testing Response Format ===")
        
        response = client.post(
            "/api/query",
            json={"query": "What is MCP?", "session_id": "format_test"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response keys: {list(data.keys())}")
            print(f"Answer type: {type(data.get('answer'))}")
            print(f"Sources type: {type(data.get('sources'))}")
            print(f"Session ID type: {type(data.get('session_id'))}")
            
            if isinstance(data.get('sources'), list) and data['sources']:
                first_source = data['sources'][0]
                print(f"First source type: {type(first_source)}")
                if isinstance(first_source, dict):
                    print(f"Source keys: {list(first_source.keys())}")
                    
        else:
            print(f"Failed response: {response.text}")

    def test_debug_course_statistics(self, client):
        """Debug course statistics endpoint"""
        print("\n=== DEBUG: Testing Course Statistics ===")
        
        response = client.get("/api/courses")
        print(f"Courses endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Total courses: {data.get('total_courses')}")
            print(f"Course titles: {data.get('course_titles', [])}")
        else:
            print(f"Courses endpoint error: {response.text}")

    def test_debug_server_health(self, client):
        """Test basic server health"""
        print("\n=== DEBUG: Testing Server Health ===")
        
        # Test if server responds to basic requests
        response = client.get("/docs")  # FastAPI docs endpoint
        print(f"Docs endpoint status: {response.status_code}")
        
        # Test if basic API structure is working
        response = client.post("/api/new-session")
        print(f"New session endpoint status: {response.status_code}")

    def test_debug_anthropic_integration(self, client):
        """Debug Anthropic API integration"""
        print("\n=== DEBUG: Testing Anthropic Integration ===")
        
        # Test simple query that should work
        response = client.post(
            "/api/query",
            json={"query": "What is 2+2?", "session_id": "math_test"}
        )
        
        print(f"Math query status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "").lower()
            if "4" in answer:
                print("✅ Anthropic API is working (answered math correctly)")
            else:
                print(f"❌ Unexpected math answer: {data.get('answer')}")
        else:
            print(f"❌ Math query failed: {response.text}")

    def test_debug_tool_calling(self, client):
        """Debug tool calling functionality"""
        print("\n=== DEBUG: Testing Tool Calling ===")
        
        # Test query that should trigger tool use
        response = client.post(
            "/api/query",
            json={"query": "What courses do you have about MCP?", "session_id": "tool_test"}
        )
        
        print(f"Course query status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            sources = data.get("sources", [])
            if sources:
                print(f"✅ Tool calling working (got {len(sources)} sources)")
                for i, source in enumerate(sources[:2]):  # Show first 2 sources
                    print(f"  Source {i+1}: {source}")
            else:
                print("❌ No sources returned - tool calling may not be working")
                print(f"Answer: {data.get('answer', '')[:200]}...")
        else:
            print(f"❌ Course query failed: {response.text}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])