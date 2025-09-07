"""
Unit tests for CourseSearchTool
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add backend to path  
sys.path.insert(0, str(Path(__file__).parent.parent))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


@pytest.mark.unit
class TestCourseSearchTool:
    """Test CourseSearchTool functionality"""

    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is correctly formatted for Anthropic API"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert schema["required"] == ["query"]
        
        # Optional parameters
        assert "course_name" in schema["properties"]
        assert "lesson_number" in schema["properties"]

    def test_execute_successful_search(self, mock_vector_store, mock_search_results):
        """Test successful search execution"""
        mock_vector_store.search.return_value = mock_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        # Verify search was called with correct parameters
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=None
        )
        
        # Verify result is formatted properly
        assert isinstance(result, str)
        assert "[Introduction to MCP - Lesson 1]" in result
        assert "This is sample course content about MCP." in result
        assert "[Advanced Web Development - Lesson 2]" in result

    def test_execute_with_course_name_filter(self, mock_vector_store, mock_search_results):
        """Test search with course name filter"""
        mock_vector_store.search.return_value = mock_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="MCP")
        
        mock_vector_store.search.assert_called_once_with(
            query="test query", 
            course_name="MCP",
            lesson_number=None
        )

    def test_execute_with_lesson_number_filter(self, mock_vector_store, mock_search_results):
        """Test search with lesson number filter"""
        mock_vector_store.search.return_value = mock_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", lesson_number=1)
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None, 
            lesson_number=1
        )

    def test_execute_with_both_filters(self, mock_vector_store, mock_search_results):
        """Test search with both course name and lesson number filters"""
        mock_vector_store.search.return_value = mock_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="MCP", lesson_number=2)
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="MCP",
            lesson_number=2
        )

    def test_execute_empty_results(self, mock_vector_store, empty_search_results):
        """Test handling of empty search results"""
        mock_vector_store.search.return_value = empty_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("nonexistent topic")
        
        assert result == "No relevant content found."

    def test_execute_empty_results_with_filters(self, mock_vector_store, empty_search_results):
        """Test handling of empty results with filters"""
        mock_vector_store.search.return_value = empty_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="MCP", lesson_number=5)
        
        assert result == "No relevant content found in course 'MCP' in lesson 5."

    def test_execute_error_handling(self, mock_vector_store, error_search_results):
        """Test handling of search errors"""
        mock_vector_store.search.return_value = error_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        assert result == "Database connection failed"

    def test_format_results_with_links(self, mock_vector_store):
        """Test result formatting with lesson links"""
        # Create search results with lesson links
        search_results = SearchResults(
            documents=["Content about lesson 1"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        # Verify lesson link was requested
        mock_vector_store.get_lesson_link.assert_called_once_with("Test Course", 1)
        
        # Verify sources are stored with links
        assert len(tool.last_sources) == 1
        source = tool.last_sources[0]
        assert source["text"] == "Test Course - Lesson 1"
        assert source["link"] == "https://example.com/lesson1"

    def test_format_results_without_links(self, mock_vector_store):
        """Test result formatting when no lesson links available"""
        search_results = SearchResults(
            documents=["Content without lesson number"],
            metadata=[{"course_title": "Test Course"}],  # No lesson_number
            distances=[0.1]
        )
        mock_vector_store.search.return_value = search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        # Verify no lesson link was requested
        mock_vector_store.get_lesson_link.assert_not_called()
        
        # Verify sources are stored without links
        assert len(tool.last_sources) == 1
        source = tool.last_sources[0]
        assert source["text"] == "Test Course"
        assert "link" not in source

    def test_format_results_unknown_course(self, mock_vector_store):
        """Test result formatting with unknown course metadata"""
        search_results = SearchResults(
            documents=["Content with missing metadata"],
            metadata=[{}],  # Empty metadata
            distances=[0.1]
        )
        mock_vector_store.search.return_value = search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        assert "[unknown]" in result
        assert "Content with missing metadata" in result

    def test_last_sources_tracking(self, mock_vector_store, mock_search_results):
        """Test that last sources are properly tracked"""
        mock_vector_store.search.return_value = mock_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"
        
        tool = CourseSearchTool(mock_vector_store)
        
        # Initially no sources
        assert tool.last_sources == []
        
        # Execute search
        tool.execute("test query")
        
        # Verify sources are tracked
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "Introduction to MCP - Lesson 1"
        assert tool.last_sources[1]["text"] == "Advanced Web Development - Lesson 2"

    def test_query_parameter_validation(self, mock_vector_store):
        """Test that query parameter is required and handled correctly"""
        tool = CourseSearchTool(mock_vector_store)
        
        # Test with empty query
        mock_vector_store.search.return_value = SearchResults.empty("No query provided")
        result = tool.execute("")
        
        mock_vector_store.search.assert_called_once_with(
            query="",
            course_name=None, 
            lesson_number=None
        )


@pytest.mark.unit
class TestToolManager:
    """Test ToolManager functionality"""

    def test_register_tool(self, mock_vector_store):
        """Test tool registration"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        
        manager.register_tool(tool)
        
        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == tool

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting tool definitions"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        definitions = manager.get_tool_definitions()
        
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_execute_tool(self, mock_vector_store, mock_search_results):
        """Test tool execution through manager"""
        mock_vector_store.search.return_value = mock_search_results
        
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        result = manager.execute_tool("search_course_content", query="test query")
        
        assert isinstance(result, str)
        assert "Introduction to MCP" in result

    def test_execute_nonexistent_tool(self):
        """Test executing non-existent tool"""
        manager = ToolManager()
        
        result = manager.execute_tool("nonexistent_tool", query="test")
        
        assert result == "Tool 'nonexistent_tool' not found"

    def test_get_last_sources(self, mock_vector_store, mock_search_results):
        """Test getting sources from last search"""
        mock_vector_store.search.return_value = mock_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"
        
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        # Initially no sources
        assert manager.get_last_sources() == []
        
        # Execute search
        manager.execute_tool("search_course_content", query="test query")
        
        # Get sources
        sources = manager.get_last_sources()
        assert len(sources) == 2

    def test_reset_sources(self, mock_vector_store, mock_search_results):
        """Test resetting sources"""
        mock_vector_store.search.return_value = mock_search_results
        
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        # Execute search to generate sources
        manager.execute_tool("search_course_content", query="test query")
        assert len(manager.get_last_sources()) > 0
        
        # Reset sources
        manager.reset_sources()
        assert manager.get_last_sources() == []

    def test_register_tool_without_name(self, mock_vector_store):
        """Test registering tool without name raises error"""
        manager = ToolManager()
        
        # Create a mock tool without name
        bad_tool = Mock()
        bad_tool.get_tool_definition.return_value = {"description": "Bad tool"}
        
        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            manager.register_tool(bad_tool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])