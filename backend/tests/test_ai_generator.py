"""
Unit tests for AIGenerator
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool


@pytest.mark.unit
class TestAIGenerator:
    """Test AIGenerator functionality"""

    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator instance for testing"""
        return AIGenerator(api_key="test-api-key", model="claude-sonnet-4-20250514")

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic client"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            yield mock_client

    def test_initialization(self, ai_generator):
        """Test AIGenerator initialization"""
        assert ai_generator.model == "claude-sonnet-4-20250514"
        assert ai_generator.base_params["model"] == "claude-sonnet-4-20250514"
        assert ai_generator.base_params["temperature"] == 0
        assert ai_generator.base_params["max_tokens"] == 800

    def test_system_prompt_content(self, ai_generator):
        """Test that system prompt contains key instructions"""
        system_prompt = ai_generator.SYSTEM_PROMPT
        
        # Check for key instruction phrases
        assert "course materials" in system_prompt.lower()
        assert "search_course_content" in system_prompt
        assert "get_course_outline" in system_prompt
        assert "general knowledge questions" in system_prompt.lower()
        assert "course-specific content questions" in system_prompt.lower()

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic):
        """Test response generation without tools"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response without tools"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        ai_generator = AIGenerator("test-key", "test-model")
        result = ai_generator.generate_response("What is 2+2?")

        # Verify API call
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        
        assert call_args["model"] == "test-model"
        assert call_args["messages"][0]["content"] == "What is 2+2?"
        assert call_args["messages"][0]["role"] == "user"
        assert "tools" not in call_args
        assert result == "Test response without tools"

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tools(self, mock_anthropic, mock_tool_manager):
        """Test response generation with tools available"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response with tools available"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        ai_generator = AIGenerator("test-key", "test-model")
        tools = mock_tool_manager.get_tool_definitions()
        
        result = ai_generator.generate_response(
            "Tell me about MCP",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify API call includes tools
        call_args = mock_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tool_choice"] == {"type": "auto"}
        assert result == "Test response with tools available"

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic):
        """Test response generation with conversation history"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response with context"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        ai_generator = AIGenerator("test-key", "test-model")
        history = "Previous conversation: User asked about courses."
        
        result = ai_generator.generate_response(
            "Follow up question",
            conversation_history=history
        )

        # Verify system prompt includes history
        call_args = mock_client.messages.create.call_args[1]
        system_content = call_args["system"]
        assert "Previous conversation:" in system_content
        assert history in system_content

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_flow(self, mock_anthropic, mock_tool_manager):
        """Test complete tool execution flow"""
        # Setup mocks
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # First response with tool use
        mock_tool_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "tool_123"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "MCP introduction"}
        mock_tool_response.content = [mock_tool_block]
        mock_tool_response.stop_reason = "tool_use"
        
        # Final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Based on the search results, MCP is..."
        
        # Configure mock client to return different responses on successive calls
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Configure tool manager
        mock_tool_manager.execute_tool.return_value = "Search results about MCP"
        
        ai_generator = AIGenerator("test-key", "test-model")
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        
        result = ai_generator.generate_response(
            "Tell me about MCP",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="MCP introduction"
        )
        
        assert result == "Based on the search results, MCP is..."

    @patch('ai_generator.anthropic.Anthropic')
    def test_handle_tool_execution_multiple_tools(self, mock_anthropic, mock_tool_manager):
        """Test handling multiple tool calls in one response"""
        # Setup mocks
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Response with multiple tool uses
        mock_tool_response = Mock()
        mock_tool_block1 = Mock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.id = "tool_1"
        mock_tool_block1.name = "search_course_content"
        mock_tool_block1.input = {"query": "MCP basics"}
        
        mock_tool_block2 = Mock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.id = "tool_2"
        mock_tool_block2.name = "get_course_outline"
        mock_tool_block2.input = {"course_name": "MCP"}
        
        mock_tool_response.content = [mock_tool_block1, mock_tool_block2]
        mock_tool_response.stop_reason = "tool_use"
        
        # Final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Combined response from multiple tools"
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        ai_generator = AIGenerator("test-key", "test-model")
        
        # Create initial API params as if from first call
        base_params = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Test query"}],
            "system": "Test system prompt"
        }
        
        result = ai_generator._handle_tool_execution(
            mock_tool_response, 
            base_params, 
            mock_tool_manager
        )

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="MCP basics")
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="MCP")

    def test_handle_tool_execution_message_construction(self, ai_generator, mock_tool_manager):
        """Test that messages are constructed properly for tool execution"""
        # Create mock initial response
        mock_initial_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "tool_123"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "test"}
        mock_initial_response.content = [mock_tool_block]
        
        mock_tool_manager.execute_tool.return_value = "Tool execution result"
        
        base_params = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Original query"}],
            "system": "System prompt"
        }
        
        with patch.object(ai_generator, 'client') as mock_client:
            mock_final_response = Mock()
            mock_final_response.content = [Mock()]
            mock_final_response.content[0].text = "Final response"
            mock_client.messages.create.return_value = mock_final_response
            
            result = ai_generator._handle_tool_execution(
                mock_initial_response,
                base_params,
                mock_tool_manager
            )
            
            # Verify the final API call has correct message structure
            final_call_args = mock_client.messages.create.call_args[1]
            messages = final_call_args["messages"]
            
            # Should have: original user message, assistant tool use, user tool results
            assert len(messages) == 3
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "Original query"
            assert messages[1]["role"] == "assistant"
            assert messages[2]["role"] == "user"
            
            # Tool result should be properly formatted
            tool_result = messages[2]["content"][0]
            assert tool_result["type"] == "tool_result"
            assert tool_result["tool_use_id"] == "tool_123"
            assert tool_result["content"] == "Tool execution result"

    @patch('ai_generator.anthropic.Anthropic')
    def test_api_error_handling(self, mock_anthropic):
        """Test handling of API errors"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API Error")
        
        ai_generator = AIGenerator("test-key", "test-model")
        
        with pytest.raises(Exception, match="API Error"):
            ai_generator.generate_response("Test query")

    def test_base_params_efficiency(self, ai_generator):
        """Test that base parameters are pre-built for efficiency"""
        # Verify base params are set during initialization
        assert "model" in ai_generator.base_params
        assert "temperature" in ai_generator.base_params
        assert "max_tokens" in ai_generator.base_params
        
        # These should be pre-computed values
        assert ai_generator.base_params["model"] == "claude-sonnet-4-20250514"
        assert ai_generator.base_params["temperature"] == 0
        assert ai_generator.base_params["max_tokens"] == 800


@pytest.mark.integration  
class TestAIGeneratorIntegration:
    """Integration tests with real components (but mocked API)"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_integration_with_real_tool_manager(self, mock_anthropic, mock_vector_store, mock_search_results):
        """Test AIGenerator with real ToolManager and CourseSearchTool"""
        # Setup Anthropic mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock tool use response
        mock_tool_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "tool_123"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "MCP introduction"}
        mock_tool_response.content = [mock_tool_block]
        mock_tool_response.stop_reason = "tool_use"
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Final integrated response"
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Setup real components
        mock_vector_store.search.return_value = mock_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"
        
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        
        ai_generator = AIGenerator("test-key", "test-model")
        
        # Execute
        result = ai_generator.generate_response(
            "Tell me about MCP",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify integration worked
        assert result == "Final integrated response"
        mock_vector_store.search.assert_called_once_with(
            query="MCP introduction",
            course_name=None,
            lesson_number=None
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])