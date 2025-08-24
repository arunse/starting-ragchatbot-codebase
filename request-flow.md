# RAG Chatbot Request Flow

This document outlines the complete request flow when a user asks a question in the RAG chatbot application.

## Frontend Flow (`script.js`)

1. **User Input**: User types question and clicks send or presses Enter
2. **Request Preparation**: `sendMessage()` function:
   - Extracts query from input field  
   - Adds user message to chat UI
   - Disables input while processing

3. **API Call**: POST to `/api/query` with:
   ```javascript
   {
     query: "user question",
     session_id: currentSessionId
   }
   ```

## Backend Flow (`app.py` → `rag_system.py`)

4. **FastAPI Endpoint**: `query_documents()` receives request
   - Creates session if none provided
   - Calls `rag_system.query()`

5. **RAG System Processing**: `rag_system.query()`:
   - Gets conversation history from `session_manager`
   - Calls `ai_generator.generate_response()` with:
     - User query + prompt template
     - Conversation history 
     - Available search tools

6. **AI Generation**: `ai_generator.py`:
   - Uses Anthropic Claude API with system prompt
   - Claude decides if it needs to search course content
   - If needed, calls the `CourseSearchTool`

7. **Vector Search** (if triggered): `vector_store.py`:
   - Converts query to embeddings using sentence-transformers
   - Searches ChromaDB for relevant course chunks
   - Returns top matches with metadata

8. **Response Generation**: 
   - Claude synthesizes search results with knowledge
   - Returns final answer to RAG system
   - Session manager stores conversation

9. **API Response**: Returns to frontend:
   ```json
   {
     "answer": "AI response",
     "sources": ["source1", "source2"], 
     "session_id": "session_id"
   }
   ```

10. **Frontend Display**: 
    - Shows AI response in chat
    - Re-enables input field
    - Updates session state

## Key Components

- **ChromaDB**: Vector database storing course embeddings at `./backend/chroma_db/`
- **Session Management**: Maintains conversation context per user
- **Smart Search**: Claude intelligently decides when to search vs. use general knowledge
- **Tool-based RAG**: Uses Anthropic's tool calling for search decisions

## Architecture Notes

The key insight is that Claude intelligently decides whether to search the vector store based on the query type - it only searches for course-specific questions, not general knowledge questions.