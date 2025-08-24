# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Package Management
```bash
# Install dependencies
uv sync

# Add new dependency
uv add package_name
```

### Environment Setup
- Create `.env` file in root with `ANTHROPIC_API_KEY=your_key_here`
- Requires Python 3.13+ and uv package manager
- Access application at http://localhost:8000

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for querying course materials using semantic search and AI-powered responses.

### Core Architecture Pattern

The system follows a **tool-based RAG pattern** where Claude decides intelligently whether to search the vector store:
- **General knowledge questions**: Answered using Claude's training data without searching
- **Course-specific questions**: Triggers vector search first, then synthesizes results

### Key Components

**RAGSystem** (`rag_system.py`): Central orchestrator that coordinates all components
- Initializes and manages: DocumentProcessor, VectorStore, AIGenerator, SessionManager, ToolManager
- Main entry point: `query(query, session_id)` method

**VectorStore** (`vector_store.py`): ChromaDB integration with dual collections
- `course_catalog`: Course metadata (titles, instructors, lesson links)
- `course_content`: Text chunks for semantic search
- Persistent storage at `./backend/chroma_db/`

**AIGenerator** (`ai_generator.py`): Anthropic Claude API wrapper with tool calling
- Uses system prompt that instructs Claude on when to search vs. use general knowledge
- Manages conversation flow and tool execution

**DocumentProcessor** (`document_processor.py`): Parses course transcript files
- Extracts course metadata from structured text format
- Chunks content (800 chars, 100 overlap) for vector storage

**SessionManager** (`session_manager.py`): Maintains conversation context per user
- Stores conversation history with configurable max length

### Data Models (`models.py`)

- **Course**: Course metadata with lessons list
- **Lesson**: Individual lesson with number, title, link
- **CourseChunk**: Text chunk with course/lesson association for vector storage

### Request Flow

1. Frontend POST to `/api/query` with `{query, session_id}`
2. FastAPI calls `rag_system.query()`
3. Gets conversation history from session manager
4. Calls `ai_generator.generate_response()` with tools
5. Claude decides whether to call `CourseSearchTool`
6. If searching: vector similarity search in ChromaDB
7. Claude synthesizes results and returns answer
8. Session updated with conversation

### Frontend Architecture

- **Pure HTML/CSS/JavaScript** (no frameworks)
- Single-page application with chat interface
- Course statistics sidebar with collapsible sections
- Session-based conversation management

### Configuration (`config.py`)

Key settings:
- `CHROMA_PATH`: "./chroma_db" (ChromaDB storage)
- `CHUNK_SIZE`: 800 chars, `CHUNK_OVERLAP`: 100 chars
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2" (sentence-transformers)
- `MAX_RESULTS`: 5 (vector search results)
- `MAX_HISTORY`: 2 (conversation context)

## Development Notes

### Document Processing
Course documents must follow specific format in `/docs/`:
```
Course Title: [Title]
Course Link: [URL]
Course Instructor: [Name]

Lesson 0: Introduction
Lesson Link: [URL]
[Content...]
```

### Vector Store Initialization
- Documents auto-loaded on app startup from `../docs` folder
- ChromaDB creates persistent storage automatically
- Embedding model downloads on first run

### API Endpoints
- `POST /api/query`: Main chat endpoint
- `GET /api/courses`: Course statistics
- `GET /docs`: FastAPI auto-generated documentation

### Tool-Based Search
The `CourseSearchTool` in `search_tools.py` provides Claude with:
- Semantic search across course content
- Course name resolution (fuzzy matching)
- Filtering by course title or lesson number
- Returns structured results with metadata
- always use use uv to run the server. do not use pip directly
- make sure you use uv to manage all dependencies
- use uv to run python files