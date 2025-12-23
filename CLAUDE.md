# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Important:** Always use `uv` to run the server and manage dependencies. Do not use `pip` directly. Make sure to use `uv` to manage all dependencies.

### Install dependencies
```bash
uv sync
```

### Run the application
```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

Or use the shell script: `./run.sh`

### Access points
- Web UI: http://localhost:8000
- API docs: http://localhost:8000/docs

## Architecture

This is a RAG (Retrieval-Augmented Generation) chatbot that answers questions about course materials using Claude AI with tool-calling capabilities.

### Request Flow

```
Frontend (vanilla JS) → POST /api/query → FastAPI (app.py)
    → RAGSystem.query() orchestrates:
        1. SessionManager: retrieves conversation history
        2. AIGenerator: calls Claude API with tools
        3. Claude decides whether to use search_course_content tool
        4. If tool used: CourseSearchTool → VectorStore → ChromaDB semantic search
        5. Claude synthesizes final response from search results
        6. SessionManager: saves exchange to history
    → Response with answer + sources returned to frontend
```

### Backend Components (`backend/`)

| File | Role |
|------|------|
| `rag_system.py` | Main orchestrator - coordinates all components |
| `ai_generator.py` | Claude API client with tool execution loop |
| `search_tools.py` | Tool definitions (`CourseSearchTool`) and `ToolManager` |
| `vector_store.py` | ChromaDB wrapper with two collections: `course_catalog` (metadata) and `course_content` (chunks) |
| `document_processor.py` | Parses course files, chunks text with sentence-aware splitting and overlap |
| `session_manager.py` | Per-session conversation history |
| `config.py` | Settings loaded from environment (chunk size, model names, etc.) |

### Document Format

Course files in `docs/` must follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [Lesson Title]
Lesson Link: [url]
[Content...]

Lesson 1: [Lesson Title]
...
```

### Key Configuration (`backend/config.py`)

- `CHUNK_SIZE`: 800 characters per chunk
- `CHUNK_OVERLAP`: 100 characters overlap between chunks
- `MAX_RESULTS`: 5 search results per query
- `MAX_HISTORY`: 2 conversation exchanges retained
- `ANTHROPIC_MODEL`: claude-sonnet-4-20250514

### Environment Variables

Required in `.env` file at project root:
```
ANTHROPIC_API_KEY=your_key_here
```


### Git and Deployment
- Never mention claude code contribution in anything (including but not limited to commits and PRs)