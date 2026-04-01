# AGENTS.md

## Project Overview
A Streamlit-based application that generates YouTube thumbnail concepts and images from video transcripts. It uses a LangGraph pipeline to first plan the thumbnail ideas (structured output) and then render them using OpenAI's image generation models.

## Preferences and Dependencies
- **Runtime:** Python 3.12 or later.
- **Package Manager:** `pip` (standard `requirements.txt`).
- **Frameworks:** Streamlit (UI), LangGraph (Orchestration), Pydantic (Schema), LangChain/OpenAI (LLM/Images).
- **Environment:** Load secrets from `.env`. Required: `OPENAI_API_KEY`.
- **LLM Backend:** Prefers OpenAI (`gpt-4.1-mini` for text, `gpt-image-1.5` for images) as per current implementation, though `AGENTS.md` previously mentioned Ollama.

## Project Structure
- `youtube_thumbnail_app.py`: Main Streamlit application with LangGraph pipeline.
- `youtube_thumbnail_app_local.py`: Local version/backup of the application.
- `.env`: Environment variables (API keys).

## Key Files and Their Purposes
- **`youtube_thumbnail_app.py`**: Contains the `ThumbnailIdea` and `ThumbnailPlan` Pydantic models, the `plan_thumbnails` and `render_thumbnails` graph nodes, and the Streamlit UI logic.
- **`AppState`**: A `TypedDict` defining the state passed through the LangGraph: `transcript`, `count`, `ideas`, and `images`.

## Build and Test Commands
- **Install dependencies:**
  ```bash
  pip install streamlit pydantic openai langgraph langchain-openai rich
  ```
- **Run the application:**
  ```bash
  streamlit run youtube_thumbnail_app.py
  ```
- **Testing:** No formal test suite exists. Verify logic by running the Streamlit app and checking the "Concepts" and "Generated thumbnails" sections.
- **Linting:** Use `ruff` or `flake8`.
  ```bash
  ruff check .
  ```

## Code Style Guidelines

### 1. Imports
- Group imports: standard library, third-party packages, local modules.
- Use explicit imports (e.g., `from typing import List, TypedDict`).

### 2. Formatting & Types
- Follow PEP 8.
- Use type hints for all function signatures (e.g., `def build_graph() -> CompiledGraph:`).
- Use Pydantic `BaseModel` for structured LLM outputs to ensure data integrity.

### 3. Naming Conventions
- Variables/Functions: `snake_case`.
- Classes (Models): `PascalCase`.
- Constants: `UPPER_SNAKE_CASE` (e.g., `TRANSCRIPT_LIMIT = 12000`).

### 4. Error Handling
- Use `st.error()` for user-facing errors in Streamlit.
- Raise `ValueError` or specific exceptions in LangGraph nodes if the LLM output fails validation (e.g., if `len(ideas) < state["count"]`).

### 5. LangGraph & LLMs
- **Structured Output:** Always use `.with_structured_output(Schema)` when expecting specific JSON formats from the LLM.
- **State Management:** Ensure all nodes return a dictionary that updates the `AppState`.
- **Prompts:** Use multi-line f-strings with `.strip()` for prompts. Include clear "Rules" or "Constraints" sections within the prompt.

### 6. UI (Streamlit)
- Keep configuration in the sidebar.
- Use `st.expander` for detailed metadata/logs.
- Provide `st.download_button` for generated assets.
