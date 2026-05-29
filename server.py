from typing import Optional, List
import sqlite3
from contextlib import closing, asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

# ─── Safe import of LLM functions ─────────────────────────────────────
try:
    from services.llm import get_llm_reply, stream_llm_reply, get_available_models
except ImportError as e:
    import logging
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("server")
    logger.error(f"LLM module import failed: {e}")

    def get_llm_reply(*args, **kwargs):
        return "Error: LLM module not loaded (check services/llm.py)."

    def stream_llm_reply(*args, **kwargs):
        yield "Error: LLM module not loaded."

    def get_available_models():
        return []

DB_PATH = "chat_history.db"
MAX_HISTORY_FOR_MODEL = 6

# ─── Database helpers ──────────────────────────────────────────────────
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def create_tables():
    with closing(get_db_connection()) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT 'New chat',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
            CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at DESC);
        """)
        conn.commit()

# ─── Lifespan context manager (replaces deprecated @app.on_event) ──────
@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    message: str
    model: Optional[str] = None
    conversation_id: Optional[int] = None
    title: Optional[str] = None
    memory: bool = True

class CreateConversationRequest(BaseModel):
    title: Optional[str] = None

# ─── Conversation helpers ──────────────────────────────────────────────
def row_to_conversation(row):
    return {
        "id": row["id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }

def row_to_message(row):
    return {
        "id": row["id"],
        "conversation_id": row["conversation_id"],
        "role": row["role"],
        "content": row["content"],
        "created_at": row["created_at"],
    }

def generate_conversation_title(user_message: str) -> str:
    cleaned = " ".join(user_message.strip().split())
    return cleaned[:60] if cleaned else "New chat"

def create_conversation(title: Optional[str] = None) -> int:
    safe_title = (title or "New chat").strip() or "New chat"
    with closing(get_db_connection()) as conn:
        cursor = conn.execute("INSERT INTO conversations (title) VALUES (?)", (safe_title,))
        conn.commit()
        return cursor.lastrowid

def get_conversation(conversation_id: int):
    with closing(get_db_connection()) as conn:
        return conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()

def maybe_update_conversation_title(conversation_id: int, user_message: str):
    with closing(get_db_connection()) as conn:
        row = conn.execute("SELECT title FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if row and (row["title"] or "").strip().lower() == "new chat":
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (generate_conversation_title(user_message), conversation_id),
            )
            conn.commit()

def add_message(conversation_id: int, role: str, content: str):
    content = content.strip()
    if not content:
        return
    with closing(get_db_connection()) as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation_id,),
        )
        conn.commit()

def get_conversation_messages(conversation_id: int, limit: Optional[int] = None):
    with closing(get_db_connection()) as conn:
        if limit:
            rows = conn.execute(
                "SELECT id, conversation_id, role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
            rows = list(reversed(rows))
        else:
            rows = conn.execute(
                "SELECT id, conversation_id, role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,),
            ).fetchall()
        return [row_to_message(r) for r in rows]

def get_recent_history_for_model(conversation_id: int, limit: int = MAX_HISTORY_FOR_MODEL) -> List[ChatMessage]:
    msgs = get_conversation_messages(conversation_id, limit=limit)
    return [ChatMessage(role=m["role"], content=m["content"]) for m in msgs]

# ─── Endpoints ─────────────────────────────────────────────────────────
@app.get("/")
def home():
    return FileResponse("frontend/index.html")

@app.get("/models")
def models():
    return {"models": get_available_models()}

@app.get("/conversations")
def list_conversations():
    with closing(get_db_connection()) as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC, id DESC"
        ).fetchall()
        return {"conversations": [row_to_conversation(r) for r in rows]}

@app.post("/conversations")
def create_new_conversation(request: CreateConversationRequest):
    conv_id = create_conversation(request.title)
    with closing(get_db_connection()) as conn:
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
    return {"conversation": row_to_conversation(row)}

@app.get("/conversations/{conversation_id}")
def get_single_conversation(conversation_id: int):
    row = get_conversation(conversation_id)
    if not row:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation": row_to_conversation(row)}

@app.get("/conversations/{conversation_id}/messages")
def list_messages(conversation_id: int):
    row = get_conversation(conversation_id)
    if not row:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = get_conversation_messages(conversation_id)
    return {"messages": messages}

@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: int):
    with closing(get_db_connection()) as conn:
        cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Conversation not found")
    return {"success": True, "deleted_conversation_id": conversation_id}

@app.post("/chat")
def chat(request: ChatRequest):
    user_message = request.message.strip()
    if not user_message:
        return {"reply": "Please type a message."}

    conversation_id = request.conversation_id
    if conversation_id is None:
        conversation_id = create_conversation(request.title or generate_conversation_title(user_message))
    else:
        if not get_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")

    maybe_update_conversation_title(conversation_id, user_message)

    history = []
    if request.memory:
        history = get_recent_history_for_model(conversation_id, limit=MAX_HISTORY_FOR_MODEL)

    add_message(conversation_id, "user", user_message)

    try:
        reply = get_llm_reply(
            user_message=user_message,
            model_name=request.model,
            history=history,
        )
    except Exception as e:
        return {"reply": f"Error: {str(e)}", "conversation_id": conversation_id}

    add_message(conversation_id, "assistant", reply)

    updated_conv = get_conversation(conversation_id)
    return {
        "reply": reply,
        "conversation_id": conversation_id,
        "conversation": row_to_conversation(updated_conv) if updated_conv else None,
    }

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream the assistant reply using Server‑Sent Events."""
    user_message = request.message.strip()
    if not user_message:
        return StreamingResponse(
            iter(["data: Please type a message.\n\n"]),
            media_type="text/event-stream"
        )

    conversation_id = request.conversation_id
    if conversation_id is None:
        conversation_id = create_conversation(request.title or generate_conversation_title(user_message))
    else:
        if not get_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")

    maybe_update_conversation_title(conversation_id, user_message)

    history = []
    if request.memory:
        history = get_recent_history_for_model(conversation_id, limit=MAX_HISTORY_FOR_MODEL)

    add_message(conversation_id, "user", user_message)

    async def generate():
        full_reply = ""
        try:
            for token in stream_llm_reply(
                user_message=user_message,
                model_name=request.model,
                history=history,
            ):
                # Keep the original token (with real newlines) for the database
                full_reply += token
                # Escape newlines for the SSE stream so data lines are not broken
                safe_token = token.replace("\n", "\\n")
                yield f"data: {safe_token}\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
        finally:
            if full_reply:
                add_message(conversation_id, "assistant", full_reply)

    return StreamingResponse(generate(), media_type="text/event-stream")