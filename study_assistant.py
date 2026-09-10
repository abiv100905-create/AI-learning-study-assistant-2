"""
study_assistant.py — AI Study Assistant core (RAG + Memory + Tools + Agent)

One file, four responsibilities, in order:
  1. Config          - env var loading
  2. RAG              - chunk + embed + search course materials (Chroma)
  3. Memory           - SQLite-backed conversation/facts/quiz history
  4. Tools + Agent    - Claude tool-use loop wiring it all together

Run it via cli.py, or `from study_assistant import StudyAssistantAgent`.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

# ============================================================================
# 1. CONFIG
# ============================================================================

@dataclass(frozen=True)
class Config:
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chroma_dir: str = os.getenv("CHROMA_DIR", "./chroma_db")
    sqlite_path: str = os.getenv("SQLITE_PATH", "./study_assistant.db")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    max_history_messages: int = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))

    def validate(self) -> None:
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env "
                "and add your key from https://console.anthropic.com"
            )


CONFIG = Config()

# ============================================================================
# 2. RAG — ingest, chunk, embed, search
# ============================================================================

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int


def _load_file_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {ext}")


def _chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """Word-based sliding-window chunking (simple, dependency-free)."""
    chunk_size = chunk_size or CONFIG.chunk_size
    overlap = overlap or CONFIG.chunk_overlap
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def discover_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_EXTENSIONS else []
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]


def load_and_chunk(root: str | Path) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in discover_files(root):
        try:
            text = _load_file_text(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {path.name}: {exc}")
            continue
        for i, piece in enumerate(_chunk_text(text)):
            all_chunks.append(Chunk(text=piece, source=path.name, chunk_index=i))
    return all_chunks


class VectorStore:
    """Persistent Chroma collection embedded locally via sentence-transformers."""

    def __init__(self, persist_dir: str = None, model_name: str = None):
        self.client = chromadb.PersistentClient(path=persist_dir or CONFIG.chroma_dir)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name or CONFIG.embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name="course_materials", embedding_function=self.embed_fn
        )

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        ids = [f"{c.source}::{c.chunk_index}" for c in chunks]
        docs = [c.text for c in chunks]
        metas = [{"source": c.source, "chunk_index": c.chunk_index} for c in chunks]
        self.collection.upsert(ids=ids, documents=docs, metadatas=metas)  # upsert = safe re-ingest
        return len(chunks)

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if self.collection.count() == 0:
            return []
        res = self.collection.query(query_texts=[query], n_results=top_k)
        hits = []
        for doc, meta, dist in zip(
            res.get("documents", [[]])[0], res.get("metadatas", [[]])[0], res.get("distances", [[]])[0]
        ):
            hits.append({"text": doc, "source": meta.get("source", "?"), "chunk_index": meta.get("chunk_index", -1), "distance": dist})
        return hits

    def list_sources(self) -> list[str]:
        if self.collection.count() == 0:
            return []
        data = self.collection.get()
        return sorted({m.get("source", "?") for m in data.get("metadatas", [])})


# ============================================================================
# 3. MEMORY — SQLite: conversation log, durable facts, quiz history
# ============================================================================

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
    role TEXT NOT NULL, content TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, fact TEXT NOT NULL UNIQUE, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS quiz_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT NOT NULL, score INTEGER NOT NULL,
    total INTEGER NOT NULL, weak_points TEXT, created_at REAL NOT NULL
);
"""


class MemoryManager:
    def __init__(self, db_path: str = None, session_id: str = "default"):
        self.db_path = db_path or CONFIG.sqlite_path
        self.session_id = session_id
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add_message(self, role: str, content: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (self.session_id, role, content, time.time()),
            )

    def get_recent_messages(self, limit: int = None) -> list[dict]:
        limit = limit or CONFIG.max_history_messages
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (self.session_id, limit),
            ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def remember_fact(self, fact: str) -> None:
        with self._conn() as conn:
            conn.execute("INSERT OR IGNORE INTO facts (fact, created_at) VALUES (?, ?)", (fact.strip(), time.time()))

    def recall_facts(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT fact FROM facts ORDER BY id ASC").fetchall()
        return [r[0] for r in rows]

    def record_quiz_result(self, topic: str, score: int, total: int, weak_points: list[str] | None = None) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO quiz_results (topic, score, total, weak_points, created_at) VALUES (?, ?, ?, ?, ?)",
                (topic, score, total, json.dumps(weak_points or []), time.time()),
            )

    def get_quiz_history(self, limit: int = 10) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT topic, score, total, weak_points FROM quiz_results ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{"topic": t, "score": s, "total": tot, "weak_points": json.loads(wp) if wp else []} for t, s, tot, wp in rows]

    def context_summary(self) -> str:
        facts, quizzes = self.recall_facts(), self.get_quiz_history(limit=5)
        lines = []
        if facts:
            lines.append("Known facts about the learner:")
            lines += [f"- {f}" for f in facts]
        if quizzes:
            lines.append("Recent quiz performance:")
            for q in quizzes:
                wp = f" (weak on: {', '.join(q['weak_points'])})" if q["weak_points"] else ""
                lines.append(f"- {q['topic']}: {q['score']}/{q['total']}{wp}")
        return "\n".join(lines) if lines else "No prior history yet."


# ============================================================================
# 4. TOOLS + AGENT
# ============================================================================

TOOLS = [
    {
        "name": "search_course_materials",
        "description": "Semantic search over the learner's ingested course materials. Call this before answering anything that might be covered by their own notes/textbook.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query"],
        },
    },
    {
        "name": "create_study_plan",
        "description": "Generate a day-by-day study plan for a topic, scoped to a number of days and optional subtopics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "days": {"type": "integer"},
                "subtopics": {"type": "array", "items": {"type": "string"}},
                "hours_per_day": {"type": "number"},
            },
            "required": ["topic", "days"],
        },
    },
    {
        "name": "generate_quiz",
        "description": "Generate a multiple-choice quiz on a topic, grounded in source_text if supplied.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "num_questions": {"type": "integer"},
                "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                "source_text": {"type": "string"},
            },
            "required": ["topic"],
        },
    },
    {
        "name": "remember_fact",
        "description": "Store a durable fact about the learner (goals, weak topics, deadlines) for future sessions.",
        "input_schema": {"type": "object", "properties": {"fact": {"type": "string"}}, "required": ["fact"]},
    },
    {
        "name": "recall_facts",
        "description": "Retrieve previously remembered facts about the learner.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "record_quiz_result",
        "description": "Log a quiz outcome so future plans/quizzes can target weak topics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "score": {"type": "integer"},
                "total": {"type": "integer"},
                "weak_points": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["topic", "score", "total"],
        },
    },
]

_QUIZ_SYSTEM_PROMPT = (
    "You generate multiple-choice quiz questions. Respond with ONLY a JSON array, "
    "no preamble, no markdown fences. Each element: "
    '{"question": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."}, '
    '"correct": "A", "explanation": "..."}'
)


def _generate_quiz(client: anthropic.Anthropic, topic: str, num_questions: int = 5,
                    difficulty: str = "medium", source_text: str | None = None) -> list[dict]:
    grounding = f"\n\nBase the questions strictly on this source material where possible:\n{source_text}" if source_text else ""
    resp = client.messages.create(
        model=CONFIG.claude_model,
        max_tokens=2000,
        system=_QUIZ_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Create {num_questions} {difficulty}-difficulty MCQs about: {topic}.{grounding}"}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text")
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse quiz JSON:\n{raw}")
        return json.loads(match.group(0))


def _format_quiz(questions: list[dict]) -> tuple[str, str]:
    display = "\n".join(
        f"\n{i}. {q['question']}\n" + "\n".join(f"   {k}) {v}" for k, v in q["options"].items())
        for i, q in enumerate(questions, 1)
    )
    key = "\nAnswer key:\n" + "\n".join(f"{i}. {q['correct']} — {q.get('explanation', '')}" for i, q in enumerate(questions, 1))
    return display, key


def _build_plan_skeleton(topic: str, days: int, subtopics: list[str] | None = None, hours_per_day: float = 1.5) -> dict:
    days = max(1, days)
    subtopics = subtopics or [topic]
    review_days = max(1, round(days * 0.2))
    learning_days = max(1, days - review_days)
    schedule = []
    for i in range(learning_days):
        focus = [subtopics[j] for j in range(len(subtopics)) if j % learning_days == i] or [subtopics[i % len(subtopics)]]
        schedule.append({"day": i + 1, "focus": focus, "hours": hours_per_day, "activity": "Learn + active recall"})
    for r in range(review_days):
        last = r == review_days - 1
        schedule.append({
            "day": learning_days + r + 1,
            "focus": ["Final review + full practice quiz"] if last else ["Cumulative review"],
            "hours": hours_per_day,
            "activity": "Spaced repetition + self-quiz",
        })
    return {"topic": topic, "total_days": days, "hours_per_day": hours_per_day, "schedule": schedule}


_SYSTEM_PROMPT_TEMPLATE = """You are a patient, encouraging AI study assistant.

You have tools to: search the learner's own course materials, build study
plans, generate quizzes, and remember/recall durable facts about the learner.

Rules:
- When the learner asks something their course materials might answer, call
  search_course_materials BEFORE answering, and cite the source filename.
- If search_course_materials returns nothing relevant, say so plainly and
  answer from general knowledge instead — never invent a citation.
- Use create_study_plan for scheduling requests.
- Use generate_quiz when asked to quiz the learner; pass retrieved material
  in as source_text when you have it, so questions are grounded.
- Use remember_fact for durable info only, not every message.

Learner memory:
{memory_summary}
"""


class StudyAssistantAgent:
    """Claude tool-use loop wiring RAG search, memory, planning, and quizzes together."""

    def __init__(self, session_id: str = "default"):
        CONFIG.validate()
        self.client = anthropic.Anthropic(api_key=CONFIG.anthropic_api_key)
        self.memory = MemoryManager(session_id=session_id)
        self.vectorstore = VectorStore()

    def _run_tool(self, name: str, tool_input: dict) -> str:
        if name == "search_course_materials":
            hits = self.vectorstore.search(tool_input["query"], top_k=tool_input.get("top_k", 4))
            if not hits:
                return "No matching course material found (has anything been ingested yet?)."
            return "\n\n".join(f"[{h['source']} #{h['chunk_index']}] {h['text']}" for h in hits)

        if name == "create_study_plan":
            plan = _build_plan_skeleton(
                tool_input["topic"], tool_input["days"], tool_input.get("subtopics"), tool_input.get("hours_per_day", 1.5)
            )
            self.memory.remember_fact(f"Is preparing for: {tool_input['topic']} over {tool_input['days']} days")
            return json.dumps(plan)

        if name == "generate_quiz":
            questions = _generate_quiz(
                self.client, tool_input["topic"], tool_input.get("num_questions", 5),
                tool_input.get("difficulty", "medium"), tool_input.get("source_text"),
            )
            display, key = _format_quiz(questions)
            return f"QUIZ:{display}\n\n(Show questions first; reveal this answer key only after the learner attempts them){key}"

        if name == "remember_fact":
            self.memory.remember_fact(tool_input["fact"])
            return "Saved."

        if name == "recall_facts":
            facts = self.memory.recall_facts()
            return "\n".join(facts) if facts else "No facts stored yet."

        if name == "record_quiz_result":
            self.memory.record_quiz_result(
                tool_input["topic"], tool_input["score"], tool_input["total"], tool_input.get("weak_points")
            )
            return "Recorded."

        return f"Unknown tool: {name}"

    def chat(self, user_message: str) -> str:
        self.memory.add_message("user", user_message)
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(memory_summary=self.memory.context_summary())
        messages = self.memory.get_recent_messages()

        while True:
            response = self.client.messages.create(
                model=CONFIG.claude_model, max_tokens=2000, system=system_prompt, tools=TOOLS, messages=messages
            )
            if response.stop_reason != "tool_use":
                final_text = "".join(b.text for b in response.content if b.type == "text")
                self.memory.add_message("assistant", final_text)
                return final_text

            messages.append({"role": "assistant", "content": response.content})
            tool_results = [
                {"type": "tool_result", "tool_use_id": b.id, "content": self._run_tool(b.name, b.input)}
                for b in response.content if b.type == "tool_use"
            ]
            messages.append({"role": "user", "content": tool_results})
