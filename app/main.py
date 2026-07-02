"""
FastAPI service: GET /health  and  POST /chat  + static chat UI.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Literal, Dict
import os
from dotenv import load_dotenv

load_dotenv()

from app.retrieval import CatalogRetriever
from app.agent import extract_state, classify_intent, generate_response, set_retriever

app = FastAPI(title="SHL Assessment Recommender", version="2.0.0")
retriever = CatalogRetriever(catalog_path="catalog.json")
set_retriever(retriever)  # give agent access to full catalog for REFINE fallback


# ── Models ────────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    messages = [msg.model_dump() for msg in request.messages]

    turn_count = sum(1 for m in messages if m["role"] == "user")
    if turn_count > 8:
        raise HTTPException(status_code=400, detail="Conversation exceeds 8 turns")

    # 1. Extract state
    state = extract_state(messages)

    # 2. Classify intent
    intent = classify_intent(messages, state, turn_count)

    # 3. Retrieve candidates
    candidates = []
    if intent.intent in ["RECOMMEND", "REFINE", "COMPARE"]:
        if intent.intent == "COMPARE":
            # COMPARE needs a broad pool: search full catalog with the assessment names
            compare_query = " ".join(intent.compare_names or ["assessment"])
            candidates = retriever.retrieve(
                query=compare_query,
                top_k=50,   # wide net so both named assessments are found
            )
            # If we didn't catch them, also append the full catalog top-50 by BM25
            if len(candidates) < 10:
                candidates = retriever.retrieve(query="assessment", top_k=50)
        else:
            query_parts = []
            if state.role:             query_parts.append(state.role)
            if state.seniority:        query_parts.append(state.seniority)
            query_parts.extend(state.skills[:5])
            if state.competency_focus: query_parts.append(state.competency_focus)
            if state.job_description:  query_parts.append(state.job_description[:200])
            query = " ".join(query_parts) if query_parts else "assessment"

            # For REFINE: parse last user message for type keywords to boost
            last_msg_lower = " ".join(
                m["content"].lower() for m in messages[-2:] if m["role"] == "user"
            )
            refine_type_kws = {
                "P": ["personality","behaviour","behavior","opq"],
                "A": ["cognitive","ability","reasoning","numerical","verbal","inductive","verify"],
                "S": ["simulation","exercise","coding sim"],
                "B": ["situational","scenarios"],
                "M": ["motivation"],
                "C": ["competenc"],
                "D": ["360","feedback"],
            }
            boost_types_refine: List[str] = []
            if intent.intent == "REFINE":
                for ttype, kws in refine_type_kws.items():
                    if any(kw in last_msg_lower for kw in kws):
                        boost_types_refine.append(ttype)
            boost_types = boost_types_refine if boost_types_refine else (
                state.test_types_requested if intent.intent == "REFINE" else None
            )

            candidates = retriever.retrieve(
                query=query,
                job_level=state.seniority,
                remote_only=state.remote_required,
                adaptive_only=state.adaptive_required,
                language=state.language,
                duration_max=state.duration_max_minutes,
                boost_types=boost_types,
                top_k=30,
            )

    # 4. Generate response — pass retriever so REFINE can fetch extra types
    valid_urls = retriever.valid_urls
    response_data = generate_response(intent, state, candidates, messages, valid_urls,
                                      retriever=retriever)

    return ChatResponse(
        reply=response_data["reply"],
        recommendations=response_data["recommendations"],
        end_of_conversation=response_data["end_of_conversation"],
    )


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    groq = os.getenv("GROQ_API_KEY", "")
    openai = os.getenv("OPENAI_API_KEY", "")
    provider = "Groq" if groq else ("OpenAI" if openai else "NONE — set GROQ_API_KEY")
    print(f"LLM provider  : {provider}")
    print(f"LLM model     : {os.getenv('LLM_MODEL', 'llama-3.1-8b-instant')}")
    print(f"Catalog size  : {len(retriever.catalog)} assessments")
    print(f"Catalog source: catalog.json (rich — entity_id, description, job_levels, etc.)")
