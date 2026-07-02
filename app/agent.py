"""
Conversation agent — v3
Fixes:
  1. Richer clarification: asks role → seniority → competency focus (optional)
  2. REFINE reliably injects requested types via two-stage retrieval
  3. COMPARE fully tested and grounded in catalog data
  4. end_of_conversation:true when user signals they're done
"""

import json, re, os
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel
from openai import OpenAI

# ── LLM client ────────────────────────────────────────────────────────────────
_client: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        groq = os.environ.get("GROQ_API_KEY", "")
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if groq:
            _client = OpenAI(api_key=groq, base_url="https://api.groq.com/openai/v1")
        else:
            _client = OpenAI(api_key=openai_key)
    return _client

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# Load catalog once at import time so REFINE can search it directly
_catalog_all: List[Dict] = []
def _load_catalog():
    global _catalog_all
    try:
        with open("catalog.json") as f:
            _catalog_all = json.load(f)
    except Exception:
        _catalog_all = []
_load_catalog()

# ── Constants ─────────────────────────────────────────────────────────────────
KEY_NAMES = {
    "A": "Ability & Aptitude (cognitive reasoning)",
    "B": "Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360 Feedback",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills (technical/domain)",
    "M": "Motivational",
    "P": "Personality & Behaviour",
    "S": "Simulation",
}

TEST_TYPE_VOCAB = {
    "P": ["personality", "behaviour", "behavior", "opq", "occupational", "character"],
    "A": ["ability", "cognitive", "reasoning", "numerical", "verbal", "inductive",
          "verify", "aptitude", "spatial", "mechanical", "calculation", "deductive",
          "logical", "g+", "g plus"],
    "K": ["skill", "knowledge", "technical", "coding", "programming", "java",
          "python", "sql", "aws", ".net", "software", "developer", "domain"],
    "B": ["situational", "judgement", "judgment", "sjt", "scenarios", "managerial"],
    "S": ["simulation", "scenario", "exercise", "inbox", "coding simulation"],
    "M": ["motivation", "motivational", "mq", "driver", "engage"],
    "C": ["competency", "competencies", "ucf", "universal competency"],
    "D": ["360", "development", "feedback", "multi-rater", "mfs"],
}

DONE_PHRASES = [
    "thank", "thanks", "that's all", "that's great", "perfect", "looks good",
    "no more", "done", "all set", "that's enough", "great, thanks",
    "i'm good", "im good", "that works", "appreciate it", "helpful",
]


# ── Data models ───────────────────────────────────────────────────────────────
class ConversationState(BaseModel):
    role: Optional[str] = None
    seniority: Optional[str] = None
    skills: List[str] = []
    competency_focus: Optional[str] = None       # e.g. "stakeholder management"
    test_types_requested: List[str] = []
    remote_required: bool = False
    adaptive_required: bool = False
    duration_max_minutes: Optional[int] = None
    language: Optional[str] = None
    job_description: Optional[str] = None
    clarify_step: int = 0                         # tracks which question we've asked


class Intent(BaseModel):
    intent: Literal["CLARIFY", "RECOMMEND", "REFINE", "COMPARE", "REFUSE", "END"]
    reasoning: str
    question: Optional[str] = None
    compare_names: Optional[List[str]] = None


# ── LLM helpers ───────────────────────────────────────────────────────────────
def _call_llm(prompt: str, max_tokens: int = 500, temperature: float = 0) -> Optional[str]:
    try:
        resp = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


def _parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        text = re.sub(r'^```(?:json)?\s*', '', text.strip())
        text = re.sub(r'\s*```$', '', text.strip())
        return json.loads(text)
    except Exception:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


# ── Keyword-based state extractor (fallback) ─────────────────────────────────
def _keyword_extract_state(messages: List[Dict]) -> ConversationState:
    """Fast keyword-based state extractor (no LLM)."""
    full = " ".join(m["content"].lower() for m in messages if m["role"] == "user")
    state = ConversationState()

    # ── Role extraction ───────────────────────────────────────────────────────
    # Known job titles (single or multi-word) — match these directly first
    KNOWN_ROLES = [
        "java developer", "python developer", "software developer", "software engineer",
        "data scientist", "data analyst", "data engineer", "machine learning engineer",
        "sales manager", "sales representative", "sales engineer", "account manager",
        "project manager", "product manager", "marketing manager", "hr manager",
        "nurse", "doctor", "physician", "pharmacist", "dentist",
        "customer service representative", "customer service agent", "call center agent",
        "financial analyst", "accountant", "business analyst",
        "devops engineer", "cloud engineer", "network engineer", "security engineer",
        "frontend developer", "backend developer", "fullstack developer", "full stack developer",
        "mobile developer", "android developer", "ios developer",
        "qa engineer", "test engineer", "quality assurance",
        "recruiter", "hr specialist", "talent acquisition",
        "graphic designer", "ux designer", "ui designer",
        "operations manager", "supply chain manager", "logistics coordinator",
        "teacher", "trainer", "consultant", "analyst", "coordinator", "specialist",
        "developer", "engineer", "manager", "analyst", "designer", "architect",
        "lead", "director", "executive", "officer", "administrator",
    ]
    # Try to find a known role in the full conversation
    for role in KNOWN_ROLES:
        if role in full:
            # Prioritize longer/more specific matches
            if state.role is None or len(role) > len(state.role):
                state.role = role.title()

    # Fallback regex patterns for roles not in the list above
    if not state.role:
        role_patterns = [
            r"hir(?:ing|ed?)\s+(?:a|an|for)?\s*([\w][\w\s]{2,35})",
            r"(?:for|a|an)\s+([\w][\w\s]{2,25}(?:developer|engineer|manager|analyst|designer|scientist|nurse|doctor|specialist|coordinator))",
            r"(?:role|position|job)[:\s]+(?:a|an)?\s*([\w][\w\s]{2,35})",
            r"^([\w][\w\s]{2,30})$",  # single/short answer that IS the role
        ]
        for pat in role_patterns:
            m = re.search(pat, full)
            if m:
                candidate = m.group(1).strip().title()
                if len(candidate) >= 3 and len(candidate) <= 50:
                    state.role = candidate
                    break

    # ── Seniority ─────────────────────────────────────────────────────────────
    if any(k in full for k in ["entry level", "entry-level", "fresher", "junior",
                                "0-2 year", "1 year", "2 year", "fresh grad"]):
        state.seniority = "entry-level"
    elif any(k in full for k in ["senior", "sr.", "lead", "principal", "staff",
                                  "7 year", "8 year", "9 year", "10 year"]):
        state.seniority = "senior"
    elif any(k in full for k in ["mid", "intermediate", "3 year", "4 year",
                                  "5 year", "6 year", "mid-level"]):
        state.seniority = "mid-level"
    elif any(k in full for k in ["manager", "management", "director",
                                  "head of", "vp "]):
        state.seniority = "manager"
    elif any(k in full for k in ["executive", "c-suite", "ceo", "cto", "cfo"]):
        state.seniority = "executive"

    # ── Competency focus ──────────────────────────────────────────────────────
    comp_hints = [
        "stakeholder", "leadership", "communication", "teamwork", "negotiation",
        "client-facing", "presentation", "decision making", "problem solving",
    ]
    found_comp = [c for c in comp_hints if c in full]
    if found_comp:
        state.competency_focus = ", ".join(found_comp[:3])

    # ── Test types explicitly requested ───────────────────────────────────────
    for code, kws in TEST_TYPE_VOCAB.items():
        if any(k in full for k in kws):
            if code not in state.test_types_requested:
                state.test_types_requested.append(code)

    # If user said "cognitive ability assessments" → they want A type AND effectively
    # told us their preference, so treat it as enough to recommend
    if "cognitive" in full or "personality" in full or "simulation" in full:
        pass  # test_types_requested already captured above

    # ── Remote ────────────────────────────────────────────────────────────────
    if "remote" in full or "online" in full:
        state.remote_required = True

    # ── Technical skills ──────────────────────────────────────────────────────
    skills_kw = ["java", "python", "javascript", "react", "angular", "node", "sql",
                 "aws", "azure", ".net", "c#", "c++", "go", "kotlin",
                 "excel", "powerpoint", "sap", "salesforce", "data science",
                 "machine learning", "tableau", "r programming"]
    state.skills = [s for s in skills_kw if s in full]

    # ── Duration limit ────────────────────────────────────────────────────────
    dm = re.search(r'(?:under|less than|max(?:imum)?|within)\s*(\d+)\s*min', full)
    if dm:
        state.duration_max_minutes = int(dm.group(1))

    # ── Language ──────────────────────────────────────────────────────────────
    lm = re.search(r'(?:in|language[:\s]+)(french|german|spanish|chinese|japanese|arabic)', full)
    if lm:
        state.language = lm.group(1).title()

    # ── JD paste ──────────────────────────────────────────────────────────────
    for msg in messages:
        if msg["role"] == "user" and len(msg["content"]) > 300:
            state.job_description = msg["content"][:500]
            break

    return state


# ── Main state extractor ──────────────────────────────────────────────────────
def extract_state(messages: List[Dict]) -> ConversationState:
    history = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    prompt = f"""Extract hiring intent from this conversation. Return ONLY valid JSON.

Fields (all optional/nullable):
- role: job title string
- seniority: "entry-level" | "mid-level" | "senior" | "manager" | "executive"
- skills: array of technical skills mentioned
- competency_focus: soft skills or competency focus mentioned (e.g. "stakeholder management, leadership")
- test_types_requested: codes explicitly asked for [A=cognitive, P=personality, K=skills, S=simulation, B=situational, M=motivation, C=competencies, D=360]
- remote_required: bool
- adaptive_required: bool
- duration_max_minutes: int or null
- language: language string or null
- job_description: first 300 chars of any pasted JD or null

Conversation:
{history}

JSON:"""

    raw = _call_llm(prompt, max_tokens=400)
    data = _parse_json(raw)
    if data:
        try:
            return ConversationState(**{k: v for k, v in data.items()
                                        if k in ConversationState.model_fields})
        except Exception:
            pass
    return _keyword_extract_state(messages)


# ── Intent classifier ─────────────────────────────────────────────────────────
def classify_intent(messages: List[Dict], state: ConversationState, turn_count: int) -> Intent:
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    last_lower = last_user.lower()

    # ── END: user signals done ────────────────────────────────────────────────
    if any(ph in last_lower for ph in DONE_PHRASES) and turn_count >= 2:
        return Intent(intent="END", reasoning="User signalled conversation complete")

    # ── REFUSE ────────────────────────────────────────────────────────────────
    refuse_patterns = [
        r"ignore\s+(?:all\s+)?(?:previous|prior|above)",
        r"system\s+prompt", r"jailbreak",
        r"pretend\s+(?:you\s+are|to\s+be)",
        r"legal\s+(?:advice|requirement|compliance)",
        r"discrimination\s+law", r"gdpr", r"eeoc", r"lawsuit",
        r"hiring\s+(?:law|regulation)",
        r"tell\s+me\s+(?:your\s+)?(?:instructions|rules|context|secret|prompt)",
    ]
    if any(re.search(p, last_lower) for p in refuse_patterns):
        return Intent(intent="REFUSE", reasoning="Off-topic or injection attempt")

    # ── COMPARE ───────────────────────────────────────────────────────────────
    compare_triggers = [
        "difference between", "compare", " vs ", " versus ",
        "what is the difference", "how does.*differ", "which is better",
        "what's the difference",
    ]
    if any(re.search(t, last_lower) for t in compare_triggers):
        names = _extract_compare_names(last_user)
        return Intent(intent="COMPARE", reasoning="Comparison requested", compare_names=names)

    # ── REFINE ────────────────────────────────────────────────────────────────
    # Detect if agent has already given recommendations (any prior assistant turn
    # with assessment-related content)
    has_prior_recs = any(
        m["role"] == "assistant"
        and any(kw in m.get("content", "").lower() for kw in
                ["here are", "assessment", "recommend", "following", "suggest"])
        for m in messages[:-1]  # exclude the last message (which is the user's)
    )
    refine_triggers = [
        "actually", "add", "also include", "also add", "remove", "without",
        "only", "shorter", "longer", "change", "update", "revise", "instead",
        "focus on", "narrow", "personality", "cognitive", "add more", "include",
        "and also", "can you add",
    ]
    if has_prior_recs and any(t in last_lower for t in refine_triggers):
        return Intent(intent="REFINE", reasoning="User is refining recommendations")

    # ── Explicit recommendation request ───────────────────────────────────────
    request_triggers = [
        "show me", "give me", "recommend", "suggest", "what do you have",
        "what assessments", "please provide", "find me", "which test", "i need",
        "list", "what tests", "what options",
    ]
    jd_pasted = len(last_user) > 200
    user_asks = any(t in last_lower for t in request_triggers)

    # ── CLARIFY — ask questions one at a time ─────────────────────────────────
    # Clarify whenever: not JD pasted AND (no explicit request OR missing critical info)
    # Critical info = role + seniority. Always ask these before recommending.
    if not jd_pasted:

        # Step 1: Need a role
        if not state.role and not state.job_description:
            last_clean = last_user.strip().lower()
            short_answer = len(last_clean.split()) <= 6
            looks_like_role = (
                short_answer and turn_count >= 2 and
                not any(kw in last_clean for kw in
                        ["what", "how", "when", "why", "can you", "please"])
            )
            if looks_like_role:
                state.role = last_user.strip().title()
            else:
                return Intent(
                    intent="CLARIFY",
                    reasoning="Missing role",
                    question="What role are you hiring for? "
                             "(e.g. Java developer, sales manager, data scientist, nurse)"
                )

        # Step 2: Always ask seniority if not yet known (even if user_asks)
        if not state.seniority and not state.job_description:
            role_str = state.role or "this role"
            return Intent(
                intent="CLARIFY",
                reasoning="Missing seniority",
                question=f"What seniority level is the {role_str}? "
                         f"(entry-level, mid-level, senior, manager, or executive)"
            )

        # Step 3: Ask about competency/test type preference (optional, turns 2-4 only)
        step3_already_asked = any(
            m["role"] == "assistant" and any(kw in m.get("content","").lower() for kw in
                ["no preference", "soft skill", "specific test types", "stakeholder",
                 "cognitive ability, personality"])
            for m in messages
        )
        if (turn_count in (2, 3, 4)
                and not state.competency_focus
                and not state.test_types_requested
                and not has_prior_recs
                and not step3_already_asked
                and not user_asks):
            role_str = state.role or "the candidate"
            seniority_str = f"{state.seniority} " if state.seniority else ""
            return Intent(
                intent="CLARIFY",
                reasoning="Asking competency focus + preferred test types",
                question=(
                    f"A couple more details to find the best fit:\n"
                    f"1. Does the {seniority_str}{role_str} need to work with stakeholders, "
                    f"lead teams, or have specific soft skills (e.g. communication, negotiation)?\n"
                    f"2. Do you want any specific assessment types — "
                    f"cognitive ability, personality/behaviour, coding simulation, or technical skills? "
                    f"(Say 'no preference' for a balanced mix)"
                )
            )

    # ── RECOMMEND (default once we have enough) ───────────────────────────────
    # Need BOTH role AND seniority before recommending (unless JD pasted or explicit request)
    has_enough = bool(
        (state.role and state.seniority)   # role + seniority both known
        or state.job_description           # or JD pasted
        or len(state.test_types_requested) >= 1  # or explicit type requested
    )
    approaching_limit = turn_count >= 6

    if user_asks or jd_pasted or has_enough or approaching_limit:
        return Intent(intent="RECOMMEND", reasoning="Sufficient context")

    # Fallback: ask for role
    return Intent(
        intent="CLARIFY",
        reasoning="Not enough context",
        question="Could you describe the role you're hiring for?"
    )


def _extract_compare_names(text: str) -> List[str]:
    for pat in [
        r'between\s+"?(.+?)"?\s+and\s+"?(.+?)"?[\?.]?$',
        r'"?(.+?)"?\s+(?:vs\.?|versus)\s+"?(.+?)"?[\?.]?$',
    ]:
        m = re.search(pat, text, re.I)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
    return []


def _state_summary(state: ConversationState) -> str:
    parts = []
    if state.role:              parts.append(f"Role: {state.role}")
    if state.seniority:         parts.append(f"Level: {state.seniority}")
    if state.skills:            parts.append(f"Skills: {', '.join(state.skills[:4])}")
    if state.competency_focus:  parts.append(f"Competency focus: {state.competency_focus}")
    if state.duration_max_minutes: parts.append(f"Max duration: {state.duration_max_minutes} min")
    if state.language:          parts.append(f"Language: {state.language}")
    if state.test_types_requested:
        labels = [KEY_NAMES.get(t, t).split("(")[0].strip() for t in state.test_types_requested]
        parts.append(f"Types wanted: {', '.join(labels)}")
    return "; ".join(parts) if parts else "general assessment"


# ── Response generator ────────────────────────────────────────────────────────
def generate_response(
    intent: Intent,
    state: ConversationState,
    candidates: List[Dict],
    messages: List[Dict],
    catalog_urls: set,
    retriever=None,
) -> Dict:

    # END
    if intent.intent == "END":
        return {
            "reply": "Great! I hope the recommended assessments are helpful for your hiring. "
                     "Feel free to start a new conversation anytime.",
            "recommendations": [],
            "end_of_conversation": True,
        }

    # REFUSE
    if intent.intent == "REFUSE":
        return {
            "reply": "I can only help with SHL assessment recommendations. "
                     "For legal or compliance questions, please consult your HR or legal team.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # CLARIFY
    if intent.intent == "CLARIFY":
        return {
            "reply": intent.question or "Could you tell me more about the role you're hiring for?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # COMPARE
    if intent.intent == "COMPARE":
        return _handle_compare(intent, candidates, catalog_urls)

    # RECOMMEND / REFINE
    if not candidates:
        return {
            "reply": "I couldn't find assessments matching your criteria in the catalog. "
                     "Could you broaden the requirements or describe the role differently?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    return _handle_recommend(intent, state, candidates, messages, catalog_urls)


# Make retriever accessible to agent for REFINE fallback
_global_retriever = None

def set_retriever(r):
    global _global_retriever
    _global_retriever = r


def _handle_compare(intent: Intent, candidates: List[Dict], catalog_urls: set) -> Dict:
    names = intent.compare_names or []
    if len(names) < 2:
        return {
            "reply": "Which two assessments would you like to compare? "
                     "For example: 'Compare OPQ32r and Verify Numerical Ability'.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # Search all candidates (compare uses a broad pool)
    found = []
    for name in names[:2]:
        nl = name.lower()
        best, best_score = None, 0
        for c in candidates:
            cn = c["name"].lower()
            # exact match
            if nl == cn:
                best = c
                break
            # word overlap score
            score = sum(1 for w in nl.split() if len(w) > 2 and w in cn)
            if score > best_score:
                best_score, best = score, c
        if best:
            found.append(best)

    if len(found) < 2:
        return {
            "reply": f"I found {len(found)} of the 2 assessments you mentioned. "
                     "Please check the names — for example: 'OPQ32r' or 'Verify Numerical Ability'.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    a, b = found[0], found[1]

    prompt = f"""Compare these two SHL assessments using ONLY the data below. Be factual and concise (3-4 sentences).

Assessment 1: {a['name']}
- Type: {KEY_NAMES.get(a['test_type'], a['test_type'])}
- Description: {a.get('description', '')[:200]}
- Job Levels: {', '.join(a.get('job_levels', [])[:4]) or 'All levels'}
- Duration: {a.get('duration_display', '') or 'Not specified'}
- Adaptive: {'Yes' if a.get('adaptive') else 'No'}

Assessment 2: {b['name']}
- Type: {KEY_NAMES.get(b['test_type'], b['test_type'])}
- Description: {b.get('description', '')[:200]}
- Job Levels: {', '.join(b.get('job_levels', [])[:4]) or 'All levels'}
- Duration: {b.get('duration_display', '') or 'Not specified'}
- Adaptive: {'Yes' if b.get('adaptive') else 'No'}

Write a grounded comparison for a hiring manager:"""

    raw = _call_llm(prompt, max_tokens=250, temperature=0.2)
    reply = raw or (
        f"{a['name']} is a {KEY_NAMES.get(a['test_type'], '')} assessment "
        f"({a.get('duration_display', 'variable duration')}), while "
        f"{b['name']} is a {KEY_NAMES.get(b['test_type'], '')} assessment "
        f"({b.get('duration_display', 'variable duration')}). "
        f"Both support remote testing."
    )

    recs = [
        {"name": e["name"], "url": e["url"], "test_type": e["test_type"]}
        for e in [a, b]
        if e["url"] in catalog_urls
    ]
    return {"reply": reply, "recommendations": recs, "end_of_conversation": False}


def _handle_recommend(
    intent: Intent,
    state: ConversationState,
    candidates: List[Dict],
    messages: List[Dict],
    catalog_urls: set,
) -> Dict:
    history = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages[-6:])
    summary = _state_summary(state)

    cand_text = "\n".join(
        f"[{i}] {c['name']} | type={c['test_type']} ({KEY_NAMES.get(c['test_type'], '').split('(')[0].strip()})"
        f" | {c.get('duration_display', '?')}"
        f" | adaptive={'Y' if c.get('adaptive') else 'N'}"
        f" | levels={', '.join(c.get('job_levels', [])[:2]) or 'All'}"
        for i, c in enumerate(candidates[:20])
    )

    refine_note = (
        "IMPORTANT: The user is REFINING. Keep relevant previous recommendations "
        "AND add new ones matching the updated constraint. Do NOT replace everything."
        if intent.intent == "REFINE"
        else "Select across DIFFERENT test types for a balanced assessment battery when possible."
    )

    prompt = f"""You are an SHL assessment recommender. Select 5-10 assessments from CANDIDATES ONLY.

Hiring context: {summary}
{refine_note}

Recent conversation:
{history}

CANDIDATES (select by [index]):
{cand_text}

Rules:
- Select 5 to 10 assessments — show the user a proper range of options
- Mix test types (cognitive + personality + skills + simulation) for a balanced battery
- For a developer role: include Java/technical skills (K), cognitive ability (A), personality (P), and a coding simulation (S) if available
- For leadership/management: include personality (P), situational judgement (B), competencies (C)
- For entry-level: include job-focused solutions and cognitive screening
- Write 1-2 sentences explaining the selection
- ONLY use indices from the CANDIDATES list above

Return ONLY valid JSON: {{"reply": "explanation", "indices": [0, 1, 2, 3, 4, 5]}}

JSON:"""

    raw = _call_llm(prompt, max_tokens=350, temperature=0.2)
    data = _parse_json(raw)

    if data and "indices" in data:
        indices = [i for i in data["indices"] if isinstance(i, int) and 0 <= i < len(candidates)]
        selected = [candidates[i] for i in indices[:10]]
        reply = data.get("reply", f"Here are {len(selected)} recommended assessments.")
    else:
        selected = candidates[:5]
        reply = f"Based on your requirements ({summary}), here are {len(selected)} assessments:"

    # ── REFINE: deterministic type injection (guaranteed — LLM can miss this) ─
    # Parse the last user message for explicit type keywords
    last_user_msg = next((m["content"].lower() for m in reversed(messages)
                          if m["role"] == "user"), "")
    force_types: List[str] = []
    type_keywords = {
        "P": ["personality", "behaviour", "behavior", "opq"],
        "A": ["cognitive", "ability", "reasoning", "numerical", "verbal", "inductive"],
        "K": ["skills", "technical", "coding", "knowledge"],
        "S": ["simulation", "exercise"],
        "B": ["situational", "scenarios"],
        "M": ["motivation", "motivational"],
        "C": ["competenc"],
        "D": ["360", "feedback"],
    }
    for ttype, kws in type_keywords.items():
        if any(kw in last_user_msg for kw in kws):
            force_types.append(ttype)

    if intent.intent == "REFINE" and force_types:
        existing_types = {c["test_type"] for c in selected}
        selected_urls = {c["url"] for c in selected}
        # Make room for injections — remove last entries of over-represented types
        for ttype in force_types:
            if ttype not in existing_types:
                # Remove the last entry that is NOT in force_types to make room
                for i in range(len(selected) - 1, -1, -1):
                    if selected[i]["test_type"] not in force_types and len(selected) >= 10:
                        selected_urls.discard(selected[i]["url"])
                        selected.pop(i)
                        break
                # Now inject from candidates first, then full catalog
                injected = False
                for c in candidates:
                    if c["test_type"] == ttype and c["url"] not in selected_urls:
                        selected.append(c)
                        selected_urls.add(c["url"])
                        existing_types.add(ttype)
                        injected = True
                        break
                if not injected and _catalog_all:
                    for c in _catalog_all:
                        if c["test_type"] == ttype and c["url"] not in selected_urls:
                            selected.append(c)
                            selected_urls.add(c["url"])
                            existing_types.add(ttype)
                            break
        type_names = [KEY_NAMES.get(t, t).split("(")[0].strip() for t in force_types]
        reply = (f"Updated shortlist — added {', '.join(type_names)} assessments "
                 f"alongside the previous recommendations.")
        selected = selected[:10]

    # ── RECOMMEND: guarantee type diversity — inject A and P if missing ───────
    if intent.intent == "RECOMMEND" and not state.test_types_requested:
        existing_types = {c["test_type"] for c in selected}
        selected_urls = {c["url"] for c in selected}
        for inject_type in ["A", "P"]:
            if inject_type not in existing_types and len(selected) < 8:
                for c in candidates:
                    if c["test_type"] == inject_type and c["url"] not in selected_urls:
                        selected.append(c)
                        selected_urls.add(c["url"])
                        break

    recs = [
        {"name": c["name"], "url": c["url"], "test_type": c["test_type"]}
        for c in selected
        if c["url"] in catalog_urls
    ][:10]

    return {"reply": reply, "recommendations": recs, "end_of_conversation": False}
