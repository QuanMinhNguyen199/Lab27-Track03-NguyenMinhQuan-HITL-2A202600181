"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from ollama import ResponseError

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─── Helpers ────────────────────────────────────────────────────────────────
async def _list_recent_threads(limit: int = 25):
    """Return a list of dicts — thread_id, pr_url, worst_risk, events, timestamp."""
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MIN(timestamp)        AS started,
                   MAX(timestamp)        AS last_event,
                   MAX(risk_level)       AS worst_risk,
                   COUNT(*)              AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT ?
            """,
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None
if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "error" not in st.session_state:
    st.session_state.error = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")

    # Display Ollama configuration
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
    with st.expander("🤖 Ollama Config", expanded=False):
        st.caption(f"**Base URL:** {ollama_base_url}")
        st.caption(f"**Model:** {ollama_model}")

    try:
        sessions = asyncio.run(_list_recent_threads(20))
        st.session_state.sessions = sessions
    except Exception:
        sessions = st.session_state.sessions

    if sessions:
        for s in sessions:
            risk_emoji = {"low": "🟢", "med": "🟡", "high": "🔴"}.get(
                s["worst_risk"], "⚪"
            )
            label = (
                f"{risk_emoji} {s['pr_url'][:50]}…  "
                f"({s['events']} events)"
            )
            if st.button(
                label,
                key=f"sidemenu_{s['thread_id']}",
                use_container_width=True,
            ):
                st.session_state.thread_id = s["thread_id"]
                st.session_state.pr_url = s["pr_url"]
                st.session_state.interrupt_payload = None
                st.session_state.final = None
                st.rerun()
    else:
        st.caption("No sessions yet. Start a new review above.")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL",
        value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(
            f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}`"
            f" — {c['body']}"
        )

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key="approval_feedback")

    col1, col2, col3 = st.columns(3)

    if col1.button("Approve", type="primary", key="btn_approve"):
        return {"choice": "approve", "feedback": feedback}

    if col2.button("Reject", key="btn_reject"):
        return {"choice": "reject", "feedback": feedback}

    if col3.button("Edit", key="btn_edit"):
        return {"choice": "edit", "feedback": feedback}

    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    questions = payload.get("questions", [])
    if not questions:
        questions = ["What is the intent of this PR?", "What tests would you add?"]

    with st.form("escalation"):
        answers: dict[str, str] = {}
        for i, q in enumerate(questions):
            answers[q] = st.text_input(
                q, key=f"escalation_q_{i}", placeholder="Your answer..."
            )
        submitted_answers = st.form_submit_button("Submit answers")
        if submitted_answers:
            # Filter out empty answers
            return {q: a for q, a in answers.items() if a.strip()} or answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(
    pr_url: str, thread_id: str, resume_value: dict | None = None
) -> dict:
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            result = await app.ainvoke(
                {"pr_url": pr_url, "thread_id": thread_id}, cfg
            )
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)

        return result


def format_graph_error(exc: Exception) -> str:
    """Turn common graph/LLM failures into actionable UI text."""
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    if isinstance(exc, ResponseError):
        return (
            "Ollama stopped the model while generating the review.\n\n"
            f"Model: `{model}`\n\n"
            f"Base URL: `{base_url}`\n\n"
            f"Details: `{exc}`\n\n"
            "Try restarting Ollama, then run `ollama ps` and `ollama run "
            f"{model} \"hello\"`. If it still closes the connection, switch "
            "to a smaller pulled model in `.env`."
        )

    if isinstance(exc, OutputParserException):
        return (
            "The model answered, but its structured JSON did not match "
            "`PRAnalysis`. I added tolerance for percent-style confidence "
            "values; if this still appears, rerun once or try a stronger "
            f"structured-output model than `{model}`.\n\nDetails: `{exc}`"
        )

    return f"{type(exc).__name__}: {exc}"


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.session_state.error = None

    with st.spinner("Fetching PR + asking the LLM..."):
        try:
            result = asyncio.run(
                run_graph(pr_url, st.session_state.thread_id)
            )
        except Exception as exc:
            st.session_state.error = format_graph_error(exc)
            result = None

    if result is not None:
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = (
        render_approval_card(payload)
        if kind == "approval_request"
        else render_escalation_card(payload)
    )
    if answer is not None:
        with st.spinner("Resuming..."):
            try:
                result = asyncio.run(
                    run_graph(
                        st.session_state.pr_url,
                        st.session_state.thread_id,
                        resume_value=answer,
                    )
                )
            except Exception as exc:
                st.session_state.error = format_graph_error(exc)
                result = None
        if result is not None:
            if "__interrupt__" in result:
                st.session_state.interrupt_payload = result["__interrupt__"][0].value
            else:
                st.session_state.interrupt_payload = None
                st.session_state.final = result
            st.rerun()

if st.session_state.error is not None:
    st.error(st.session_state.error)

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if action.startswith("auto") or action.startswith("committed"):
        st.success(
            f"✓ {action} — comment posted to {st.session_state.pr_url}"
        )
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    else:
        st.info(f"final_action = {action}")
    st.caption(
        f"thread_id = {st.session_state.thread_id}  ·  replay: "
        f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )
