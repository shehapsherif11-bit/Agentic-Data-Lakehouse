# ==========================================
# 1. Warning suppression (Production Mode)
# Scoped to the noisy, known-safe categories only - blanket-suppressing
# every UserWarning/DeprecationWarning in the whole process can also hide
# a real deprecation notice from LangChain/Streamlit that you'd actually
# want to see before it becomes a breaking upgrade.
# ==========================================
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain.*")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain.*")

import logging
import os
import sys
import time

import pandas as pd
import plotly.express as px
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from dotenv import load_dotenv

load_dotenv()

# Public live demo: runs on a frozen sample (DuckDB) instead of the Databricks warehouse.
# Without Databricks credentials (a public deployment) the app always uses the sample, so a visitor never
# reaches a warehouse that is not there. Must run before the agent imports below, which read DEMO_MODE.
if "DEMO_MODE" not in os.environ and not all(
    os.getenv(v) for v in ("DATABRICKS_HOST", "DATABRICKS_HTTP_PATH", "DATABRICKS_TOKEN")
):
    os.environ["DEMO_MODE"] = "1"
DEMO_MODE =os.getenv("DEMO_MODE", "").lower() in ("1", "true", "yes")
DEMO_MAX_QUESTIONS = int(os.getenv("DEMO_MAX_QUESTIONS", "20"))  # chat questions per visitor session
REPO_URL = "https://github.com/shehapsherif11-bit/Agentic-Data-Lakehouse"

# ==========================================
# 2. Wire up the agents package
# ==========================================
_AI_AGENTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "ai_agents"))
if _AI_AGENTS_PATH not in sys.path:
    sys.path.append(_AI_AGENTS_PATH)

# Import from `router_graph` (the pure graph module) rather than `router`
# (the CLI entrypoint). `router.py` additionally imports Rich and the
# Arabic-shaping libraries, which this web UI doesn't need - Streamlit is
# its own presentation layer. This also avoids paying for/needing those
# optional CLI dependencies in a web deployment.
try:
    from src.agent.router_graph import build_graph, logger, warm_up_db
    from src.agent.router_config import ROUTE_LABELS, GROQ_API_KEY
    from src.agent.analyst_graph import execute_guarded
    from src.agent.metrics_registry import COLUMN_CATALOG, TABLE_GRAINS
    from src.agent import quick_reports
    IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 - surfaced to the user as a friendly startup error below
    IMPORT_ERROR = e
    logger = logging.getLogger("streamlit_app")

MAX_INPUT_CHARS = 4000       # guard against pasting huge blobs of text into the chat
MAX_HISTORY_MESSAGES = 60    # cap in-memory chat history so a very long session doesn't bloat the page

NODE_STATUS_LABELS = {
    "router": "🧭 Reading your question and choosing the right path...",
    "analysis": "📊 Analysing the data and checking every number...",
    "general": "🧠 Writing the answer...",
    "advisor": "💡 Comparing the results to give a recommendation...",
    "followup": "🩺 Diagnosing the problem and comparing with peers...",
}

VIEW_ASK = "💬 Ask"
VIEW_REPORTS = "📊 Quick reports"

# One-click questions. Each one is a real route of the system (ranking fast path, period change,
# derived measure, honest refusal, Arabic).
EXAMPLES = [
    ("🏆", "Top 9 restaurants by orders"),
    ("📉", "Which week had the biggest drop in revenue?"),
    ("🥧", "Top 10 restaurants with their percentage contribution to overall revenue"),
    ("💰", "Which restaurant makes the most profit?"),
    ("🇪🇬", "أفضل 5 مدن من حيث الإيراد"),
]

WELCOME = (
    "Hi! I'm the AI analyst for the Gold layer of this data lakehouse. Ask me anything about orders, revenue, restaurants, "
    "cities or customers, in English or Arabic. Every number I give you comes from an executed query, "
    "and I'll tell you when the data can't answer.\n\n"
    "أهلاً! اسألني أي سؤال عن الطلبات والإيرادات والمطاعم والمدن، بالعربي أو بالإنجليزي."
)

# ==========================================
# 3. Page configuration
# ==========================================
st.set_page_config(
    page_title="Lakehouse AI Analyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stChatFloatingInputContainer { padding-bottom: 20px; }
    .stChatMessage { border-radius: 10px; padding: 10px; margin-bottom: 10px; }
    h1 { padding-bottom: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 4. Startup checks - fail loudly but *nicely*, not with a raw traceback
# ==========================================
if IMPORT_ERROR is not None:
    st.error(
        "⚠️ The agent system could not be loaded at startup.\n\n"
        f"Technical details: `{IMPORT_ERROR}`\n\n"
        "Check that every required package is installed."
    )
    st.stop()

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY is missing from the environment. Add it and restart the app.")
    st.stop()


@st.cache_resource
def load_agent_system():
    graph = build_graph()
    try:
        # Background: wake the warehouse (or load the demo sample), open the first connection and load the
        # Gold schema now, so the first user question doesn't pay the cold start.
        warm_up_db(background=True)
    except Exception as e:  # noqa: BLE001 - warm-up is an optimisation only
        logger.warning("DB warm-up not started: %s", e)
    return graph


try:
    app_graph = load_agent_system()
except Exception as e:
    logger.exception("Failed to build the agent graph")
    st.error(f"⚠️ Something went wrong while preparing the agent system: {e}")
    st.stop()


def demo_banner_text() -> str:
    """What the demo data is, taken from the sample's manifest, so nobody mistakes it for the full dataset."""
    base = "**Live demo on a data sample.** Totals, rankings and customers here describe the sample, not the whole platform."
    try:
        from src.utils.demo_database import DemoDuckDBUtil
        m = DemoDuckDBUtil().manifest()
        if m.get("orders"):
            featured = len(m.get("featured_brands", []))
            return (
                f"**Live demo on a data sample:** {m['orders']:,} orders from {m['brands']:,} restaurant brands "
                f"({featured} biggest brands in full, plus a random {m['sample_pct']}% of the rest), "
                f"{m['date_from']} to {m['date_to']}. The full pipeline holds {m['full_orders']:,} orders on Databricks. "
                "Brand-level numbers for the biggest brands are exact; city totals, customers and averages describe the sample."
            )
    except Exception:  # noqa: BLE001 - the banner is informational only
        pass
    return base


def schema_markdown() -> str:
    """The tables and columns the agent knows about (its own catalog, so what you read is what it uses)."""
    lines = []
    for table, cols in COLUMN_CATALOG.items():
        grain = TABLE_GRAINS.get(table, {}).get("grain", "").split(" (")[0].split(". ")[0]
        lines.append(f"**{table.split('.')[-1]}** · {grain}")
        lines.append(", ".join(f"`{c}`" for c in cols))
        lines.append("")
    return "\n\n".join(lines)


# ==========================================
# 5. Session state
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content=WELCOME)]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"streamlit_session_{int(time.time())}"
if "turn_meta" not in st.session_state:
    # per-assistant-message metadata (route/confidence), keyed by chat_history index
    st.session_state.turn_meta = {}
if "questions_used" not in st.session_state:
    st.session_state.questions_used = 0
if "view" not in st.session_state:
    st.session_state.view = VIEW_ASK

graph_config = {"configurable": {"thread_id": st.session_state.thread_id}}


def ask_example(question: str):
    """Button callback: queue the question and switch to the chat view."""
    st.session_state.queued_question = question
    st.session_state.view = VIEW_ASK


def clear_chat():
    st.session_state.chat_history = [st.session_state.chat_history[0]]
    st.session_state.turn_meta = {}
    st.session_state.thread_id = f"streamlit_session_{int(time.time())}"


# ==========================================
# 6. Sidebar
# ==========================================
with st.sidebar:
    st.title("🧠 Data Analyst")
    st.caption("LangGraph agent over a Databricks + dbt lakehouse")

    st.markdown("#### Try asking")
    for icon, question in EXAMPLES:
        st.button(f"{icon}  {question}", key=f"ex_{question}", on_click=ask_example, args=(question,))

    st.markdown("#### How it stays safe")
    st.markdown(
        "1. Every query is parsed by an **AST guard**: one `SELECT`, Gold schema only, no writes.\n"
        "2. File, HTTP and AI functions are blocked; results are capped at 10,000 rows.\n"
        "3. Numbers in an answer are **checked against the executed results**.\n"
        "4. If the data can't answer (for example profit: there is no cost data), it says so instead of guessing."
    )

    with st.expander("🗂️ Database schema"):
        st.markdown(schema_markdown())

    col1, col2 = st.columns(2)
    with col1:
        st.button("🗑️ Clear chat", on_click=clear_chat)
    with col2:
        transcript = "\n\n".join(
            f"**{'You' if isinstance(m, HumanMessage) else 'Assistant'}:** {m.content}"
            for m in st.session_state.chat_history
        )
        st.download_button("⬇️ Export", transcript, file_name="conversation.md", mime="text/markdown")

    quota_box = st.empty()
    st.caption(f"[Source code on GitHub]({REPO_URL})")


def update_quota():
    if DEMO_MODE:
        left = max(0, DEMO_MAX_QUESTIONS - st.session_state.questions_used)
        quota_box.caption(f"Demo questions left in this session: {left} of {DEMO_MAX_QUESTIONS}")


update_quota()

# ==========================================
# 7. Header
# ==========================================
st.title("📊 Lakehouse AI Analyst")
st.caption("Ask in plain English or Arabic. See the SQL, the data and the chart behind every answer.")
if DEMO_MODE:
    st.info(demo_banner_text())

st.radio("View", [VIEW_ASK, VIEW_REPORTS], key="view", horizontal=True, label_visibility="collapsed")


# ==========================================
# 8. Quick reports: fixed SQL, no LLM, same guard and cache as chat
# ==========================================
def to_frame(rows, columns=None) -> pd.DataFrame:
    df = pd.DataFrame(list(rows), columns=list(columns) if columns else None)
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m")
        else:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass
    return df


def report_chart(df: pd.DataFrame, report):
    if report.kind == "line":
        fig = px.line(df, x=report.x, y=report.y, markers=True)
        fig.update_yaxes(rangemode="tozero")
    elif report.kind == "barh":
        fig = px.bar(df, x=report.y, y=report.x, orientation="h")
        fig.update_yaxes(autorange="reversed")
    else:
        fig = px.bar(df, x=report.x, y=report.y)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=380)
    return fig


def render_reports():
    st.subheader("Quick reports")
    st.caption("Fixed SQL, no LLM. Every report passes the same safety guard and query cache as a chat question.")
    by_id = quick_reports.BY_ID
    choice = st.selectbox("Report", list(by_id), format_func=lambda i: by_id[i].title)
    report = by_id[choice]
    st.caption(report.description)

    started = time.time()
    with st.spinner("Running the query..."):
        res = quick_reports.run_report(report, execute_guarded)
    elapsed = time.time() - started

    if not res.get("success"):
        st.error(f"The report could not run: {res.get('error') or 'unknown error'}")
        return
    df = to_frame([[r.get(c) for c in res["columns"]] for r in res["rows"]], res["columns"])
    if df.empty:
        st.info("The query ran but returned no rows.")
        return
    st.caption(f"{len(df):,} rows · {elapsed:.2f}s")
    st.plotly_chart(report_chart(df, report))
    st.dataframe(df, hide_index=True)
    with st.expander("🔍 SQL behind this report"):
        st.code(res["sql"].strip(), language="sql")


if st.session_state.view == VIEW_REPORTS:
    render_reports()
    st.stop()


# ==========================================
# 9. Chat interface
# ==========================================
def render_extras(meta: dict):
    """Chart, caption, evidence and performance panels under an assistant answer."""
    if not meta:
        return
    if meta.get("viz_html"):
        st.components.v1.html(meta["viz_html"], height=450, scrolling=True)

    if meta.get("route"):
        emoji, name, _ = ROUTE_LABELS.get(meta["route"], ("🧭", meta["route"], "bold"))
        parts = [f"{emoji} {name}", f"route confidence {meta.get('confidence', 0.0):.2f}"]
        latencies = meta.get("stage_latencies") or {}
        if latencies:
            parts.append(f"⏱️ {sum(latencies.values()):.1f}s")
            parts.append(f"🧠 {meta.get('llm_call_count') or 0} LLM calls")
        st.caption(" · ".join(parts))

    evidence = [e for e in (meta.get("evidence") or []) if hasattr(e, "executed_sql")]
    if evidence:
        with st.expander(f"🔍 How I got this ({len(evidence)} {'query' if len(evidence) == 1 else 'queries'})"):
            for e in evidence:
                st.code(e.executed_sql.strip(), language="sql")
                st.caption(f"{e.row_count:,} rows · executed {e.executed_at}")
                if e.truncated:
                    st.warning("⚠️ The results were truncated because they exceeded the maximum allowed rows.")
                if e.rows:
                    st.dataframe(to_frame(e.rows[:50], e.columns), hide_index=True)
                st.divider()

    if meta.get("stage_latencies") or meta.get("llm_telemetry"):
        with st.expander("Performance stats (latencies and LLM telemetry)"):
            if meta.get("stage_latencies"):
                st.markdown("**Node latencies (s):**")
                st.json(meta["stage_latencies"])
            if meta.get("llm_telemetry"):
                st.markdown("**LLM call telemetry:**")
                st.dataframe(meta["llm_telemetry"])


for i, msg in enumerate(st.session_state.chat_history):
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg.content)
            render_extras(st.session_state.turn_meta.get(i))

# ==========================================
# 10. Handle new user input
# ==========================================
typed_query = st.chat_input("Ask a question about the data...")
queued_query = st.session_state.pop("queued_question", None)
user_query = (queued_query or typed_query or "").strip()

if user_query:
    if len(user_query) > MAX_INPUT_CHARS:
        st.warning(f"⚠️ That message is too long ({len(user_query)} characters). The limit is {MAX_INPUT_CHARS}. Please shorten it.")
    elif DEMO_MODE and st.session_state.questions_used >= DEMO_MAX_QUESTIONS:
        st.warning(
            f"You've used all {DEMO_MAX_QUESTIONS} demo questions for this session. "
            f"Quick reports are still available, and the full project is on [GitHub]({REPO_URL})."
        )
    else:
        st.session_state.questions_used += 1
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("assistant", avatar="🤖"):
            status_box = st.status("🧭 Reading your question...", expanded=False)
            final_answer = "Sorry, I could not produce an answer."
            meta = {}
            try:
                for chunk in app_graph.stream({"messages": [HumanMessage(content=user_query)]}, config=graph_config):
                    node_name = next(iter(chunk.keys()), None)
                    if node_name in NODE_STATUS_LABELS:
                        status_box.update(label=NODE_STATUS_LABELS[node_name])

                snapshot = app_graph.get_state(graph_config).values
                final_answer = snapshot.get("final_answer", final_answer)
                if snapshot.get("route"):
                    meta = {"route": snapshot["route"], "confidence": snapshot.get("confidence", 0.0)}
                    # Evidence persists across turns in the checkpoint: only show its provenance on turns
                    # that actually rely on it (a GENERAL chat turn must not display an old SQL query).
                    if snapshot.get("turn_evidence"):
                        meta["evidence"] = snapshot["turn_evidence"]
                meta["viz_html"] = snapshot.get("viz_html", "")
                meta["llm_call_count"] = snapshot.get("llm_call_count", 0)
                meta["stage_latencies"] = snapshot.get("stage_latencies", {})
                meta["llm_telemetry"] = snapshot.get("llm_telemetry", [])

                status_box.update(label="✅ Done", state="complete")
                st.markdown(final_answer)
                render_extras(meta)

            except Exception:
                # Log the full exception server-side; show the user a short,
                # non-leaky message instead of raw internals (connection
                # strings, stack traces, etc. can end up in exception text).
                logger.exception("Unhandled error while answering a Streamlit chat turn")
                status_box.update(label="❌ Something went wrong", state="error")
                final_answer = "❌ Something unexpected went wrong while answering. Please try again."
                meta = {}
                st.error(final_answer)

        st.session_state.chat_history.append(AIMessage(content=final_answer))
        # Save meta, visualization and stats to survive Streamlit reruns
        st.session_state.turn_meta[len(st.session_state.chat_history) - 1] = meta
        update_quota()

        # Keep the in-memory history bounded so a very long-running session
        # doesn't grow the page/session state indefinitely.
        if len(st.session_state.chat_history) > MAX_HISTORY_MESSAGES:
            overflow = len(st.session_state.chat_history) - MAX_HISTORY_MESSAGES
            st.session_state.chat_history = [st.session_state.chat_history[0]] + st.session_state.chat_history[1 + overflow:]
            st.session_state.turn_meta = {
                k - overflow: v for k, v in st.session_state.turn_meta.items() if k - overflow >= 0
            }
