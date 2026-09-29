# ==========================================
# 1. Warning suppression (Production Mode)
# Scoped to the noisy, known-safe categories only — blanket-suppressing
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

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# ==========================================
# 2. Wire up the agents package
# ==========================================
_AI_AGENTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "ai_agents"))
if _AI_AGENTS_PATH not in sys.path:
    sys.path.append(_AI_AGENTS_PATH)

# Import from `router_graph` (the pure graph module) rather than `router`
# (the CLI entrypoint). `router.py` additionally imports Rich and the
# Arabic-shaping libraries, which this web UI doesn't need — Streamlit is
# its own presentation layer. This also avoids paying for/needing those
# optional CLI dependencies in a web deployment.
try:
    from src.agent.router_graph import build_graph, sql_analyst_agent, etl_analyst_agent, analyst_agent, logger
    from src.agent.router_config import ROUTE_LABELS, GROQ_API_KEY
    IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 - surfaced to the user as a friendly startup error below
    IMPORT_ERROR = e
    analyst_agent = None
    logger = logging.getLogger("streamlit_app")

MAX_INPUT_CHARS = 4000       # guard against pasting huge blobs of text into the chat
MAX_HISTORY_MESSAGES = 60    # cap in-memory chat history so a very long session doesn't bloat the page

NODE_STATUS_LABELS = {
    "router": "🧭 تحليل السؤال وتحديد الوكيل المناسب...",
    "analysis": "📊 يتم تحليل السؤال بعمق...",
    "sql": "🗄️ يتم الاستعلام من قاعدة البيانات...",
    "etl": "🌐 يتم تنفيذ عملية الاستخراج / المعالجة...",
    "general": "🧠 يتم صياغة الرد...",
    "polish": "✨ يتم صقل الإجابة النهائية...",
}

# ==========================================
# 3. Page configuration
# ==========================================
st.set_page_config(
    page_title="Data Intelligence Agent | AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stChatFloatingInputContainer { padding-bottom: 20px; }
    .stChatMessage { border-radius: 10px; padding: 10px; margin-bottom: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 4. Startup checks — fail loudly but *nicely*, not with a raw traceback
# ==========================================
if IMPORT_ERROR is not None:
    st.error(
        "⚠️ تعذّر تحميل نظام الوكلاء (Agent System) عند بدء التشغيل.\n\n"
        f"تفاصيل تقنية: `{IMPORT_ERROR}`\n\n"
        "تأكد إن مجلد `ai_agents` وملفاته موجودين، وإن كل الباكدجات المطلوبة متثبتة."
    )
    st.stop()

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY غير موجود في متغيرات البيئة (.env). ضيفه وأعد تشغيل التطبيق.")
    st.stop()


@st.cache_resource
def load_agent_system():
    return build_graph()


try:
    app_graph = load_agent_system()
except Exception as e:
    logger.exception("Failed to build the agent graph")
    st.error(f"⚠️ حصل خطأ أثناء تجهيز نظام الوكلاء: {e}")
    st.stop()

# ==========================================
# 5. Session state
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(
            content=(
                "أهلاً بك يا هندسة! أنا المساعد الذكي الخاص بقاعدة بيانات Zomato. "
                "تقدر تسألني أي سؤال عن الداتا، أو تطلب مني أسحب بيانات من روابط خارجية. "
                "إزاي أقدر أساعدك النهاردة؟ 🚀"
            )
        )
    ]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"streamlit_session_{int(time.time())}"
if "turn_meta" not in st.session_state:
    # per-assistant-message metadata (route/confidence), keyed by chat_history index
    st.session_state.turn_meta = {}

graph_config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ==========================================
# 6. Sidebar — reflects REAL sub-agent availability instead of a hardcoded "Online"
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2042/2042885.png", width=100)
    st.title("⚙️ System Status")

    if analyst_agent is not None:
        st.success("✅ Data Analyst: Online")
    else:
        st.error("❌ Data Analyst: Unavailable")

    if sql_analyst_agent is not None:
        st.success("✅ SQL Agent: Online")
    else:
        st.warning("⚠️ SQL Agent: Unavailable (legacy)")

    if etl_analyst_agent is not None:
        st.success("✅ ETL Agent: Online")
    else:
        st.error("❌ ETL Agent: Unavailable")

    st.success("✅ Master Router: Active")

    st.markdown("---")
    st.markdown("### 💡 أمثلة للأسئلة:")
    st.info("📊 Show me revenue by month")
    st.info("📈 What is the average order value by city?")
    st.info("🔍 Why did sales decline? Which restaurant contributed most?")
    st.info("🌐 Extract data from an API URL")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ مسح المحادثة", use_container_width=True):
            st.session_state.chat_history = [st.session_state.chat_history[0]]
            st.session_state.turn_meta = {}
            st.session_state.thread_id = f"streamlit_session_{int(time.time())}"
            st.rerun()
    with col2:
        transcript = "\n\n".join(
            f"**{'أنت' if isinstance(m, HumanMessage) else 'المساعد'}:** {m.content}"
            for m in st.session_state.chat_history
        )
        st.download_button(
            "⬇️ تصدير", transcript, file_name="conversation.md", mime="text/markdown", use_container_width=True
        )

# ==========================================
# 7. Chat interface
# ==========================================
st.title("📊 Zomato AI Data Analyst")
st.markdown("Powered by LangGraph, Databricks & Agentic AI")

for i, msg in enumerate(st.session_state.chat_history):
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg.content)
            meta = st.session_state.turn_meta.get(i)
            if meta:
                if meta.get("viz_html"):
                    st.components.v1.html(meta["viz_html"], height=450, scrolling=True)
                
                emoji, name, _ = ROUTE_LABELS.get(meta.get("route", ""), ("🧭", meta.get("route", ""), "bold"))
                caption_text = f"{emoji} {name} · confidence {meta.get('confidence', 0.0):.2f}"
                
                llm_calls = meta.get("llm_call_count")
                if llm_calls:
                    latencies = meta.get("stage_latencies", {})
                    total_time = sum(latencies.values()) if latencies else 0
                    caption_text += f" | ⏱️ {total_time:.1f}s | 🧠 {llm_calls} LLM Calls"
                
                st.caption(caption_text)
                
                if meta.get("stage_latencies"):
                    with st.expander("Performance Stats (Latencies)"):
                        st.json(meta["stage_latencies"])

# ==========================================
# 8. Handle new user input
# ==========================================
user_query = st.chat_input("اكتب سؤالك هنا...")

if user_query:
    user_query = user_query.strip()

if user_query:
    if len(user_query) > MAX_INPUT_CHARS:
        st.warning(f"⚠️ الرسالة طويلة جدًا ({len(user_query)} حرف). الحد الأقصى {MAX_INPUT_CHARS} حرف — اختصرها وحاول تاني.")
    else:
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("assistant", avatar="🤖"):
            status_box = st.status("🧭 المدير بيحلل سؤالك...", expanded=False)
            final_answer = "عذراً، لم أتمكن من توليد إجابة."
            route_meta = None
            try:
                for chunk in app_graph.stream({"messages": [HumanMessage(content=user_query)]}, config=graph_config):
                    node_name = next(iter(chunk.keys()), None)
                    if node_name in NODE_STATUS_LABELS:
                        status_box.update(label=NODE_STATUS_LABELS[node_name])

                snapshot = app_graph.get_state(graph_config).values
                final_answer = snapshot.get("final_answer", final_answer)
                if snapshot.get("route"):
                    route_meta = {"route": snapshot["route"], "confidence": snapshot.get("confidence", 0.0)}

                status_box.update(label="✅ تم", state="complete")
                st.markdown(final_answer)

                # Render visualization if the analyst pipeline generated one
                viz_html = snapshot.get("viz_html", "")
                if viz_html:
                    st.components.v1.html(viz_html, height=450, scrolling=True)

                if route_meta:
                    emoji, name, _ = ROUTE_LABELS.get(route_meta["route"], ("🧭", route_meta["route"], "bold"))
                    caption_text = f"{emoji} {name} · confidence {route_meta['confidence']:.2f}"
                    
                    llm_calls = snapshot.get("llm_call_count")
                    if llm_calls:
                        latencies = snapshot.get("stage_latencies", {})
                        total_time = sum(latencies.values()) if latencies else 0
                        caption_text += f" | ⏱️ {total_time:.1f}s | 🧠 {llm_calls} LLM Calls"
                    
                    st.caption(caption_text)
                    
                    if snapshot.get("stage_latencies"):
                        with st.expander("Performance Stats (Latencies)"):
                            st.json(snapshot.get("stage_latencies"))

            except Exception as e:
                # Log the full exception server-side; show the user a short,
                # non-leaky message instead of raw internals (connection
                # strings, stack traces, etc. can end up in exception text).
                logger.exception("Unhandled error while answering a Streamlit chat turn")
                status_box.update(label="❌ حصل خطأ", state="error")
                final_answer = "❌ حصل خطأ غير متوقع أثناء معالجة سؤالك. جرّب تاني، ولو المشكلة استمرت كلّم فريق الدعم."
                st.error(final_answer)

        st.session_state.chat_history.append(AIMessage(content=final_answer))
        
        # Save meta, visualization and stats to survive Streamlit reruns
        meta_to_save = route_meta or {}
        if snapshot:
            meta_to_save["viz_html"] = snapshot.get("viz_html", "")
            meta_to_save["llm_call_count"] = snapshot.get("llm_call_count", 0)
            meta_to_save["stage_latencies"] = snapshot.get("stage_latencies", {})
            
        st.session_state.turn_meta[len(st.session_state.chat_history) - 1] = meta_to_save

        # Keep the in-memory history bounded so a very long-running session
        # doesn't grow the page/session state indefinitely.
        if len(st.session_state.chat_history) > MAX_HISTORY_MESSAGES:
            overflow = len(st.session_state.chat_history) - MAX_HISTORY_MESSAGES
            st.session_state.chat_history = [st.session_state.chat_history[0]] + st.session_state.chat_history[1 + overflow:]
            st.session_state.turn_meta = {
                k - overflow: v for k, v in st.session_state.turn_meta.items() if k - overflow >= 0
            }
