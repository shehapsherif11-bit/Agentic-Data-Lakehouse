import os
import requests
from typing import Literal
from dotenv import load_dotenv
from src.agent.llm_factory import get_llm
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

# ==========================================
# التعديل هنا: تحديث مسارات الاستيراد للمطعم الجديد 🍽️
# ==========================================
from src.agent.state import SQLAgentState
from src.utils.database import DatabricksUtil

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ==========================================
# 0. فحص سيرفرات Groq لجلب موديل يعمل بنجاح
# ==========================================
def get_working_model():
    print("Fetching and testing your allowed models from Groq... 🔄")
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            models = res.json().get("data", [])
            for m in models:
                model_name = m["id"]
                m_lower = model_name.lower()

                # استبعاد الموديلات الضعيفة، الصوت، الحماية، والصور
                if any(bad in m_lower for bad in ["guard", "whisper", "vision", "llava", "tts", "prompt-guard"]):
                    continue
                # التأكد إنه موديل لغوي قوي (لائحة الموديلات النصية الحالية على Groq)
                if not any(good in m_lower for good in ["llama", "gemma", "mixtral", "gpt-oss", "qwen", "compound", "kimi", "deepseek"]):
                    continue

                print(f"Testing model: {model_name}...")
                try:
                    # اختبار فعلي للموديل
                    test_llm = get_llm(model_name=model_name, temperature=0.0, max_retries=1)
                    test_llm.invoke("hi")
                    print(f"✅ BINGO! Verified & Working Model: {model_name}")
                    return model_name
                except Exception:
                    print(f"❌ Failed: {model_name}")
                    continue
    except Exception as e:
        print(f"Error fetching models: {e}")

    # فولباك محدث - آخر الموديلات النصية الشغالة فعليًا على Groq (سبتمبر 2026)
    fallback = "openai/gpt-oss-20b"
    print(f"⚠️ Warning: Falling back to {fallback}")
    return fallback

ACTIVE_MODEL = get_working_model()

# ==========================================
# 1. إعدادات النماذج
# ==========================================
llm = get_llm(
    model_name=ACTIVE_MODEL,
    temperature=0.0
)

class JudgeSchema(BaseModel):
    answer: Literal["yes", "no"] = Field(description="Is the SQL query safe?")
    comments: str = Field(description="Reasoning")

try:
    judge_llm = llm.with_structured_output(JudgeSchema)
except Exception:
    judge_llm = llm

# ==========================================
# 2. بناء الـ Nodes
# ==========================================
db_util = DatabricksUtil()

def curate_question(state: SQLAgentState) -> dict:
    try:
        last_msg = state.messages[-1]
        user_msg = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        prompt = f"Curate this question to make it clear for SQL generation. Just return the clear question.\nUser Question: {user_msg}"
        response = llm.invoke(prompt)
        return {"curated_question": response.content}
    except Exception:
        return {"curated_question": str(state.messages[-1].content)}

def prompt_query_context(state: SQLAgentState) -> dict:
    schema_details = db_util.get_schema_details()
    system_prompt = f"""You are a senior SQL analyst. Write a Databricks SQL query to answer the user's question.
    RULES:
    1. Only use SELECT queries. NEVER use INSERT, UPDATE, DELETE, or DROP.
    2. Use the exact table names and column names provided in the Schema Details.
    3. Limit the results to 10 rows using LIMIT 10.
    4. Return ONLY the raw SQL query. No markdown, no backticks (```sql).

    {schema_details}
    """
    return {"prompt_query_context": system_prompt}

def generate_sql(state: SQLAgentState) -> dict:
    messages = [
        SystemMessage(content=state.prompt_query_context),
        HumanMessage(content=state.curated_question)
    ]
    response = llm.invoke(messages)
    clean_sql = response.content.replace("```sql", "").replace("```", "").strip()
    return {"generated_sql_query": clean_sql}

def is_safe_sql(state: SQLAgentState) -> dict:
    # 1. حماية بالنص (Text-based Fallback)
    lower_sql = state.generated_sql_query.lower()
    forbidden_words = ["insert", "update", "delete", "drop", "truncate", "alter", "create", "grant"]
    if any(word in lower_sql for word in forbidden_words):
        return {"is_safe": "no", "comments": "Query contains forbidden modification commands."}

    # 2. حماية بالـ AI Judge
    judge_prompt = f"Is this SQL SAFE to run? It MUST be a SELECT query and MUST NOT modify data.\nSQL: {state.generated_sql_query}"
    try:
        response = judge_llm.invoke(judge_prompt)
        if isinstance(response, dict):
            return {"is_safe": response.get("answer", "yes"), "comments": response.get("comments", "Checked dict")}
        elif hasattr(response, "answer"):
            return {"is_safe": response.answer, "comments": response.comments}
        else:
            return {"is_safe": "yes", "comments": "Safe by basic text validation"}
    except Exception:
        return {"is_safe": "yes", "comments": "Safe by basic text validation (LLM fail)"}

def execute_sql(state: SQLAgentState) -> dict:
    result = db_util.execute_sql(state.generated_sql_query)
    return {"sql_execution_result": str(result)}

def represent_final_answer(state: SQLAgentState) -> dict:
    # لو النتيجة فيها error، قول كده صراحة بدل ما تحاول "تشرح" الerror
    if "error" in state.sql_execution_result.lower():
        return {"final_answer": f"حصل خطأ أثناء تنفيذ الاستعلام:\n{state.sql_execution_result}"}

    if not state.sql_execution_result or state.sql_execution_result.strip() in ("", "[]", "None"):
        return {"final_answer": "الاستعلام لم يرجع أي نتائج من قاعدة البيانات."}

    represent_prompt = (
        f"You are a helpful Data Analyst. Translate the raw SQL execution results "
        f"into a clear, natural answer for the user.\n"
        f"IMPORTANT: ONLY use the data provided below. Do NOT add any information, "
        f"numbers, or facts that are not explicitly in the SQL results. "
        f"If the results are empty or unclear, say so honestly.\n"
        f"User Question: {state.curated_question}\n"
        f"SQL Execution Result: {state.sql_execution_result}"
    )
    response = llm.invoke(represent_prompt)
    return {"final_answer": response.content}

def cancel_sql(state: SQLAgentState) -> dict:
    return {"final_answer": f"Sorry, the generated SQL query was deemed unsafe. Reason: {state.comments}"}

# ==========================================
# 3. بناء خريطة الـ Graph
# ==========================================
workflow = StateGraph(SQLAgentState)

workflow.add_node("curate_question", curate_question)
workflow.add_node("prompt_query_context", prompt_query_context)
workflow.add_node("generate_sql", generate_sql)
workflow.add_node("is_safe_sql", is_safe_sql)
workflow.add_node("execute_sql", execute_sql)
workflow.add_node("represent_final_answer", represent_final_answer)
workflow.add_node("cancel_sql", cancel_sql)

workflow.add_edge(START, "curate_question")
workflow.add_edge("curate_question", "prompt_query_context")
workflow.add_edge("prompt_query_context", "generate_sql")
workflow.add_edge("generate_sql", "is_safe_sql")

def check_safety(state: SQLAgentState):
    safety = state.is_safe.lower() if state.is_safe else "no"
    if safety == "yes":
        return "execute_sql"
    else:
        return "cancel_sql"

workflow.add_conditional_edges("is_safe_sql", check_safety, {"execute_sql": "execute_sql", "cancel_sql": "cancel_sql"})
workflow.add_edge("execute_sql", "represent_final_answer")
workflow.add_edge("represent_final_answer", END)
workflow.add_edge("cancel_sql", END)

sql_analyst_agent = workflow.compile()

# ==========================================
# 4. التجربة العملية
# ==========================================
if __name__ == "__main__":
    print("\nAgent is thinking... Please wait 🤖\n")

    user_question = "What are the top 3 restaurants based on customer rating?"

    initial_state = {
        "messages": [HumanMessage(content=user_question)]
    }

    result = sql_analyst_agent.invoke(initial_state)

    print("\n" + "="*60)
    print("🤖 Agent Final Answer:")
    print("="*60)
    print(result.get("final_answer", "No answer generated."))

    print("\n" + "="*60)
    print("💻 Under the Hood (Generated SQL):")
    print("="*60)
    print(result.get("generated_sql_query", "No SQL generated."))