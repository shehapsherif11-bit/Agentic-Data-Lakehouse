"""
src/agent/etl_agent.py

ReAct ETL agent. Compared to the previous version:
  * model selection no longer hardcodes ANY specific Groq model id — not
    even as a "safe" fallback. The previous fallback (`llama-3.1-8b-instant`)
    is exactly what produced the reported `404 model_not_found` error once
    Groq decommissioned it; a fallback that's just as hardcoded as the
    primary choice isn't a real fix, it just delays the same failure.
    Model selection now goes through `config.get_active_groq_model()`,
    backed by `groq_model_resolver.py`, which checks Groq's live model
    catalog and verifies a candidate actually responds before using it —
    see that module's docstring for the full self-healing behavior.
  * the tool surface is simplified around the new intelligent pipeline
    (see etl_tools.py) instead of a scrape -> raw-text -> LLM-guess chain.
  * Exposes `system_prompt` (lowercase) at module level, per this project's
    import convention, so `src/agent/router.py` can import it without an
    ImportError.
"""
import logging

from langchain_core.messages import SystemMessage, HumanMessage
from src.agent.llm_factory import get_llm
from langgraph.prebuilt import create_react_agent

from src.tools import config
from src.tools.etl_tools import etl_toolkit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("etl.agent")

ACTIVE_MODEL = config.get_active_groq_model()
logger.info("ETL agent using Groq model: %s", ACTIVE_MODEL)

llm = get_llm(model_name=ACTIVE_MODEL, temperature=0.0)

system_prompt = """You are a Senior ETL Data Engineer. Your job is to extract data from APIs or
websites and save it in a clean, structured, professional format.

RULES:
1. If the source is an API (returns JSON), use `extract_from_api` directly.
2. If the source is a website, use `extract_website_data`.
   - If the user names or implies specific data points to extract, in ANY language, you MUST pass
     them via `fields` as a comma-separated list of clear English field labels. Translate/normalize
     the user's wording first.
     Example: user says "استخرج اسم المنتج، السعر، والقسم أو التصنيف" ->
              fields="Product Name, Price, Category"
     Example: user says "get me the title, author and date of each article" ->
              fields="Title, Author, Date"
   - Only leave `fields` empty when the user asks for "whatever is useful/valuable" with no
     specifics named — never leave it empty just because the request was phrased as a sentence
     instead of a bare list.
3. To transform/filter an existing CSV, use `execute_pandas_code`.
4. Never fabricate data that isn't present in the source. If a field is missing, it stays missing.
5. If a tool's result starts with "ERROR", or a saved file's data is generic page metadata rather
   than what was asked for, you MUST relay that honestly to the user. Never claim success or invent
   a plausible-sounding summary when the underlying tool did not actually get the requested data.
6. Always report the final output file path to the user — only when extraction actually succeeded.
"""

# Backward/forward-compatible alias: this project has had both naming
# conventions in play across files at different points (lowercase
# `system_prompt` per this project's explicit convention, uppercase
# `SYSTEM_PROMPT` used by earlier versions of the router). Exporting both
# names pointing at the same string means neither import style can break
# the other — this is exactly the class of bug that caused the ETL agent
# to fail to load at startup after a previous update ("cannot import name
# 'SYSTEM_PROMPT'"). Both names always stay in sync since one is just a
# reference to the other, not a separate copy.
SYSTEM_PROMPT = system_prompt

etl_analyst_agent = create_react_agent(model=llm, tools=etl_toolkit)


def run(user_question: str):
    initial_state = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_question),
        ]
    }
    final_chunk = None
    for chunk in etl_analyst_agent.stream(initial_state):
        final_chunk = chunk
        if "agent" in chunk:
            msg = chunk["agent"]["messages"][0]
            if getattr(msg, "tool_calls", None):
                print(f"tool call: {msg.tool_calls[0]['name']}")
            elif msg.content:
                print(f"agent: {msg.content}")
        elif "tools" in chunk:
            print("tool finished")
    return list(final_chunk.values())[0]["messages"][-1].content if final_chunk else None


if __name__ == "__main__":
    question = (
        "Extract user data from 'https://jsonplaceholder.typicode.com/users' and save it to "
        "'data/raw_users.csv'. Then filter the users who work at the company 'Romaguera-Crona', "
        "and save the result to 'data/filtered_users.csv'."
    )
    answer = run(question)
    print("\n" + "=" * 60)
    print("Final Agent Answer:")
    print("=" * 60)
    print(answer)