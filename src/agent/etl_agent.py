"""
ReAct ETL agent. Compared to the previous version:
  * model selection no longer hardcodes ANY specific Groq model id — not
    even as a "safe" fallback. The previous fallback (`llama-3.1-8b-instant`)
    is exactly what produced the reported `404 model_not_found` error once
    Groq decommissioned it; a fallback that's just as hardcoded as the
    primary choice isn't a real fix, it just delays the same failure.
  * Model selection now goes through `resolve_groq_model`,
    backed by `groq_model_resolver.py`, which checks Groq's live model
    catalog and verifies a candidate actually responds before using it —
    see that module's docstring for the full self-healing behavior.
  * the tool surface is simplified around the new intelligent pipeline
    (see etl_tools.py) instead of a scrape -> raw-text -> LLM-guess chain.
"""
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from src.tools import config
from src.tools import groq_model_resolver
from src.tools.etl_tools import etl_toolkit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("etl.agent")

# Resolve the model safely using the factory method required by our new resolver
ACTIVE_MODEL = groq_model_resolver.resolve_groq_model(
    api_key=config.GROQ_API_KEY,
    chat_model_factory=lambda model_id: ChatGroq(
        model=model_id, 
        temperature=0.0, 
        api_key=config.GROQ_API_KEY
    )
)

logger.info("ETL agent using Groq model: %s", ACTIVE_MODEL)

llm = ChatGroq(model=ACTIVE_MODEL, temperature=0.0, api_key=config.GROQ_API_KEY)

# Defined as lowercase so router.py can import it without errors
system_prompt = """You are a Senior ETL Data Engineer. Your job is to extract data from APIs or
websites and save it in a clean, structured, professional format.

RULES:
1. If the source is an API (returns JSON), use `extract_from_api` directly.
2. If the source is a website, use `extract_website_data`. Pass `fields` only
   if the user asked for specific fields; otherwise leave it empty and let
   the engine decide what's valuable.
3. To transform/filter an existing CSV, use `execute_pandas_code`.
4. Never fabricate data that isn't present in the source. If a field is
   missing, it stays missing.
5. Always report the final output file path to the user.
"""

# Maintain uppercase alias just in case it's used elsewhere in the file
SYSTEM_PROMPT = system_prompt 

etl_analyst_agent = create_react_agent(model=llm, tools=etl_toolkit)


def run(user_question: str):
    initial_state = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
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