import os
import logging
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

def get_llm(model_name: str = None, temperature: float = 0.0, max_retries: int = 2, request_timeout: int = 15):
    groq_api_key = os.getenv("GROQ_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    
    # Primary: Groq (Ultra-fast 0.7s latency on LPUs, high reliable quota)
    if groq_api_key:
        model = model_name if (model_name and not model_name.startswith("nvidia/")) else "openai/gpt-oss-120b"
        logger.info(f"Initializing Groq LLM (model: {model})")
        return ChatGroq(
            model=model,
            temperature=temperature,
            api_key=groq_api_key,
            max_retries=max_retries,
            request_timeout=request_timeout
        )
    # Secondary: OpenRouter
    elif openrouter_api_key:
        model = model_name or os.getenv("GENERAL_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
        logger.info(f"Initializing OpenRouter LLM (model: {model})")
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            request_timeout=35
        )
    else:
        raise ValueError("Neither GROQ_API_KEY nor OPENROUTER_API_KEY found in environment.")
