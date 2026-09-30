import pytest
import time
from langchain_core.messages import HumanMessage
from groq import RateLimitError
import httpx
from src.agent.router_graph import analysis_node

def test_rate_limit_handled_gracefully():
    # We will mock the analyst_agent to raise RateLimitError
    import src.agent.router_graph as rg
    
    class FakeLLMGraph:
        def invoke(self, *args, **kwargs):
            # Groq's RateLimitError requires a response and body
            response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"))
            raise RateLimitError("Rate limit exceeded", response=response, body={})
            
    original_agent = rg.analyst_agent
    rg.analyst_agent = FakeLLMGraph()
    
    try:
        start_time = time.time()
        state = {
            "messages": [HumanMessage(content="What is the revenue?")],
            "language": "en"
        }
        
        result = analysis_node(state)
        duration = time.time() - start_time
        
        assert duration < 5.0, f"Took {duration} seconds, should be < 5s!"
        assert result["final_answer"] == "The service is busy, please try again in a moment."
    finally:
        rg.analyst_agent = original_agent
