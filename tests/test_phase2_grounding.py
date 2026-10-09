import pytest
from src.agent.analyst_graph import build_analyst_graph
from langchain_core.messages import HumanMessage
import src.agent.analyst_graph as ag
from src.agent.analyst_state import Evidence
import datetime

class FakeLLMPhase2:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0
        
    def invoke(self, prompt, *args, **kwargs):
        if self.call_count >= len(self.responses):
            res = "{}"
        else:
            res = self.responses[self.call_count]
        self.call_count += 1
        class Response:
            content = res
        return Response()

def test_evidence_immutability():
    # Evidence must be frozen
    ev = Evidence(
        query_id="q1",
        executed_sql="SELECT 1",
        columns=("a",),
        rows=((1,),),
        row_count=1,
        truncated=False,
        executed_at="2023-01-01T00:00:00Z"
    )
    
    with pytest.raises(Exception): # FrozenInstanceError or similar
        ev.row_count = 5

def test_grounding_fallback_template(monkeypatch):
    graph = build_analyst_graph()
    
    # We want to jump straight to insight generator to test it, or run full graph
    # Let's mock DB to return 100
    llm = FakeLLMPhase2([
        '{"is_followup": false, "intent_type": "simple_query", "metrics": ["revenue"], "dimensions": [], "filters": ["city = Cairo"], "is_driver_question": false}',
        '{}', 
        '```sql\nSELECT 100 as rev FROM workspace.zomato_gold.t\n```', 
        # First try: LLM invents a number "500"
        '{"insight": "The revenue is 500.", "viz_spec": {"should_visualize": false}}',
        # Second try: LLM STILL invents a number "600"
        '{"insight": "The revenue is 600.", "viz_spec": {"should_visualize": false}}'
    ])
    monkeypatch.setattr(ag, "_analyst_llm", llm)
    
    class MockDB:
        def execute_queries(self, queries):
            return [{"purpose": q.get("purpose", ""), "success": True, "rows": [{"rev": 100}], "row_count": 1, "columns": ["rev"]} for q in queries]
    monkeypatch.setattr(ag, "_db", MockDB())
    
    state = {
        "messages": [HumanMessage(content="What is the revenue?")],
        "language": "en",
        "repair_attempts": 0,
        "current_step": 0
    }
    
    result = graph.invoke(state)
    
    ans = result.get("final_answer", "")
    assert "The data returned 1 rows." in ans
    assert "Please view the 'How I got this' expander" in ans

def test_grounding_correct_passes(monkeypatch):
    graph = build_analyst_graph()
    llm = FakeLLMPhase2([
        '{"is_followup": false, "intent_type": "simple_query", "metrics": ["revenue"], "dimensions": [], "filters": ["city = Cairo"], "is_driver_question": false}',
        '{}', 
        '```sql\nSELECT 100 as rev FROM workspace.zomato_gold.t\n```', 
        # LLM tells the truth
        '{"insight": "The revenue is 100.", "viz_spec": {"should_visualize": false}}'
    ])
    monkeypatch.setattr(ag, "_analyst_llm", llm)
    
    class MockDB:
        def execute_queries(self, queries):
            return [{"purpose": q.get("purpose", ""), "success": True, "rows": [{"rev": 100}], "row_count": 1, "columns": ["rev"]} for q in queries]
    monkeypatch.setattr(ag, "_db", MockDB())
    
    state = {
        "messages": [HumanMessage(content="What is the revenue?")],
        "language": "en",
        "repair_attempts": 0,
        "current_step": 0
    }
    
    result = graph.invoke(state)
    assert result.get("final_answer", "") == "The revenue is 100."

def test_grounding_arabic_indic_digits(monkeypatch):
    graph = build_analyst_graph()
    llm = FakeLLMPhase2([
        '{"is_followup": false, "intent_type": "simple_query", "metrics": ["revenue"], "dimensions": [], "filters": ["city = Cairo"], "is_driver_question": false}',
        '{}', 
        '```sql\nSELECT 100 as rev FROM workspace.zomato_gold.t\n```', 
        # LLM tells the truth but with Arabic-Indic digits 
        '{"insight": "الإيرادات هي ١٠٠.", "viz_spec": {"should_visualize": false}}'
    ])
    monkeypatch.setattr(ag, "_analyst_llm", llm)
    
    class MockDB:
        def execute_queries(self, queries):
            return [{"purpose": q.get("purpose", ""), "success": True, "rows": [{"rev": 100}], "row_count": 1, "columns": ["rev"]} for q in queries]
    monkeypatch.setattr(ag, "_db", MockDB())
    
    state = {
        "messages": [HumanMessage(content="What is the revenue?")],
        "language": "ar",
        "repair_attempts": 0,
        "current_step": 0
    }
    
    result = graph.invoke(state)
    assert result.get("final_answer", "") == "الإيرادات هي ١٠٠."
