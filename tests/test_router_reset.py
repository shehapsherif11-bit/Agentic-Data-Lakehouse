import pytest
from src.agent.router_graph import build_graph
from langchain_core.messages import HumanMessage

def test_router_resets_analyst_state():
    graph = build_graph()
    
    # Simulate a failed question state
    failed_state = {
        "messages": [HumanMessage(content="Failed question")],
        "analyst_memory": {
            "repair_attempts": 3,
            "error": "Max retries exceeded",
            "error_type": "guard",
            "safety_blocked": True,
            "sql_queries": [{"safety_blocked": True, "error": "Bad SQL", "error_type": "guard"}],
            "query_results": []
        }
    }
    
    new_message = HumanMessage(content="New fresh question")
    failed_state["messages"].append(new_message)
    
    from src.agent.router_graph import analysis_node
    import src.agent.router_graph as rg
    
    captured_input = {}
    class MockAnalyst:
        def invoke(self, state, config=None, **kwargs):
            captured_input.update(state)
            return {"final_answer": "Mocked answer"}
            
    original_analyst = rg.analyst_agent
    rg.analyst_agent = MockAnalyst()
    
    try:
        analysis_node(failed_state)
        
        assert captured_input["repair_attempts"] == 0
        assert "error" not in captured_input or not captured_input["error"]
        assert "error_type" not in captured_input or not captured_input["error_type"]
        assert "safety_blocked" not in captured_input or not captured_input["safety_blocked"]
        assert "sql_queries" not in captured_input or not captured_input["sql_queries"]
    finally:
        rg.analyst_agent = original_analyst
