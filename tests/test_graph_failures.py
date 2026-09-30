import pytest
from src.agent.analyst_graph import build_analyst_graph
from langchain_core.messages import HumanMessage
from src.utils.database import DatabricksUtil
import src.agent.analyst_graph as ag

class FakeLLM:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0
        
    def invoke(self, prompt, *args, **kwargs):
        res = self.responses[self.call_count]
        self.call_count += 1
        class Response:
            content = res
        return Response()

def test_scenario_a_repair_success(monkeypatch):
    graph = build_analyst_graph()
    
    llm = FakeLLM([
        '{"is_followup": false, "intent_type": "simple_query", "metrics": ["revenue"], "dimensions": [], "filters": [], "is_driver_question": false}',
        '{}', 
        '```sql\nSELECT * FROM t\n```', 
        '```sql\nSELECT a FROM workspace.zomato_gold.t\n```', 
        '{}',
        '{}',
        '{"is_complete": true}',
        '{"insight": "Done", "viz_spec": {"should_visualize": false}}',
        '{}'
    ])
    monkeypatch.setattr(ag, "_analyst_llm", llm)
    
    db_calls = []
    class MockDB:
        def execute_queries(self, queries):
            db_calls.append(queries)
            return [{"purpose": q.get("purpose", ""), "success": True, "rows": [{"a": 1}], "row_count": 1, "columns": ["a"]} for q in queries]
    monkeypatch.setattr(ag, "_db", MockDB())
    
    state = {
        "messages": [HumanMessage(content="test")],
        "language": "en",
        "repair_attempts": 0,
        "current_step": 0
    }
    
    result = graph.invoke(state)
    
    assert result["repair_attempts"] == 1
    assert len(db_calls) == 1
    assert db_calls[0][0]["sql"] == "SELECT a FROM workspace.zomato_gold.t LIMIT 10000"
    assert result.get("error") is None or result.get("error") == ""

def test_scenario_b_max_repairs(monkeypatch):
    graph = build_analyst_graph()
    
    llm = FakeLLM([
        '{"is_followup": false, "intent_type": "simple_query", "metrics": ["revenue"], "dimensions": [], "filters": [], "is_driver_question": false}',
        '{}',
        '```sql\nDROP TABLE workspace.zomato_gold.t\n```',
        '```sql\nDROP TABLE workspace.zomato_gold.t\n```',
        '```sql\nDROP TABLE workspace.zomato_gold.t\n```',
        '```sql\nDROP TABLE workspace.zomato_gold.t\n```',
    ])
    monkeypatch.setattr(ag, "_analyst_llm", llm)
    
    db_calls = []
    class MockDB:
        def execute_queries(self, queries):
            db_calls.append(queries)
            return []
    monkeypatch.setattr(ag, "_db", MockDB())
    
    state = {
        "messages": [HumanMessage(content="test")],
        "language": "en",
        "repair_attempts": 0,
        "current_step": 0
    }
    
    result = graph.invoke(state)
    
    assert result["repair_attempts"] == 3
    assert len(db_calls) == 0 
    assert "error_type" in result["sql_queries"][0]
    assert result["sql_queries"][0]["error_type"] == "guard"

def test_scenario_c_connection_retry(monkeypatch):
    db = DatabricksUtil()
    
    attempts = [0]
    class MockCursor:
        def execute(self, sql):
            attempts[0] += 1
            if attempts[0] == 1:
                raise Exception("socket closed")
        def fetchall(self): return []
        def __enter__(self): return self
        def __exit__(self, *args): pass
        @property
        def description(self): return []
        
    class MockConn:
        open = True
        def cursor(self): return MockCursor()
        
    class MockWrapper:
        def __init__(self, c): self.conn = c
        def __enter__(self): return self.conn
        def __exit__(self, *args): pass
    
    monkeypatch.setattr(db, "get_connection", lambda: MockWrapper(MockConn()))
    # Also need to mock databricks.sql.connect inside the retry block
    import databricks.sql
    monkeypatch.setattr(databricks.sql, "connect", lambda **k: MockConn())
    
    res = db.execute_queries([{"sql": "SELECT 1"}])
    assert attempts[0] == 2
    assert res[0]["success"] is True
    assert res[0]["error"] is None
