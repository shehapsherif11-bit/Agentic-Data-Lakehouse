import os
import unittest
from datetime import datetime, timezone

from src.agent.telemetry import (
    TelemetryCallbackHandler,
    set_current_stage,
    reset_current_stage,
    start_turn_telemetry,
    get_turn_telemetry,
    format_telemetry_table,
    _write_audit_record,
    AUDIT_LOG_PATH
)
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult, Generation, ChatGeneration

class TestTelemetry(unittest.TestCase):

    def test_stage_tracking(self):
        token = set_current_stage("test_stage_xyz")
        try:
            from src.agent.telemetry import get_current_stage
            self.assertEqual(get_current_stage(), "test_stage_xyz")
        finally:
            reset_current_stage(token)

    def test_telemetry_handler_on_end(self):
        handler = TelemetryCallbackHandler()
        start_turn_telemetry()
        
        token = set_current_stage("sql_generator")
        try:
            handler.on_llm_start({}, [], run_id="run-1")
            
            ai_msg = AIMessage(
                content="SELECT 1",
                response_metadata={
                    "model_name": "openai/gpt-oss-120b",
                    "model_provider": "groq",
                    "token_usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 25,
                        "completion_tokens_details": {"reasoning_tokens": 10}
                    }
                }
            )
            llm_result = LLMResult(generations=[[ChatGeneration(message=ai_msg)]])
            handler.on_llm_end(llm_result, run_id="run-1")
            
            records = get_turn_telemetry()
            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertEqual(rec["node"], "sql_generator")
            self.assertEqual(rec["model"], "openai/gpt-oss-120b")
            self.assertEqual(rec["provider"], "groq")
            self.assertFalse(rec["was_fallback"])
            self.assertEqual(rec["prompt_tokens"], 100)
            self.assertEqual(rec["completion_tokens"], 25)
            self.assertEqual(rec["reasoning_tokens"], 10)
            self.assertGreaterEqual(rec["wall_time"], 0.0)
        finally:
            reset_current_stage(token)

    def test_format_telemetry_table(self):
        sample = [{
            "node": "intent_analyzer",
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
            "was_fallback": False,
            "retries": 0,
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "reasoning_tokens": 15,
            "wall_time": 0.45
        }]
        table = format_telemetry_table(sample)
        self.assertIn("intent_analyzer", table)
        self.assertIn("openai/gpt-oss-120b", table)
        self.assertIn("0.450", table)

if __name__ == "__main__":
    unittest.main()
