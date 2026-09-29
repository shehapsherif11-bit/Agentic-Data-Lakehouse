import os

file_path = r'd:\Data ENG Project\src\agent\analyst_graph.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

decorator_code = '''import time
from functools import wraps

def track_performance(stage_name):
    def decorator(func):
        @wraps(func)
        def wrapper(state: AnalystState):
            start = time.time()
            result = func(state)
            latency = time.time() - start
            
            # Count how many LLM calls happened by looking if it's an LLM node
            is_llm_node = stage_name not in ['metric_resolver', 'sufficiency_check', 'sql_executor', 'sql_validator', 'result_validator']
            
            updates = result if result else {}
            
            llm_count = state.get('llm_call_count', 0)
            latencies = state.get('stage_latencies', {})
            
            new_latencies = dict(latencies)
            new_latencies[stage_name] = round(latency, 2)
            
            updates['stage_latencies'] = new_latencies
            if is_llm_node:
                updates['llm_call_count'] = llm_count + 1
                
            return updates
        return wrapper
    return decorator
'''

if 'def track_performance' not in content:
    idx = content.find('logger = logging.getLogger("analyst")')
    content = content[:idx] + decorator_code + '\n' + content[idx:]

    nodes = [
        ('intent_analyzer', 'intent_analyzer'),
        ('metric_resolver', 'metric_resolver'),
        ('data_sufficiency_check', 'sufficiency_check'),
        ('analysis_planner', 'analysis_planner'),
        ('sql_generator', 'sql_generator'),
        ('sql_validator', 'sql_validator'),
        ('sql_executor', 'sql_executor'),
        ('sql_repair', 'sql_repair'),
        ('result_validator', 'result_validator'),
        ('result_analyzer', 'result_analyzer'),
        ('driver_analysis', 'driver_analysis'),
        ('completeness_check', 'completeness_check'),
        ('insight_generator', 'insight_generator')
    ]
    
    for func_name, stage_name in nodes:
        old_def = f'def {func_name}(state: AnalystState) -> dict:'
        new_def = f'@track_performance("{stage_name}")\ndef {func_name}(state: AnalystState) -> dict:'
        content = content.replace(old_def, new_def)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('Performance tracking added')
else:
    print('Already added')
