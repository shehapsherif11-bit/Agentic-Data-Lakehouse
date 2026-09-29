import sys
import json
sys.path.insert(0, r'd:\Data ENG Project')
from src.agent.analyst_graph import analyst_agent
from langchain_core.messages import HumanMessage
import warnings
warnings.filterwarnings('ignore')

def run_query(q):
    print(f'\n\n{"="*50}\nQUERY: {q}\n{"="*50}')
    try:
        res = analyst_agent.invoke({
            'messages': [HumanMessage(content=q)],
            'language': 'en',
            'retry_count': 0,
            'current_step': 0,
        })
        print('\n--- INTENT ---')
        print(json.dumps(res.get('intent', {}), indent=2))
        print('\n--- METRICS ---')
        print(json.dumps(res.get('resolved_metrics', []), indent=2))
        print('\n--- SUFFICIENCY ---')
        print(json.dumps(res.get('data_sufficiency', {}), indent=2))
        print('\n--- PLAN ---')
        print(json.dumps(res.get('analysis_plan', {}), indent=2))
        print('\n--- SQL ---')
        for sql in res.get('sql_queries', []):
            print(f'Purpose: {sql.get("purpose")}')
            print(f'{sql.get("sql")}\n')
            if sql.get("error"):
                print(f'ERROR: {sql.get("error")}')
        print('\n--- QUERY RESULTS SUMMARY ---')
        for idx, qr in enumerate(res.get('query_results', [])):
            print(f'Result {idx} ({qr.get("purpose")}): {qr.get("row_count")} rows')
        print('\n--- DRIVER ANALYSIS DECISION ---')
        print('is_driver_question:', res.get('intent', {}).get('is_driver_question'))
        print('needs_driver_analysis:', res.get('analysis_plan', {}).get('needs_driver_analysis'))
        print('driver summary:', res.get('analysis_result', {}).get('driver_summary', ''))
        
    except Exception as e:
        print('Error:', e)
        import traceback
        traceback.print_exc()

run_query('What are the top 10 restaurants by sales?')
run_query('Which week had the biggest drop in profit margin?')
run_query('Why did sales decrease in the worst-performing week?')
