import os
import requests
import databricks.sql
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. جلب بيانات الاتصال
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# الدالة السحرية: بتسحب اسم الموديل الشغال حالاً من السيرفر عشان منخمنش
def get_active_model():
    try:
        res = requests.get("https://api.groq.com/openai/v1/models", headers=HEADERS)
        if res.status_code == 200:
            models = res.json().get("data", [])
            for m in models:
                # نختار أي موديل Llama أو Mixtral شغال دلوقتي
                if "llama" in m["id"].lower() or "mixtral" in m["id"].lower():
                    return m["id"]
            if models:
                return models[0]["id"]
    except Exception as e:
        print(f"Error fetching models: {e}")
    return "llama3-8b-8192" # احتياطي

# الكود هيقرأ الموديل الشغال أوتوماتيك
ACTIVE_MODEL = get_active_model()

def get_sentiment(review_text):
    if not review_text or len(str(review_text).strip()) == 0:
        return "Neutral"

    payload = {
        "model": ACTIVE_MODEL, 
        "messages": [
            {
                "role": "user",
                "content": f"Classify the sentiment of the following restaurant review as ONLY 'Positive', 'Negative', or 'Neutral'. Do not write any other words.\n\nReview: {str(review_text)}"
            }
        ]
    }
    
    try:
        response = requests.post(GROQ_URL, headers=HEADERS, json=payload, timeout=15)
        if response.status_code != 200:
            print(f"API Error from Groq: {response.text}")
            return "Neutral"
            
        sentiment = response.json()["choices"][0]["message"]["content"].strip()
        
        if "positive" in sentiment.lower(): return "Positive"
        elif "negative" in sentiment.lower(): return "Negative"
        else: return "Neutral"
    except Exception as e:
        print(f"Code Exception: {e}")
        return "Neutral"

def process_reviews():
    print("Connecting to Databricks...")
    print(f"Target locked! Using dynamically fetched active model: {ACTIVE_MODEL}")
    
    connection = databricks.sql.connect(
        server_hostname=DATABRICKS_HOST,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN
    )
    
    cursor = connection.cursor()
    
    try:
        cursor.execute("DROP TABLE IF EXISTS workspace.zomato_gold.ai_enriched_reviews")
        
        cursor.execute("""
            CREATE TABLE workspace.zomato_gold.ai_enriched_reviews (
                review_id INT,
                comment STRING,
                sentiment STRING
            )
        """)
        
        print("Fetching 10 reviews from Databricks...")
        cursor.execute("""
            SELECT review_id, comment 
            FROM workspace.zomato_silver.silver_reviews 
            LIMIT 10
        """)
        
        columns = [desc[0] for desc in cursor.description]
        reviews = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        if not reviews:
            print("No new reviews to process. Exiting.")
            return

        print(f"Found {len(reviews)} reviews. Starting AI processing...")
        
        enriched_data = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_review = {executor.submit(get_sentiment, row["comment"]): row for row in reviews}
            
            for future in as_completed(future_to_review):
                row = future_to_review[future]
                try:
                    sentiment = future.result()
                    enriched_data.append((row["review_id"], row["comment"], sentiment))
                except Exception as exc:
                    pass # متطبعش إيرورات في النص عشان منزحمهاش

        print("Saving truly enriched reviews back to Databricks...")
        
        insert_query = """
            INSERT INTO workspace.zomato_gold.ai_enriched_reviews (review_id, comment, sentiment)
            VALUES (?, ?, ?)
        """
        cursor.executemany(insert_query, enriched_data)
        
        print("Successfully enriched and saved all reviews! 🎉")
        
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    process_reviews()