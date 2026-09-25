import os
import databricks.sql
from dotenv import load_dotenv

# تحميل متغيرات البيئة (نفس اللي استخدمناها في بايبلاين الداتا)
load_dotenv()

class DatabricksUtil:
    def __init__(self):
        self.host = os.getenv("DATABRICKS_HOST")
        self.http_path = os.getenv("DATABRICKS_HTTP_PATH")
        self.token = os.getenv("DATABRICKS_TOKEN")
        # هنركز على طبقة الـ Gold لأن دي الداتا النظيفة والجاهزة للتحليل
        self.catalog = "workspace"
        self.schema = "zomato_gold"

    def get_connection(self):
        """إنشاء اتصال مع Databricks"""
        return databricks.sql.connect(
            server_hostname=self.host,
            http_path=self.http_path,
            access_token=self.token
        )

    def get_schema_details(self) -> str:
        """
        الدالة دي بتدخل Databricks، بتجيب أسماء الجداول، وعواميد كل جدول، 
        وعينة من الداتا (3 صفوف) عشان تغذي بيهم הـ LLM
        """
        context = f"Database Schema Details for {self.catalog}.{self.schema}:\n\n"
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # 1. جلب أسماء الجداول
                    cursor.execute(f"SHOW TABLES IN {self.catalog}.{self.schema}")
                    # استخدام طريقة تحويل الرد لقاموس لضمان عدم حدوث إيرور
                    cols = [desc[0] for desc in cursor.description]
                    tables = [dict(zip(cols, row)) for row in cursor.fetchall()]
                    
                    for table in tables:
                        table_name = table["tableName"]
                        context += f"Table Name: {self.catalog}.{self.schema}.{table_name}\n"
                        
                        # 2. جلب أسماء العواميد وأنواع الداتا لكل جدول
                        cursor.execute(f"DESCRIBE {self.catalog}.{self.schema}.{table_name}")
                        desc_cols = [desc[0] for desc in cursor.description]
                        columns = [dict(zip(desc_cols, row)) for row in cursor.fetchall()]
                        
                        context += "Columns and Data Types:\n"
                        for col in columns:
                            if col["col_name"] and not col["col_name"].startswith("#"): # لتجاهل التعليقات الداخلية
                                context += f" - {col['col_name']}: {col['data_type']}\n"
                            
                        # 3. سحب عينة من الداتا (أول 3 صفوف فقط) عشان הـ LLM يفهم المحتوى
                        cursor.execute(f"SELECT * FROM {self.catalog}.{self.schema}.{table_name} LIMIT 3")
                        sample_cols = [desc[0] for desc in cursor.description]
                        sample_data = [dict(zip(sample_cols, row)) for row in cursor.fetchall()]
                        
                        context += "Sample Data (Top 3 rows):\n"
                        for row in sample_data:
                            context += f" {row}\n"
                            
                        context += "-" * 40 + "\n\n"
            return context
        except Exception as e:
            return f"Error fetching schema: {e}"

    def execute_sql(self, query: str) -> str:
        """الدالة دي هتاخد كود הـ SQL اللي הـ Agent كتبه، وتنفذه، وترجع النتيجة"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    cols = [desc[0] for desc in cursor.description]
                    results = [dict(zip(cols, row)) for row in cursor.fetchall()]
                    return str(results)
        except Exception as e:
            return f"SQL Execution Error: {e}"

# لو حابب تتأكد إن الكود شغال، جرب تعمل Run للفايل ده لوحده
if __name__ == "__main__":
    db = DatabricksUtil()
    print("Fetching Schema Details... Please wait.")
    schema_info = db.get_schema_details()
    print(schema_info)