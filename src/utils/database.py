import os
import databricks.sql
from dotenv import load_dotenv
import time
import difflib

# تحميل متغيرات البيئة (نفس اللي استخدمناها في بايبلاين الداتا)
load_dotenv()

class DatabricksUtil:
    _schema_cache: dict = {}  # class-level cache
    _schema_cache_time: float = 0
    SCHEMA_CACHE_TTL = 300  # 5 minutes

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

    def get_cached_schema(self) -> dict:
        """Returns cached schema info: {table_name: {columns: [{name, type}], sample: [...]}}.
        Fetches from Databricks only if cache is empty or expired."""
        current_time = time.time()
        if self.__class__._schema_cache and (current_time - self.__class__._schema_cache_time < self.__class__.SCHEMA_CACHE_TTL):
            return self.__class__._schema_cache

        schema_dict = {}
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # 1. جلب أسماء الجداول
                    cursor.execute(f"SHOW TABLES IN {self.catalog}.{self.schema}")
                    cols = [desc[0] for desc in cursor.description]
                    tables = [dict(zip(cols, row)) for row in cursor.fetchall()]
                    
                    for table in tables:
                        table_name = table["tableName"]
                        schema_dict[table_name] = {"columns": [], "sample": []}
                        
                        # 2. جلب أسماء العواميد وأنواع الداتا لكل جدول
                        cursor.execute(f"DESCRIBE {self.catalog}.{self.schema}.{table_name}")
                        desc_cols = [desc[0] for desc in cursor.description]
                        columns = [dict(zip(desc_cols, row)) for row in cursor.fetchall()]
                        
                        for col in columns:
                            if col["col_name"] and not col["col_name"].startswith("#"):
                                schema_dict[table_name]["columns"].append({"name": col["col_name"], "type": col["data_type"]})
                        
                        # 3. سحب عينة من الداتا (أول 3 صفوف فقط)
                        cursor.execute(f"SELECT * FROM {self.catalog}.{self.schema}.{table_name} LIMIT 3")
                        sample_cols = [desc[0] for desc in cursor.description]
                        sample_data = [dict(zip(sample_cols, row)) for row in cursor.fetchall()]
                        
                        schema_dict[table_name]["sample"] = sample_data
            
            self.__class__._schema_cache = schema_dict
            self.__class__._schema_cache_time = current_time
            return schema_dict
        except Exception as e:
            return {"error": f"Error fetching schema: {e}"}

    def validate_columns(self, table_name: str, columns: list[str]) -> dict:
        """Validate that columns exist in the given table.
        Returns: {valid: bool, invalid_columns: list[str], suggestions: dict[str, str]}
        Suggestions maps invalid column to closest valid column name."""
        schema = self.get_cached_schema()
        if "error" in schema:
            return {"valid": False, "invalid_columns": columns, "suggestions": {}, "error": schema["error"]}
        
        if table_name not in schema:
            return {"valid": False, "invalid_columns": columns, "suggestions": {}, "error": f"Table {table_name} not found"}
            
        valid_columns = [col["name"] for col in schema[table_name]["columns"]]
        invalid_columns = []
        suggestions = {}
        
        for col in columns:
            if col not in valid_columns:
                invalid_columns.append(col)
                matches = difflib.get_close_matches(col, valid_columns, n=1)
                if matches:
                    suggestions[col] = matches[0]
                    
        return {
            "valid": len(invalid_columns) == 0,
            "invalid_columns": invalid_columns,
            "suggestions": suggestions
        }

    def execute_queries(self, queries: list[dict]) -> list[dict]:
        """Execute multiple SQL queries sequentially.
        Input: [{sql: str, purpose: str}]
        Returns: [{purpose, sql, columns, rows: list[dict], row_count, error, success}]
        Each query is independent — a failure in one doesn't stop the rest."""
        results = []
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    for query_info in queries:
                        sql = query_info.get("sql", "")
                        purpose = query_info.get("purpose", "")
                        result_item = {
                            "purpose": purpose,
                            "sql": sql,
                            "columns": [],
                            "rows": [],
                            "row_count": 0,
                            "error": None,
                            "success": False
                        }
                        try:
                            cursor.execute(sql)
                            if cursor.description:
                                cols = [desc[0] for desc in cursor.description]
                                result_item["columns"] = cols
                                rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
                                result_item["rows"] = rows
                                result_item["row_count"] = len(rows)
                            result_item["success"] = True
                        except Exception as e:
                            result_item["error"] = str(e)
                            
                        results.append(result_item)
        except Exception as e:
            return [{"error": f"Execution Error: {e}", "success": False}]
            
        return results

    def get_schema_as_dict(self) -> dict:
        """Return schema as a structured dict instead of a formatted string.
        Returns: {table_name: {columns: [{name: str, type: str}], row_count_approx: int}}
        Uses the cache."""
        schema = self.get_cached_schema()
        if "error" in schema:
            return schema
            
        result = {}
        for table_name, data in schema.items():
            result[table_name] = {
                "columns": data["columns"],
                "row_count_approx": len(data.get("sample", []))
            }
        return result

# لو حابب تتأكد إن الكود شغال، جرب تعمل Run للفايل ده لوحده
if __name__ == "__main__":
    db = DatabricksUtil()
    print("Fetching Schema Details... Please wait.")
    schema_info = db.get_schema_details()
    print(schema_info)