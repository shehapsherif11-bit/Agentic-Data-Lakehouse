import logging
import sqlglot
import sqlglot.expressions as exp
from dataclasses import dataclass

logger = logging.getLogger("sql_safety_guard")

@dataclass
class SafetyVerdict:
    safe: bool
    reason: str

class SQLGuardError(Exception):
    pass

FORBIDDEN_NODE_TYPES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, 
    exp.Merge, exp.Alter, exp.Command, exp.Use, exp.Grant
)

DANGEROUS_FUNCTIONS = {
    "read_files", "http_request", "remote_query", "ai_query", "input_file_name"
}

ALLOWED_SCHEMA = "workspace.zomato_gold"
MAX_ROWS = 10000

# Whole families of functions that send row data to a model endpoint (ai_query, ai_gen, ai_classify, ...).
DANGEROUS_FUNCTION_PREFIXES = ("ai_",)


def _function_name(func: exp.Func) -> str:
    """Lower-cased name of a function as the SQL spells it.

    sqlglot parses functions it does not know (ai_query, http_request, remote_query, ...) as exp.Anonymous.
    For those, Func.sql_name() returns the class name "ANONYMOUS", never the real name, so a blocklist
    that compares sql_name() never matches. The real name lives on the node itself.
    """
    if isinstance(func, exp.Anonymous):
        return str(func.name).lower()
    return func.sql_name().lower()

def guard_sql(sql: str) -> tuple[str, bool]:
    """
    Validates SQL using AST parsing (sqlglot) against security rules.
    Returns the normalized safe SQL, or raises SQLGuardError.
    """
    if not sql or not sql.strip():
        raise SQLGuardError("Empty SQL query.")
        
    try:
        # Parse exactly one statement
        statements = sqlglot.parse(sql, read="databricks")
    except Exception as e:
        raise SQLGuardError(f"Syntax error or unsupported dialect: {str(e)}")
        
    if not statements:
        raise SQLGuardError("Empty SQL query.")
    if len(statements) > 1:
        raise SQLGuardError("Multiple SQL statements detected. Only one statement is allowed.")
        
    ast = statements[0]
    if ast is None:
         raise SQLGuardError("Could not parse statement.")
         
    # 1. Root must be SELECT or UNION (WITH is handled internally as part of SELECT in sqlglot)
    if not isinstance(ast, (exp.Select, exp.Union)):
        raise SQLGuardError(f"Root statement must be SELECT or UNION. Found: {ast.__class__.__name__}")
        
    # 2. Block forbidden AST nodes anywhere in the tree
    for node in ast.find_all(exp.Expression):
        if isinstance(node, FORBIDDEN_NODE_TYPES):
            raise SQLGuardError(f"Forbidden operation detected: {node.__class__.__name__}")
            
    # 3. No SELECT * (COUNT(*) is fine)
    # This includes `SELECT *` and `SELECT t.*`
    for select_node in ast.find_all(exp.Select):
        for expr in select_node.expressions:
            # Check for direct Star or Star inside a Column (t.*)
            has_star = isinstance(expr, exp.Star) or (isinstance(expr, exp.Column) and isinstance(expr.this, exp.Star))
            if has_star:
                parent = expr.parent
                if not isinstance(parent, exp.Count):
                    raise SQLGuardError("SELECT * is not allowed. Please specify columns explicitly.")
                    
    # 4. Table validation
    cte_names = set()
    for cte in ast.find_all(exp.CTE):
        cte_names.add(cte.alias.lower())
        
    for table in ast.find_all(exp.Table):
        # 4b. Table-valued functions (TVFs) check: 
        # If table.this is not an Identifier, it might be a function like read_files()
        if not isinstance(table.this, exp.Identifier):
            raise SQLGuardError(f"Table-valued functions or non-identifier table references are not allowed: {table.sql()}")
            
        table_name = table.name.lower()
        # A bare name that matches a CTE is a CTE reference. A qualified name never is: skipping it would let
        # "WITH s AS (...) SELECT ... FROM workspace.zomato_silver.s" read a schema the guard must block.
        if not table.db and not table.catalog and table_name in cte_names:
            continue
            
        db_name = table.db.lower()
        catalog_name = table.catalog.lower()
        
        full_schema = ""
        if catalog_name:
            full_schema = f"{catalog_name}.{db_name}"
        elif db_name:
            full_schema = db_name
            
        if full_schema != ALLOWED_SCHEMA:
            raise SQLGuardError(f"Table '{table.sql()}' is not in the allowed schema '{ALLOWED_SCHEMA}'. Must be fully qualified.")

    # 5. Dangerous functions
    for func in ast.find_all(exp.Func):
        name = _function_name(func)
        if name in DANGEROUS_FUNCTIONS or name.startswith(DANGEROUS_FUNCTION_PREFIXES):
            raise SQLGuardError(f"Dangerous function detected: {name}")
            
    # 6. Cap every result at MAX_ROWS: add a LIMIT when there is none and lower one that is larger. A LIMIT
    #    that is not a plain integer cannot be capped, so it is rejected. limit_injected is True whenever the
    #    guard set or lowered the limit, which lets callers flag a possibly truncated result.
    limit_injected = False
    limit_node = ast.args.get("limit")
    if not limit_node:
        ast = ast.limit(MAX_ROWS)
        limit_injected = True
    else:
        limit_value = limit_node.expression
        if not (isinstance(limit_value, exp.Literal) and limit_value.is_int):
            raise SQLGuardError("LIMIT must be a plain integer.")
        if int(limit_value.this) > MAX_ROWS:
            limit_node.set("expression", exp.Literal.number(MAX_ROWS))
            limit_injected = True

    # Return normalized SQL
    return ast.sql(dialect="databricks"), limit_injected

def check_sql_safety(sql: str) -> SafetyVerdict:
    """Wrapper that returns SafetyVerdict for compatibility."""
    try:
        guard_sql(sql)
        return SafetyVerdict(True, "")
    except SQLGuardError as e:
        return SafetyVerdict(False, str(e))

def check_multiple_queries(queries: list[dict]) -> list[dict]:
    """
    Validate a batch of query dicts [{sql, purpose, ...}].
    """
    results = []
    for q in queries:
        try:
            safe_sql, limit_injected = guard_sql(q.get("sql", ""))
            updated = {**q, "sql": safe_sql, "safety_passed": True, "error": None, "error_type": None, "limit_injected": limit_injected}
        except SQLGuardError as e:
            logger.warning(
                "SQL BLOCKED by safety guard | purpose='%s' | reason='%s'",
                q.get("purpose", ""), str(e)
            )
            updated = {**q, "safety_passed": False, "error": f"SECURITY BLOCKED: {str(e)}", "error_type": "guard"}
        results.append(updated)
    return results
