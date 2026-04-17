import sqlite3
import re
import json
from typing import Annotated

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.ollama import Ollama


DB_PATH = "store.db"

SCHEMA = """
suppliers(id, name, country, lead_time_days)
products(id, name, category, price, stock, supplier_id)
customers(id, name, city, joined_date)
orders(id, product_id, customer_id, quantity, order_date, status)
status values: 'completed', 'pending', 'cancelled'

Notes:
- orders.status values: 'completed', 'pending', 'cancelled'
- products.supplier_id references suppliers.id
- orders.product_id references products.id
- orders.customer_id references customers.id
""".strip()

PROMPT = f"""You are a SQLite expert. Write ONE valid SQLite SELECT query for the question.
Return ONLY the SQL, no explanation, no markdown.

RULES:
- Return ONLY the raw SQL query, nothing else
- Only generate SELECT queries
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, PRAGMA, CREATE
- Use JOINs when data from multiple tables is needed
- Prefer clear column names and aliases
- Add LIMIT 20 unless the user explicitly asks for all results

Schema:
{SCHEMA}

Question: """



llm = Ollama(model="gemma4:e2b", request_timeout=300.0)


def get_sql(question: str) -> str:
    response = llm.complete(PROMPT + question)
    sql = str(response).strip()

    match = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1).strip()

    return sql


def validate_sql(sql: str) -> str:
    cleaned = sql.strip().strip(";")
    lowered = cleaned.lower()

    forbidden = ["insert", "update", "delete", "drop", "alter", "pragma", "create", "attach"]
    if not lowered.startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    for word in forbidden:
        if re.search(rf"\b{word}\b", lowered):
            raise ValueError(f"Forbidden SQL keyword detected: {word}")

    return cleaned + ";"


def run_sql(sql: str):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(sql)
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows


def get_schema() -> str:
    """Useful for viewing the exact schema of the store database before writing or checking SQL."""
    print("[tool] get_schema called")
    return SCHEMA

def summarize_rows(
    question: Annotated[str, "The original business question the rows are answering"],
    rows_json: Annotated[str, "A JSON string containing SQL result rows"],
) -> str:
    """Useful for turning raw SQL result rows into a short business-language explanation."""
    print(f"[tool] summarize_rows called with question= {question}")

    try:
        rows = json.loads(rows_json)
    except Exception as e:
        return f"Could not parse rows_json: {e}"

    if not isinstance(rows, list):
        return "The provided rows_json is not a JSON list."

    if len(rows) == 0:
        return "No rows were returned for that question."

    prompt = f"""You are a business analyst.
Summarize the SQL result rows into a short, accurate answer.

Rules:
- Base your answer only on the rows provided
- Do not invent tables, columns, or facts
- Be concise but clear
- If useful, mention the most important values directly

Question:
{question}

Rows:
{json.dumps(rows, indent=2)}

Answer:
"""
    response = llm.complete(prompt)
    return str(response).strip()


def ask_store_db(
    question: Annotated[str, "A natural-language question about the store database"]
) -> str:
    """Useful for answering questions about suppliers, products, customers, and orders in the store database."""
    print(f"[tool] ask_store_db called with question= {question}")

    try:
        sql = get_sql(question)
        print(f"[tool] generated sql = {sql}")

        sql = validate_sql(sql)
        rows = run_sql(sql)

        result = {
            "sql": sql,
            "row_count": len(rows),
            "rows": rows[:10],
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)
    

async def main():
    agent = FunctionAgent(
        tools=[get_schema, ask_store_db, summarize_rows],
        llm=llm,
        system_prompt=(
            "You are a store analytics assistant. "
            "Use get_schema when you need to inspect the database structure. "
             """
You are a store analytics assistant.

Workflow:
1. Use get_schema if the question depends on database structure.
2. Use ask_store_db to retrieve data.
3. Use summarize_rows to produce the final business answer.

Do not answer directly from ask_store_db output when summarize_rows is available.
"""
        ),
        streaming=False,
    )

    response = await agent.run(
        "Which product categories generate the highest completed order volume?"
    )
    print("\n[final answer]")
    print(str(response))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())