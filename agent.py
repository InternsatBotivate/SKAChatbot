import os
from typing import List, TypedDict
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import StateGraph, END
from sqlalchemy import create_engine
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()


# --- Pydantic Models for Intent Classification ---
class DatabaseQuery(BaseModel):
    """The user is asking a question that requires a database query, or can be solved by an sql query"""

    pass


class Conversation(BaseModel):
    """The user is greeting, making a small talk or asking a general knowledge question not related to database or cannot be handled with sql."""

    pass


# --- Agent State Definition ---
class AgentState(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    query: str
    result: str
    answer: str
    retries: int
    intent: str


db_uri = os.getenv("DATABASE_URI")
engine = create_engine(db_uri)

db = SQLDatabase(engine=engine)
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
execute_query_tool = QuerySQLDatabaseTool(db=db)


@tool
def get_current_datetime() -> str:
    """Returns today's date and the current time in ISO 8601 format."""
    return datetime.now().isoformat()


# --- Graph Nodes ---
def classify_intent_node(state: AgentState):
    """Classifies the user's question by forcing the LLM to call a specific tool."""
    print("--- Classifying Intent (with Function Calling) ---")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an intent classifier. Call the appropriate tool based on the user's last message.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    tools = [DatabaseQuery, Conversation]
    llm_with_tools = llm.bind_tools(tools)
    runnable = prompt | llm_with_tools
    ai_message = runnable.invoke(
        {
            "question": state["question"],
            "chat_history": state.get("chat_history", []),
        }
    )
    intent = (
        "Conversation"
        if not ai_message.tool_calls
        else ai_message.tool_calls[0]["name"]
    )
    print(f"Intent: {intent}")
    return {"intent": intent}


def handle_conversation_node(state: AgentState):
    """Creates natural conversation with the user."""
    print("--- Handling Conversation ---")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a friendly assistant solves user's database related queries, Diya, for Mr.Abhishek, you refer him as Mr. Abhishek. Reply to the user politely with a short relevant relevant response. Reply in English or Hindi based on user's question. All currencies are in Rupees until mentioned other wise. Greet user according to current time, i.e., 'Good Morning', 'Good Evening', etc. when needed. Don't just greet on every response.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    tools = [get_current_datetime]
    agent_runnable = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent_runnable, tools=tools, verbose=True)
    result = agent_executor.invoke(
        {"question": state["question"], "chat_history": state.get("chat_history", [])}
    )
    print(f"Final Answer: {result['output']}")
    return {"answer": result["output"]}


def generate_query_node(state: AgentState):
    """Takes the user's question, generates a SQL query, and adds it to the state."""
    print("--- Generating SQL Query ---")

    system_prompt = """You are an AI expert in writing PostgreSQL queries.
    Given a user question and conversation history, create a syntactically correct PostgreSQL query.
    The query should be in its simplest form.
    The query should fullfill user's query.
    The query should work on the given schema.
    {schema}

    --- Querying Rules ---
    1.  **CRITICAL `UNION` RULE:** When using `UNION` or `UNION ALL`, you **MUST NOT** use `SELECT *`. The tables have different columns and this will cause an error.
    2.  **HOW TO FIX `UNION`:** You must explicitly list the columns to select. Identify a set of common, meaningful columns (e.g., "Task", "Status", "Assignee", "Priority", "Due_Date"). For tables that are missing one of these columns, you **MUST** select `NULL` and cast it to the appropriate type, aliasing it to the common column name. For example: `SELECT "Task", "Status", NULL::text AS "Assignee" FROM "Checklist"`.
    3. Use advanced matching techniques, to respond to more flexible queries.

    --- Database Descriptions ---
    - When a user asks about "tasks" or "kaam", they are referring to entries where a table has fields relevant to tasks, like "TaskID", or "Task Description". You MUST query one of given tables that is related to tasks. DO NOT invent or query a non-existent table named "tasks".
    - When a user asks about "po", they are usually referring to entries where a table has fields relevant to Purchase Orders like "Quantity", "PO Number".
    - When a user asks about "orders", they are usually referring to entries where a table has fields relevant to Orders like "Dispatch Quantity", "Order Number", "Transporter Name" and "Brand Name".
    - When a user asks about "Employee", they are usually referring to entries where a table has fields relevant to Employee Details like "Designation", "Name as per Aadhar", "Mobile Number" and "SKA-Joining ID".
    - When a user asks about "Store OUT", they are usually referring to entries where a table has fields relevant to Store OUT like "Store Out Number", "Indentor Name", "Department", "Area", "Product Name ", "Quantity" and "Amount".
    - When a user asks about "Store IN", they are usually referring to entries where a table has fields relevant to Store IN like "Indent Number", "What", "Product Name", "Vendor Name", "Rate ", "Quantity" and "Payment Term".
    - When a user asks about "Souda", they are usually referring to entries where a table has fields relevant to Souda like "Sauda Number", "Indentor Name", "Date Of Sauda", "Area", "Broker Name", "Party Name", "Delear Name", "Rate", "Order Quantity (Ton)", "Total Dispatch Qty", "Pending Qty", "Order Cancel Qty", "Sauda Status" and "Brand Name".
    - When a user asks about "INVOICE/invoice", they are usually referring to entries where a table has fields relevant to INVOICE  like "Unique No", "Order Number", "Party Name", "Sauda No.", "Do No.", "Bill Date", "Bill No.", "Bill Image", "Delivery Term", "Tramsporter Name", "Pending Qty", "Vehicle No.", "LR-Number", "Bill Status", "Size", "Section", "Qty", "Rate", "Customer Discount" and "UDAAN/VIDHAN". 
    - When a user refers to sheets they are actually talking about tables.

    - When user asks for report, they are usually demanding a sumamry of the data relevant to the context along side sample rows.The summary should include data like, total rows, total completed, total pending, total amount, total amount pending, etc. this should be specific to the sheet in question
    - The database deals with several types of data: Tasks, Purchase Orders, Sales, Production, Inventory, Finance, Employees, and Enquiries.
    - Here is a list of tables that fall in each category:
        - **Tasks**
            - **Checklist**: contains details of recurring tasks.
            - **Delegation**: contains details of delegation tasks (doer-wise, name-wise, giver-wise, department-wise).
        - **Purchase Orders**
            - **PO Pending**: contains purchase order information, including products, quantities, rates, total amounts, and current fulfillment status.
            - **Purchase Receipt**: contains material that has been received in the plant.
        - **Inventory**
            - **Store OUT**: records materials issued from the store.
            - **Store IN**: records materials received into the store.
        - **Employees**
            -**Active Employee Details**: contains detailed information about active company employees, including joining ID, name, father's name, date of joining, designation, address, date of birth, gender, mobile number, bank account details, email, qualification, and department.
        - **Sales**
            - **Souda**: contains details of sales orders including broker name, party name, rate, souda/sauda quantity, pending quantity, sauda/souda status, and brand name. It may be asked as sauda or souda
            - **INVOICE **: contains details of invoices including party name, order number, bill number, bill date, transporter name, vehicle number, delivery term, brand name (UDAAN/VIDHAN), quantity, rate, and bill status.
    - Do not take table as there names suggest. Use the above guide to get the relevant table.
    - When user asks query based on some identity, that can be present in other tables, and there is no previous context for choosing a table, give data, or all occurances.
    ------------------------
    
    --- Data Dictionary ---
    - The "Status" column: 'Completed', 'Yes', 'Done' all mean the task is complete. NULL/Empty, 'Not Complete', 'Pending' may mean the task is pending. Basically anything not complete is pending.
    - The "Priority" column: 'High', 'Urgent', 'H' all mean high priority. 'Low' and 'L' mean low priority.
    -----------------------

    - **IMPORTANT:** Only return the SQL query. Do not add any other text or explanation.
    - **IMPORTANT:** If a table or column name contains a space or is a reserved keyword, you MUST wrap it in double quotes. For example: "Task Description".
    - **IMPORTANT:** Use the columns provided in the schema, if user mention a column that is not in schema, try to find the closest relevant column in the schema.
    """

    if "Error:" in state.get("result", ""):
        system_prompt += """
        \n---
        The previous query you wrote failed. Here is the error message: 
        {error}
        You likely violated the CRITICAL UNION RULE. Do not use SELECT *. Instead, select specific columns and use NULL placeholders for columns that do not exist in some tables. Please write a new corrected SQL query.
        ---
        """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    llm = ChatOpenAI(model="o4-mini")
    runnable = prompt | llm
    raw_query = runnable.invoke(
        {
            "question": state["question"],
            "chat_history": state.get("chat_history", []),
            "schema": db.get_table_info(),
            "error": state.get("result", ""),
        }
    ).content
    sql_query = raw_query.strip().replace("```sql", "").replace("```", "").strip()
    print(f"Generated Query: {sql_query}")
    retries = state.get("retries", 0)
    return {"query": sql_query, "retries": retries + 1}


def execute_query_node(state: AgentState):
    """Executes the SQL query and returns the result."""
    print("--- Executing SQL Query ---")
    query = state["query"]
    result = execute_query_tool.invoke(query)
    print(f"Query Result: {result}")
    return {"result": result}


def summarize_result_node(state: AgentState):
    """Takes the query result and creates a natural language answer."""
    print("--- Summarizing Result ---")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, Diya. Your job is to answer the user's question in concise manner, based on the data provided, which should be easy and fast to read, with markup and lists and tables if needed. Only reply in English or Hindi based on user's question. Do not give any clarification about how you got the result. When datais too big,for examples in reports and big data fetches, give summary like Number of Pending, Number of Completed, Total number of rows, and other stuff like total amount, total amount to be paid, etc. Basically a summary of the data. Never give more than 20 rows of data, whether that be in list or tables.",
            ),
            (
                "human",
                """Based on the user's question: "{question}"
        The following SQL query was generated: "{query}"
        And here is the result from the database: "{result}"
        Please provide a clear, natural language answer.
        Normalize table names, and remove _ in between words.
        """,
            ),
        ]
    )
    runnable = prompt | llm
    answer = runnable.invoke(
        {
            "question": state["question"],
            "query": state["query"],
            "result": state["result"],
        }
    ).content
    print(f"Final Answer: {answer}")
    return {"answer": answer}


def handle_error_node(state: AgentState):
    """Handles cases where the agent gives up after multiple retries."""
    print("--- 😩 Agent failed after multiple retries ---")
    error_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, Diya, for a SQL database. The query you generated failed multiple times. Just say to the user that you couldn't find the answer. Resturn small easy to read with markup response. All currencies are in Rupees until mentioned other wise.",
            ),
            (
                "human",
                """The user asked: "{question}"
        Your last attempted SQL query was: "{query}"
        It failed with the error: "{error}"
        Please provide a clear, natural language response apologizing for the failure and offering advice.""",
            ),
        ]
    )
    runnable = error_prompt | llm
    answer = runnable.invoke(
        {
            "question": state["question"],
            "query": state["query"],
            "error": state.get("result", "Unknown error"),
        }
    ).content
    print(f"Final Answer: {answer}")
    return {"answer": answer}


# --- Conditional Edges ---
def decide_intent_path(state: AgentState):
    return (
        "generate_query"
        if state["intent"] == "DatabaseQuery"
        else "handle_conversation"
    )


def decide_result_status(state: AgentState):
    if "Error:" in state["result"]:
        print("--- Query failed. Looping back to generate a new query. ---")
        return "handle_error" if state["retries"] > 7 else "generate_query"
    return "summarize_result"


# --- Build the Graph ---
graph = StateGraph(AgentState)
graph.add_node("classify_intent", classify_intent_node)
graph.add_node("handle_conversation", handle_conversation_node)
graph.add_node("generate_query", generate_query_node)
graph.add_node("execute_query", execute_query_node)
graph.add_node("summarize_result", summarize_result_node)
graph.add_node("handle_error", handle_error_node)

graph.set_entry_point("classify_intent")
graph.add_conditional_edges(
    "classify_intent",
    decide_intent_path,
    {"generate_query": "generate_query", "handle_conversation": "handle_conversation"},
)
graph.add_edge("generate_query", "execute_query")
graph.add_conditional_edges(
    "execute_query",
    decide_result_status,
    {
        "generate_query": "generate_query",
        "summarize_result": "summarize_result",
        "handle_error": "handle_error",
    },
)
graph.add_edge("handle_conversation", END)
graph.add_edge("handle_error", END)
graph.add_edge("summarize_result", END)

agent = graph.compile()

# --- Run the Agent ---
if __name__ == "__main__":
    # Example question
    initial_state = {
        "question": "How many tasks are pending in the delegation list?",
        "chat_history": [],
    }
    final_state = agent.invoke(initial_state)
    print("\n--- Final State ---")
    print(final_state["answer"])
