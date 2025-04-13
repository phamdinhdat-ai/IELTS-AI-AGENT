from langchain.tools import Tool, tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper # Example external tool
# Assume vector_db and embedding functions are accessible or passed
# In a real app, these would be initialized based on customer config

# Placeholder for vector DB access (needs proper initialization per customer)
# This is tricky - tool functions ideally shouldn't depend on global state loaded per request.
# A better pattern involves creating tool *instances* per request with the right DB connection.
# For simplicity here, we assume a way to get the customer-specific retriever.

# Global or passed-in retriever placeholder - MUST BE REPLACED WITH DYNAMIC LOADING
customer_retriever_placeholder = None

@tool("customer_knowledge_base")
def customer_knowledge_base_tool(query: str) -> str:
    """
    Searches and returns information from the customer's specific knowledge base.
    Use this for questions about internal policies, procedures, product specs, etc.
    Input should be a clear question about the internal knowledge.
    """
    global customer_retriever_placeholder # BAD PRACTICE - Needs refactoring
    if customer_retriever_placeholder is None:
        return "Error: Knowledge base retriever not initialized for this customer."
    try:
        docs = customer_retriever_placeholder.invoke(query)
        # Format the results concisely for the agent
        formatted_docs = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in docs])
        if not formatted_docs:
            return "No relevant information found in the knowledge base."
        return formatted_docs
    except Exception as e:
        return f"Error searching knowledge base: {e}"

@tool("calculator")
def calculator_tool(expression: str) -> str:
    """
    Calculates the result of a mathematical expression.
    Input should be a valid mathematical expression (e.g., '2 + 2', 'sqrt(16)').
    """
    try:
        # WARNING: eval is dangerous with untrusted input. Use a safer math parser in production!
        # Example using 'numexpr' or 'sympy' would be safer.
        result = eval(expression, {"__builtins__": None}, {"sqrt": lambda x: x**0.5})
        return f"The result of '{expression}' is {result}"
    except Exception as e:
        return f"Error calculating expression '{expression}': {e}. Please provide a valid math expression."

# Example external tool (requires pip install duckduckgo-search)
search = DuckDuckGoSearchAPIWrapper()
search_tool = Tool(
    name="internet_search",
    func=search.run,
    description="Useful for answering questions about current events or topics not found in the customer knowledge base. Input should be a search query."
)

# --- Tool Registry ---
# Map tool names (used in config) to actual tool objects
available_tools = {
    "customer_knowledge_base": customer_knowledge_base_tool,
    "calculator": calculator_tool,
    "internet_search": search_tool, # Uncomment if using search
}

def get_customer_tools(tool_names: list) -> list:
    """ Returns a list of Tool objects based on allowed names """
    tools = []
    for name in tool_names:
        if name in available_tools:
            tools.append(available_tools[name])
        else:
            print(f"Warning: Tool '{name}' configured but not found in available_tools.")
    return tools