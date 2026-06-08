from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


field_descriptions = {
    "translated_query" : """Full rephrased version of the user query using precise 
                        LangChain/LangGraph/LangSmith terminology. Resolves all pronouns 
                        and vague references using chat history.""",

    "sub_queries"      : """List of self-contained sub-queries, each carrying full semantic 
                        meaning independently. Single-topic queries produce a list of one. 
                        Maximum 4 sub-queries. Each must name the specific framework 
                        (LangChain / LangGraph / LangSmith) it applies to.""",

    "needs_retrieval"  : """True if the query requires fetching documentation context. 
                        False for questions answerable from general LLM knowledge 
                        (e.g. 'what is LangChain?', 'who made LangGraph?').""",

    "guardrail"        : """Rejection reason string if the query must not be processed. 
                        None otherwise. Triggered for: non-LangChain/LangGraph/LangSmith 
                        topics, API Reference questions, or gibberish input."""
    }

class QueryPlanner(BaseModel):
    """
    For structured llm output from query-planner node.
    """
    translated_query : str        = Field(description= field_descriptions["translated_query"])
    sub_queries      : list[str]  = Field(description= field_descriptions["sub_queries"])
    needs_retrieval  : bool       = Field(description= field_descriptions["needs_retrieval"])
    guardrail        : str | None = Field(description= field_descriptions["guardrail"])



class AgentGraphState(TypedDict):
    """
    The schema of state that holds information
    throughout graph's execution. 
    """
    user_input         : str
    query_plan         : QueryPlanner | None
    retrieved_context  : list[dict]
    context_sufficient : bool
    response           : str
    chat_history       : Annotated[List, add_messages]
    status_mssg        : list[str]