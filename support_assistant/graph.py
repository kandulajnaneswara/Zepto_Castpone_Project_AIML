import os
from typing import TypedDict, List

# Import LangGraph
try:
    from langgraph.graph import StateGraph, END

except Exception as e:
    print(f"[Fatal Error] Could not import langgraph: {e}")
    raise

# Import ChromaDB
try:
    import chromadb

except Exception as e:
    print(f"[Fatal Error] Could not import chromadb: {e}")
    raise

# Import Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer

except Exception as e:
    print(f"[Fatal Error] Could not import sentence-transformers: {e}")
    raise

# Import prompt builder
from prompt_template import build_prompt

# Configuration constants
docs_dir = "docs"
chroma_dir = "chromadb"
collection_name = "Zepto_policy_corpus"
embedding_model_name = "all-MiniLM-L6-v2"

# Policy Keywords
policy_keywords = ("delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours")

# Predefined response when user's question isn't identified as a policy question
direct_answer_fallback = ("I can only answer questions about Zepto policies right now.")

# Load the variables only when needed
# Variables initially contains nothing
_embedding_model = None
_chroma_collection = None

# Getting the embedding model
def get_embedding_model():
    global _embedding_model
    # Check whether _embedding_model is already loaded
    if _embedding_model is None:
        try:
            # Load the embedding model if its not loaded
            _embedding_model = SentenceTransformer(embedding_model_name)

        except Exception as e:
            print(f"[Error] Could not Load embedding model: {e}")
            _embedding_model = None

    return _embedding_model

# Getting the Chromadb collection
def get_chroma_collection():
    global _chroma_collection
    # Check whether connection to ChromaDB established or not
    if _chroma_collection is None:
        try:
            # Connect to the ChromaDB database stored in chromadb
            client = chromadb.PersistentClient(path= chroma_dir)
            # Get the 'Zepto_policy_corpus' collection if it already exists. (If not create it)
            _chroma_collection = client.get_or_create_collection(name= collection_name)

        except Exception as e:
            print(f"[Error] Could not connect to ChromaDB collection: {e}")
            _chroma_collection = None

    return _chroma_collection

# Setting MOCK_LLM variable
# MOCK_LLM unset or "1"  -> required, graded, fully offline mock baseline
# MOCK_LLM = "0"         -> optional, ungraded real-LLM extension
def is_mock_mode() -> bool:
    """MOCK_LLM unset or '1' => mock (graded baseline). '0' => real LLM (optional)"""
    mock_mode = os.environ.get("MOCK_LLM", "1") != "0"
    return mock_mode

# Define the information that travels through the graph nodes
class GraphState(TypedDict, total= False):
    # user's question
    query: str
    # Classification result (either policy_question or general_question)
    intent: str
    # Policy documents retrieved from ChromaDB
    retrieved_chunks: List[dict]
    # generated answer
    answer: str
    # IDs of the documents used
    sources: List[str]
    # How confident the system is in its answer
    confidence: float

# Node - 1: Classify Intent
def classify_intent(state: GraphState) -> GraphState:
    try:
        # Get the query from the state
        query = state.get("query", "") or ""
        # Convert query into lowercase
        query_lower = query.lower()

        # Mock classification
        if is_mock_mode():
            # Mock mode (graded baseline): no LLM call, Keyword based
            if any(keyword in query_lower for keyword in policy_keywords):
                intent = "policy_question"
            else:
                intent = "general_question"

        else:
            # MOCK_LLM = 0 (Call the LLM to classify)
            intent = classify_intent_with_llm(query)

        # Store the intent
        state["intent"] = intent
        return state

    except Exception as e:
        print(f"[Error] classify_intent failed: {e}")
        # For general_question, the graph can still produce a valid response rather than crashing
        state["intent"] = "general_question"
        return state


# Intended to call an actual LLM
def classify_intent_with_llm(query: str) -> str:
    try:
        # If the intent to call an actual LLM hasn't been implemented yet
        raise NotImplementedError("Actual LLM classification is an optional extension."
                                  "Set MOCK_LLM = 1 (unset) to use the graded baseline"
                                  "Set MOCK_LLM = 0 to use actual LLM")

    except Exception as e:
        print(f"[Error] classify_intent_with_llm failed: {e}")
        return "general_question"


# Node - 2: retrieve and answer for policy question
def retrieve_and_answer(state: GraphState) -> GraphState:
    try:
        # Retrieve the query
        # Get the user's question
        query = state.get("query", "") or ""
        # Search for top_3 relevant chunks
        retrieved_chunks = retrieve_top_chunks(query, top_k= 3)
        # Store in the state
        state["retrieved_chunks"] = retrieved_chunks

        # if ChromaDB returns nothing then sets the answer, sources and confidence
        if not retrieved_chunks:
            state["answer"] = ("I could not find relevant policy information to that question.")
            state["sources"] = []
            state["confidence"] = 0.0
            return state

        # Get the best chunk result
        top_chunk = retrieved_chunks[0]

        # If mock mode is enabled
        if is_mock_mode():
            # Mock mode (graded baseline): No LLM call
            # The top_chunk should take the first 200 characters
            top_chunk_snippet = top_chunk["text"][:200]
            # The state will return the retrieved text
            state["answer"] = f"Based on retrieved context: {top_chunk_snippet}"
            # Store the sources
            # initialize empty list
            source_list = []
            # Loop through the chunks
            for c in retrieved_chunks:
                # Extract and append the ID
                source_list.append(c["id"])
            # Assign to the state
            state["sources"] = source_list
            # Store the confidence
            state["confidence"] = 1.0

        else:
            # MOCK_LLM = 0 --> prompt the actual LLM
            # Initialize empty list
            text_parts = []
            # Loop through the chunks
            for c in retrieved_chunks:
                # Extract and append the text
                text_parts.append(c["text"])
            # Join the text segments
            context_text = "\n\n".join(text_parts)

            # Create LLM prompt
            prompt = build_prompt(query= query, retrieved_context= context_text)
            # Call the actual LLM for the final answer
            answer_text = call_actual_llm_with_validation(prompt, retrieved_chunks)
            # The state will return the answer text
            state["answer"] = answer_text
            # Store the sources
            # initialize empty list
            source_list = []
            # Loop through the chunks
            for c in retrieved_chunks:
            # Extract and append the ID
                source_list.append(c["id"])
            # Assign to the state
            state["sources"] = source_list
            # Store the confidence for actual LLM path
            state["confidence"] = 0.8

        return state

    except Exception as e:
        print(f"[Error] retrieve_and_answer failed: {e}")
        # the graph can still produce a valid response rather than crashing
        state["answer"] = "An error occured while retrieving policy information."
        state["sources"] = []
        state["confidence"] = 0.0

        return state

# Search ChromaDB for top relevant chunks
def retrieve_top_chunks(query: str, top_k: int= 3) -> List[dict]:
    try:
        # Get the embedding model
        model = get_embedding_model()
        # Get the ChromaDB connection
        collection = get_chroma_collection()
        # Check availability of model and collection
        if model is None or collection is None:
            print(f"[Error] Embedding model or Chroma collection unavailable")
            return []

        # Convert query to embeddings
        query_embedding = model.encode([query]).tolist()

        # Search ChromaDB results
        results = collection.query(query_embeddings= query_embedding, n_results= top_k)
        # Create empty chunk list
        chunks = []
        # Extract IDs and documents
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        # Pair the IDs and documents together
        for chunk_id, chunk_text in zip(ids, documents):
            chunks.append({"id": chunk_id, "text": chunk_text})

        return chunks

    except Exception as e:
        print(f"[Error] retrieve_top_chunks failed: {e}")
        return []

# Intended to call an actual LLM
def call_actual_llm_with_validation(prompt: str, retrieved_chunks: List[dict]) -> str:
    # MOCK_LLM=0, the required retry-on-schema-failure logic: retry up to 2 additional times with a corrective instruction before giving up.
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            # If the intent to call an actual LLM hasn't been implemented yet
            raise NotImplementedError("Actual LLM classification is an optional extension."
                                      "Set MOCK_LLM = 1 (unset) to use the graded baseline"
                                      "Set MOCK_LLM = 0 to use actual LLM")

        except Exception as e:
            print(f"[Error] call_actual_llm_with_validation attempt {attempt + 1}/{max_retries + 1} failed: {e}")
            if attempt == max_retries:
                return "[Error] Actual LLM answer generation failed after retries"

    return "[Error] Actual LLM answer generation failed."


# Node - 3: Direct answers for general question
def direct_answer(state: GraphState) -> GraphState:
    try:
        # Get the user's question
        query = state.get("query", "") or ""
        # If mock mode is enabled
        if is_mock_mode():
            # Mock mode (graded baseline): No LLM call
            state["answer"] = direct_answer_fallback

        else:
            # Optional MOCK_LLM = 0, prompt the actual LLM directly. No retrieval
            state["answer"] = call_actual_llm_direct(query)

        # Empty for general question, per the schema
        state["sources"] =[]
        state["confidence"] = 1.0
        return state

    except Exception as e:
        print(f"[Error] direct_answer failed: {e}")
        state["answer"] = direct_answer_fallback
        state["sources"] = []
        state["confidence"] = 0.0
        return state

# Send a general question directly to an LLM
def call_actual_llm_direct(query: str) -> str:
    # Optional MOCK_LLM=0 extension placeholder for un-grounded direct answers
    try:
        raise NotImplementedError("Actual LLM classification is an optional extension."
                                  "Set MOCK_LLM = 1 (unset) to use the graded baseline"
                                  "Set MOCK_LLM = 0 to use actual LLM")

    except Exception as e:
        print(f"[Error] call_actual_llm_direct failed: {e}")
        return direct_answer_fallback

# Conditional routing function (does not itself depend on MOCK_LLM)
def route_after_classification(state: GraphState) -> str:
    try:
        # Get the intent from state
        intent = state.get("intent", "general_question")
        if intent == "policy_question":
            return "retrieve_and_answer"
        return "direct_answer"
    
    except Exception as e:
        print(f"[ERROR] route_after_classification failed: {e}")
        return "direct_answer"

# Build the LangGraph
def build_graph():
    try:
        # Create LangGraph whose shared state follows Graphstate
        workflow = StateGraph(GraphState)

        # Add nodes
        # Register a node and tell langgraph to execute when reaches the node
        workflow.add_node("classify_intent", classify_intent)
        workflow.add_node("retrieve_and_answer", retrieve_and_answer)
        workflow.add_node("direct_answer", direct_answer)

        # Set the starting node
        workflow.set_entry_point("classify_intent")

        # Conditional edges (call route_after_classification() to decide which node should execute next)
        workflow.add_conditional_edges("classify_intent",route_after_classification,
                                       {"retrieve_and_answer": "retrieve_and_answer",
                                        "direct_answer": "direct_answer",
                                        })

        # Connect nodes to END 
        workflow.add_edge("retrieve_and_answer", END)
        workflow.add_edge("direct_answer", END)

        # Compile the graph
        return workflow.compile()
    
    except Exception as e:
        print(f"[Fatal Error] build_graph failed: {e}")
        raise
 
 
# Compiled graph, built once at import time and reused across requests.
try:
    compiled_graph = build_graph()
except Exception as e:
    print(f"[Fatal Error] Could not build the LangGraph graph: {e}")
    compiled_graph = None
 
# The main entry point
def run_graph(query: str) -> dict:
    """Entry point used by main.py: runs the graph for a single query."""
    try:
        # Check the graph 
        if compiled_graph is None:
            raise RuntimeError("Graph was not built successfully.")
        # Create initial state
        initial_state: GraphState = {"query": query}
        # Execute the graph
        final_state = compiled_graph.invoke(initial_state)
        # Return only the important information
        return {"answer": final_state.get("answer", direct_answer_fallback),
                "intent": final_state.get("intent", "general_question"),
                "sources": final_state.get("sources", []),
                "confidence": final_state.get("confidence", 0.0)
                }
                
    except Exception as e:
        print(f"[Error] run_graph failed: {e}")
        return {"answer": "An internal error occurred while processing your question.",
                "intent": "general_question",
                "sources": [],
                "confidence": 0.0
                }