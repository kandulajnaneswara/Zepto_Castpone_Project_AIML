try:
    # Import FastAPI
    from fastapi import FastAPI, HTTPException

except Exception as e:
    print(f"[Fatal Error] Could not import FastAPI: {e}")
    raise

# Import the request and response schemas 
from schema import AskRequest, AskResponse
# Import graph
from graph import run_graph

# Create a fastAPI application
app = FastAPI(title="Zepto Support Assistant",
              description="A small RAG-based support assistant over Zepto's policy corpus.",
              version="1.0.0")
 

# Create GET endpoint(decorator) 
@app.get("/")
# Health check endpoint
def health_check():
    """Simple health-check endpoint."""
    try:
        # API returns JSON
        return {"status": "ok", "service": "zepto-support-assistant"}
    
    except Exception as e:
        print(f"[Error] health_check failed: {e}")
        return {"status": "error"}
 
# Create POST endpoint 
@app.post("/ask", response_model=AskResponse)
# Ask endpoint
def ask(request: AskRequest) -> AskResponse:
    """
    Accepts {"query": str} and returns the validated AskResponse ({"answer": str, "sources": [str], "confidence": float}).
    """
    try:
        # Validate the query
        if not request.query or not request.query.strip():
            # If query is empty, API returns an HTTP 400 error
            raise HTTPException(status_code=400, detail="'query' must not be empty.")

        # Send the question to the LangGraph 
        result = run_graph(request.query)
 
        try:
            # Validate Response against the Pydantic schema before returning.
            response = AskResponse(**result)
            return response
        
        except Exception as validation_error:
            print(f"[Error] Response failed schema validation: {validation_error}")
            # Return a clearly marked, schema-valid error response instead of crashing the request.
            return AskResponse(answer="An error occurred while validating the response.",
                               sources=[],
                               confidence=0.0)
        
    except HTTPException:
        raise

    except Exception as e:
        print(f"[Error] /ask failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")