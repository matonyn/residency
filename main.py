import os
import networkx as nx
from fastapi import FastAPI, HTTPException
from supabase import create_client, Client
from pydantic import BaseModel

# 1. Setup Supabase Connection
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_SERVICE_KEY") # Use Service Key to bypass RLS
supabase: Client = create_client(url, key)

app = FastAPI()

# 2. Define the Algorithm
def run_allocation_logic(total_budget: int):
    # A. Fetch Data from Supabase
    response = supabase.table("demand_requests").select("*").execute()
    requests = response.data # List of dicts
    
    # B. Build the Graph
    G = nx.DiGraph()
    G.add_node("Ministry_Source", demand=-total_budget)
    
    for req in requests:
        req_id = req['id']
        slots = req['slots_needed']
        # Cost logic: High priority score = Low cost (Negative weight)
        cost = -req['priority_score'] 
        
        G.add_node(req_id, demand=slots)
        G.add_edge("Ministry_Source", req_id, capacity=slots, weight=cost)

    # C. Solve (Min Cost Flow)
    # Note: NetworkX requires supply == demand. 
    # If Budget < Total Needs, we add a "Trash" node to absorb the deficit.
    total_needed = sum(r['slots_needed'] for r in requests)
    if total_needed > total_budget:
        deficit = total_needed - total_budget
        G.add_node("Unfunded_Sink", demand=deficit)
        G.add_edge("Ministry_Source", "Unfunded_Sink", capacity=deficit, weight=0)
    
    flow_dict = nx.min_cost_flow(G)
    
    # D. Format Results
    results = []
    for req in requests:
        req_id = req['id']
        allocated = flow_dict["Ministry_Source"].get(req_id, 0)
        if allocated > 0:
            results.append({
                "request_id": req_id,
                "allocated_count": allocated
            })
            
    return results

# 3. The API Endpoint
class AllocationRequest(BaseModel):
    budget: int

@app.post("/run-allocation")
async def trigger_allocation(payload: AllocationRequest):
    try:
        results = run_allocation_logic(payload.budget)
        
        # E. Write back to Supabase
        if results:
            supabase.table("allocations").insert(results).execute()
            
        return {"status": "success", "allocated_count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))