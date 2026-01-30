import os
import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel
from typing import Dict, List, Optional

# --- 1. Setup & Config ---
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(url, key)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SpecialtyInput(BaseModel):
    id: str
    name: str
    rank: int
    normalizedScore: int


class RegionInput(BaseModel):
    id: str
    name: str
    type: str
    coefficient: int


class RegionSpecialtyInput(BaseModel):
    regionId: str
    specialtyId: str
    historical_deficit: int
    request_uz_amount: int


class AllocationRequest(BaseModel):
    budget: int  # The hard limit (e.g., 2500)
    specialties: Optional[List[SpecialtyInput]] = None
    regions: Optional[List[RegionInput]] = None
    region_specialty_inputs: Optional[List[RegionSpecialtyInput]] = None


# --- 2. Helper: The Priority Calculator ---
def calculate_priority_score(
    region_name: str,
    specialty_name: str,
    specialties_map: Dict[str, int],
    regions_map: Dict[str, int]
) -> int:
    """
    Combines Specialty Rank and Region Score into a single 'Weight' for the graph.
    Formula: (SpecialtyScore * 0.5) + (RegionScore * 0.5)
    """
    spec_score = specialties_map.get(specialty_name, 50)
    region_score = regions_map.get(region_name, 50)
    
    final_score = (spec_score * 0.5) + (region_score * 0.5)
    return int(final_score)

# --- 3. The Core Algorithm ---
def run_allocation_logic(payload: AllocationRequest):
    total_grants = payload.budget
    
    # Build lookup maps from the payload
    specialties_map: Dict[str, int] = {}
    regions_map: Dict[str, int] = {}
    
    if payload.specialties:
        for s in payload.specialties:
            specialties_map[s.name] = s.normalizedScore
    
    if payload.regions:
        for r in payload.regions:
            regions_map[r.name] = r.coefficient
    
    # Build region_specialty lookup: key = "regionId|specialtyId"
    region_specialty_map: Dict[str, RegionSpecialtyInput] = {}
    if payload.region_specialty_inputs:
        for rsi in payload.region_specialty_inputs:
            key = f"{rsi.regionId}|{rsi.specialtyId}"
            region_specialty_map[key] = rsi
    
    # Fetch demands from Supabase demand_requests table
    # Table has: region, specialty, historical_deficit, request_uz_amount, priority_score
    demand_response = supabase.table("demand_requests").select("*").execute()
    requests = demand_response.data
    
    if not requests:
        return []
    
    # B. Build the Graph
    G = nx.DiGraph()
    
    # NODE 0: The Ministry's Wallet
    G.add_node("Ministry_Source", demand=-total_grants) 
    
    total_needed_slots = 0
    valid_requests = []
    
    for req in requests:
        req_id = req.get('id', f"{req['region']}_{req['specialty']}")
        region_name = req['region']
        specialty_name = req['specialty']
        
        # Use historical_deficit + request_uz_amount as the "need"
        historical_deficit = req.get('historical_deficit', 0) or 0
        request_uz_amount = req.get('request_uz_amount', 0) or 0
        slots_needed = historical_deficit + request_uz_amount
        
        if slots_needed < 1:
            continue
            
        total_needed_slots += slots_needed
        
        # Get priority score - either from the table or calculate from payload
        priority_score = req.get('priority_score', 0)
        if priority_score == 0 and (specialties_map or regions_map):
            priority_score = calculate_priority_score(
                region_name, specialty_name, specialties_map, regions_map
            )
        
        # COST LOGIC: NetworkX minimizes cost. We want to maximize Priority.
        cost = 1000 - priority_score
        
        # Add Node for this specific Request
        node_id = f"{region_name}|{specialty_name}"
        G.add_node(node_id, demand=slots_needed)
        
        # Add Edge from Ministry to Request
        G.add_edge("Ministry_Source", node_id, capacity=slots_needed, weight=cost)
        
        valid_requests.append({
            "node_id": node_id,
            "region_name": region_name,
            "specialty_name": specialty_name,
            "slots_needed": slots_needed,
            "priority_score": priority_score
        })

    if not valid_requests:
        return []

    # C. Handle Overflow (If Requests > Grants)
    if total_needed_slots > total_grants:
        fake_grants = total_needed_slots - total_grants
        G.add_node("Dummy_Source", demand=-fake_grants)
        for vr in valid_requests:
            G.add_edge("Dummy_Source", vr["node_id"], capacity=vr["slots_needed"], weight=999999)

    # D. Solve
    try:
        flow_dict = nx.min_cost_flow(G)
    except nx.NetworkXUnfeasible:
        # If the graph is infeasible, return empty
        return []
    
    # E. Parse Results
    results = []
    for vr in valid_requests:
        node_id = vr["node_id"]
        allocated_real = flow_dict.get("Ministry_Source", {}).get(node_id, 0)
        
        if allocated_real > 0:
            results.append({
                "region_id": vr["region_name"],
                "region_name": vr["region_name"],
                "specialty_id": vr["specialty_name"],
                "specialty_name": vr["specialty_name"],
                "allocated_quota": allocated_real
            })
            
    return results

# --- 4. Endpoint ---
@app.post("/run-allocation")
async def trigger_allocation(payload: AllocationRequest):
    try:
        results = run_allocation_logic(payload)
        
        # Optionally save results to allocations table

        if results:
            from datetime import datetime
            now = datetime.utcnow().isoformat()
            storage_results = [
                {
                    "request_id": r.get("request_id") or r.get("node_id") or f"{r['region_name']}|{r['specialty_name']}",
                    "allocated_count": r["allocated_quota"],
                    "allocation_date": now
                }
                for r in results
            ]
            # Clear old results and insert new
            supabase.table("allocations").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            supabase.table("allocations").insert(storage_results).execute()

            # Delete all rows from demand_requests after allocation is done
            supabase.table("demand_requests").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()

        return {
            "status": "success",
            "allocations": results,
            "grants_used": sum(r['allocated_quota'] for r in results),
            "requests_funded": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "ok"}