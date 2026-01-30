import os
import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

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
    # Optional: weights for different factors
    specialty_weight: Optional[float] = 0.4
    region_weight: Optional[float] = 0.3
    deficit_weight: Optional[float] = 0.3


# --- 2. Improved Priority Calculator ---
def calculate_priority_score(
    region_name: str,
    specialty_name: str,
    historical_deficit: int,
    request_uz_amount: int,
    specialties_map: Dict[str, int],
    regions_map: Dict[str, int],
    specialty_weight: float = 0.4,
    region_weight: float = 0.3,
    deficit_weight: float = 0.3
) -> float:
    """
    Multi-factor priority calculation:
    - Specialty importance (normalized 0-100)
    - Region government priority (coefficient 0-100)
    - Historical deficit urgency (normalized by max deficit)
    
    Returns a score between 0-100
    """
    # Get base scores (default to 50 if not found)
    spec_score = specialties_map.get(specialty_name, 50)
    region_score = regions_map.get(region_name, 50)
    
    # Normalize deficit score (assuming max deficit around 100)
    # Higher deficit = higher priority
    deficit_score = min(100, (historical_deficit / 100.0) * 100) if historical_deficit > 0 else 0
    
    # Weighted combination
    final_score = (
        spec_score * specialty_weight +
        region_score * region_weight +
        deficit_score * deficit_weight
    )
    
    return final_score


# --- 3. Enhanced Core Algorithm ---
def run_allocation_logic(payload: AllocationRequest):
    """
    Min-cost max-flow allocation algorithm using NetworkX.
    
    Key improvements:
    - Better priority calculation with multiple factors
    - Clearer graph structure
    - Better handling of over-subscription
    - More detailed logging
    """
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
    
    # Fetch demands from Supabase demand_requests table
    demand_response = supabase.table("demand_requests").select("*").execute()
    requests = demand_response.data
    
    if not requests:
        print("No demand requests found in database")
        return []
    
    # --- Build the Network Flow Graph ---
    G = nx.DiGraph()
    
    # Source node: Ministry's scholarship budget
    G.add_node("SOURCE", demand=-total_grants)
    
    # Sink node: Absorbs all allocated scholarships
    G.add_node("SINK", demand=total_grants)
    
    total_requested_slots = 0
    valid_requests = []
    
    # Process each demand request
    for req in requests:
        region_name = req['region']
        specialty_name = req['specialty']
        
        # Calculate total need
        historical_deficit = req.get('historical_deficit', 0) or 0
        request_uz_amount = req.get('request_uz_amount', 0) or 0
        slots_needed = historical_deficit + request_uz_amount
        
        if slots_needed < 1:
            continue
            
        total_requested_slots += slots_needed
        
        # Calculate priority score
        priority_score = req.get('priority_score', 0)
        if priority_score == 0 or (specialties_map or regions_map):
            priority_score = calculate_priority_score(
                region_name=region_name,
                specialty_name=specialty_name,
                historical_deficit=historical_deficit,
                request_uz_amount=request_uz_amount,
                specialties_map=specialties_map,
                regions_map=regions_map,
                specialty_weight=payload.specialty_weight,
                region_weight=payload.region_weight,
                deficit_weight=payload.deficit_weight
            )
        
        # Create unique node ID
        node_id = f"{region_name}|{specialty_name}"
        
        # Add intermediate node for this request (with no demand - just a pass-through)
        G.add_node(node_id, demand=0)
        
        # EDGE 1: SOURCE -> Request Node
        # Cost is inverted priority (lower cost = higher priority)
        # Capacity is the maximum we can allocate to this request
        cost = 1000 - int(priority_score)
        G.add_edge("SOURCE", node_id, capacity=slots_needed, weight=cost)
        
        # EDGE 2: Request Node -> SINK
        # Zero cost, same capacity
        G.add_edge(node_id, "SINK", capacity=slots_needed, weight=0)
        
        valid_requests.append({
            "node_id": node_id,
            "region_name": region_name,
            "specialty_name": specialty_name,
            "slots_needed": slots_needed,
            "priority_score": priority_score,
            "historical_deficit": historical_deficit,
            "request_uz_amount": request_uz_amount,
            "request_id": req.get('id', node_id)
        })

    if not valid_requests:
        print("No valid requests after filtering")
        return []

    # Log allocation scenario
    print(f"Total Budget: {total_grants}")
    print(f"Total Requested: {total_requested_slots}")
    print(f"Valid Requests: {len(valid_requests)}")
    
    if total_requested_slots < total_grants:
        print(f"WARNING: Under-subscription by {total_grants - total_requested_slots} slots")
    elif total_requested_slots > total_grants:
        print(f"Over-subscription: {total_requested_slots - total_grants} slots will not be funded")

    # --- Solve the Min-Cost Flow Problem ---
    try:
        flow_dict = nx.min_cost_flow(G)
    except nx.NetworkXUnfeasible:
        print("ERROR: Graph is infeasible - cannot satisfy flow constraints")
        return []
    except Exception as e:
        print(f"ERROR: Flow calculation failed: {str(e)}")
        return []
    
    # --- Parse Results ---
    results = []
    total_allocated = 0
    
    for vr in valid_requests:
        node_id = vr["node_id"]
        # Flow from SOURCE to this request node
        allocated = flow_dict.get("SOURCE", {}).get(node_id, 0)
        
        if allocated > 0:
            total_allocated += allocated
            results.append({
                "request_id": vr["request_id"],
                "region_id": vr["region_name"],
                "region_name": vr["region_name"],
                "specialty_id": vr["specialty_name"],
                "specialty_name": vr["specialty_name"],
                "allocated_quota": allocated,
                "requested_quota": vr["slots_needed"],
                "priority_score": round(vr["priority_score"], 2),
                "historical_deficit": vr["historical_deficit"],
                "request_uz_amount": vr["request_uz_amount"],
                "fulfillment_rate": round((allocated / vr["slots_needed"]) * 100, 1)
            })
    
    # Sort by priority score (highest first)
    results.sort(key=lambda x: x["priority_score"], reverse=True)
    
    print(f"Total Allocated: {total_allocated} / {total_grants}")
    print(f"Requests Funded: {len(results)} / {len(valid_requests)}")
    
    return results


# --- 4. Enhanced Endpoint ---
@app.post("/run-allocation")
async def trigger_allocation(payload: AllocationRequest):
    """
    Run the scholarship allocation algorithm and save results.
    
    Returns allocation results with detailed statistics.
    """
    try:
        # Run the allocation algorithm
        results = run_allocation_logic(payload)
        
        if not results:
            return {
                "status": "warning",
                "message": "No allocations made - check demand_requests table",
                "allocations": [],
                "grants_used": 0,
                "requests_funded": 0
            }
        
        # Save results to allocations table
        now = datetime.utcnow().isoformat()
        storage_results = [
            {
                "request_id": r["request_id"],
                "region_name": r["region_name"],
                "specialty_name": r["specialty_name"],
                "allocated_count": r["allocated_quota"],
                "requested_count": r["requested_quota"],
                "priority_score": r["priority_score"],
                "allocation_date": now
            }
            for r in results
        ]
        
        # Clear old allocations
        del_resp = supabase.table("allocations").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        
        # Insert new allocations
        ins_resp = supabase.table("allocations").insert(storage_results).execute()
        if hasattr(ins_resp, 'error') and ins_resp.error:
            print("Supabase allocations insert error:", ins_resp.error)
            return {"status": "error", "error": str(ins_resp.error)}

        # Optionally: Archive demand_requests instead of deleting
        # This preserves historical data for analysis
        archive_resp = supabase.table("demand_requests").update({
            "archived": True,
            "archived_at": now
        }).eq("archived", False).execute()
        
        # Calculate statistics
        total_allocated = sum(r['allocated_quota'] for r in results)
        total_requested = sum(r['requested_quota'] for r in results)
        avg_fulfillment = sum(r['fulfillment_rate'] for r in results) / len(results)
        
        return {
            "status": "success",
            "allocations": results,
            "statistics": {
                "total_budget": payload.budget,
                "grants_used": total_allocated,
                "grants_remaining": payload.budget - total_allocated,
                "total_requested": total_requested,
                "requests_total": len(results),
                "requests_fully_funded": sum(1 for r in results if r['fulfillment_rate'] >= 100),
                "requests_partially_funded": sum(1 for r in results if 0 < r['fulfillment_rate'] < 100),
                "average_fulfillment_rate": round(avg_fulfillment, 1)
            }
        }
        
    except Exception as e:
        import traceback
        print("ERROR:", str(e))
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "scholarship-allocation-api"}


@app.get("/allocation-stats")
async def get_allocation_stats():
    """Get statistics about current allocations"""
    try:
        # Get current allocations
        alloc_resp = supabase.table("allocations").select("*").execute()
        allocations = alloc_resp.data
        
        if not allocations:
            return {"status": "no_data", "message": "No allocations found"}
        
        # Calculate statistics
        total_allocated = sum(a.get('allocated_count', 0) for a in allocations)
        
        # Group by region
        by_region = {}
        for a in allocations:
            region = a.get('region_name', 'Unknown')
            if region not in by_region:
                by_region[region] = 0
            by_region[region] += a.get('allocated_count', 0)
        
        # Group by specialty
        by_specialty = {}
        for a in allocations:
            specialty = a.get('specialty_name', 'Unknown')
            if specialty not in by_specialty:
                by_specialty[specialty] = 0
            by_specialty[specialty] += a.get('allocated_count', 0)
        
        return {
            "status": "success",
            "total_allocated": total_allocated,
            "total_requests": len(allocations),
            "by_region": by_region,
            "by_specialty": by_specialty
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))