import os
import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
import uuid

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


class AllocationRequest(BaseModel):
    budget: int  # Total scholarships to allocate (e.g., 2500)
    specialty_weight: Optional[float] = 0.4
    region_weight: Optional[float] = 0.3
    deficit_weight: Optional[float] = 0.3
    run_by: Optional[str] = "system"
    notes: Optional[str] = None
    year: Optional[int] = None


# --- 2. Enhanced Priority Calculator ---
def calculate_priority_score(
    specialty_score: int,
    region_coefficient: int,
    historical_deficit: int,
    specialty_weight: float = 0.4,
    region_weight: float = 0.3,
    deficit_weight: float = 0.3
) -> float:
    """
    Multi-factor priority calculation:
    - Specialty importance (normalized 0-100)
    - Region government priority (coefficient 0-100)
    - Historical deficit urgency (normalized)
    
    Returns a score between 0-100
    """
    # Normalize deficit score (assuming max deficit around 100)
    deficit_score = min(100, (historical_deficit / 100.0) * 100) if historical_deficit > 0 else 0
    
    # Weighted combination
    final_score = (
        specialty_score * specialty_weight +
        region_coefficient * region_weight +
        deficit_score * deficit_weight
    )
    
    return final_score


# --- 3. Enhanced Core Algorithm ---
def run_allocation_logic(payload: AllocationRequest):
    """
    Min-cost max-flow allocation algorithm using NetworkX.
    Now fully integrated with the enhanced database schema.
    """
    total_grants = payload.budget
    allocation_year = payload.year or datetime.now().year
    
    # Create a new allocation run record
    batch_id = str(uuid.uuid4())
    run_start = datetime.utcnow()
    
    try:
        # Insert allocation run record
        run_record = {
            "batch_id": batch_id,
            "total_budget": total_grants,
            "specialty_weight": payload.specialty_weight,
            "region_weight": payload.region_weight,
            "deficit_weight": payload.deficit_weight,
            "run_by": payload.run_by,
            "run_year": allocation_year,
            "notes": payload.notes,
            "status": "running",
            "started_at": run_start.isoformat(),
            "algorithm_version": "v2.0"
        }
        
        run_response = supabase.table("allocation_runs").insert(run_record).execute()
        if hasattr(run_response, 'error') and run_response.error:
            print(f"Warning: Could not create allocation run record: {run_response.error}")
        
        # Fetch all specialties with their scores
        specialties_response = supabase.table("specialties").select("*").eq("archived", False).execute()
        specialties_data = specialties_response.data or []
        specialties_map = {s['name']: s for s in specialties_data}
        
        # Fetch all regions with their coefficients
        regions_response = supabase.table("regions").select("*").eq("archived", False).execute()
        regions_data = regions_response.data or []
        regions_map = {r['name']: r for r in regions_data}
        
        # Fetch pending demand requests
        demand_response = supabase.table("demand_requests").select("*").eq("status", "pending").eq("archived", False).execute()
        requests = demand_response.data or []
        
        if not requests:
            print("No pending demand requests found in database")
            # Update run status
            supabase.table("allocation_runs").update({
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "total_allocated": 0,
                "total_requested": 0,
                "requests_count": 0
            }).eq("batch_id", batch_id).execute()
            return []
        
        print(f"Found {len(requests)} pending demand requests")
        
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
            request_id = req['id']
            
            # Calculate total need
            historical_deficit = req.get('historical_deficit', 0) or 0
            request_uz_amount = req.get('request_uz_amount', 0) or 0
            slots_needed = historical_deficit + request_uz_amount
            
            if slots_needed < 1:
                continue
            
            total_requested_slots += slots_needed
            
            # Get specialty and region data
            specialty = specialties_map.get(specialty_name, {})
            region = regions_map.get(region_name, {})
            
            specialty_score = specialty.get('normalized_score', 50)
            region_coefficient = region.get('coefficient', 50)
            
            # Calculate priority score (use stored one if exists and valid)
            priority_score = req.get('priority_score', 0)
            if priority_score is None or priority_score == 0:
                priority_score = calculate_priority_score(
                    specialty_score=specialty_score,
                    region_coefficient=region_coefficient,
                    historical_deficit=historical_deficit,
                    specialty_weight=payload.specialty_weight,
                    region_weight=payload.region_weight,
                    deficit_weight=payload.deficit_weight
                )
                
                # Update the priority score in the database
                supabase.table("demand_requests").update({
                    "priority_score": round(priority_score, 2)
                }).eq("id", request_id).execute()
            
            # Create unique node ID
            node_id = f"{region_name}|{specialty_name}|{request_id}"
            
            # Add intermediate node for this request
            G.add_node(node_id, demand=0)
            
            # EDGE 1: SOURCE -> Request Node
            # Cost is inverted priority (lower cost = higher priority)
            cost = 1000 - int(priority_score)
            G.add_edge("SOURCE", node_id, capacity=slots_needed, weight=cost)
            
            # EDGE 2: Request Node -> SINK
            # Zero cost, same capacity
            G.add_edge(node_id, "SINK", capacity=slots_needed, weight=0)
            
            valid_requests.append({
                "node_id": node_id,
                "request_id": request_id,
                "region_id": region.get('id'),
                "specialty_id": specialty.get('id'),
                "region_name": region_name,
                "specialty_name": specialty_name,
                "slots_needed": slots_needed,
                "priority_score": priority_score,
                "historical_deficit": historical_deficit,
                "request_uz_amount": request_uz_amount
            })
        
        if not valid_requests:
            print("No valid requests after filtering")
            # Update run status
            supabase.table("allocation_runs").update({
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "total_allocated": 0,
                "total_requested": total_requested_slots,
                "requests_count": 0
            }).eq("batch_id", batch_id).execute()
            return []
        
        # Log allocation scenario
        print(f"=== Allocation Scenario ===")
        print(f"Total Budget: {total_grants}")
        print(f"Total Requested: {total_requested_slots}")
        print(f"Valid Requests: {len(valid_requests)}")
        print(f"Weights: Specialty={payload.specialty_weight}, Region={payload.region_weight}, Deficit={payload.deficit_weight}")
        
        if total_requested_slots < total_grants:
            print(f"ℹ️  Under-subscription by {total_grants - total_requested_slots} slots")
        elif total_requested_slots > total_grants:
            print(f"⚠️  Over-subscription: {total_requested_slots - total_grants} slots will not be fully funded")
        
        # --- Solve the Min-Cost Flow Problem ---
        try:
            flow_dict = nx.min_cost_flow(G)
        except nx.NetworkXUnfeasible:
            error_msg = "Graph is infeasible - cannot satisfy flow constraints"
            print(f"ERROR: {error_msg}")
            
            # Update run status as failed
            supabase.table("allocation_runs").update({
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "error_message": error_msg
            }).eq("batch_id", batch_id).execute()
            
            return []
        except Exception as e:
            error_msg = f"Flow calculation failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            
            # Update run status as failed
            supabase.table("allocation_runs").update({
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "error_message": error_msg
            }).eq("batch_id", batch_id).execute()
            
            return []
        
        # --- Parse Results ---
        results = []
        total_allocated = 0
        fully_funded = 0
        partially_funded = 0
        unfunded = 0
        
        for vr in valid_requests:
            node_id = vr["node_id"]
            # Flow from SOURCE to this request node
            allocated = flow_dict.get("SOURCE", {}).get(node_id, 0)
            
            requested = vr["slots_needed"]
            fulfillment_rate = (allocated / requested * 100) if requested > 0 else 0
            
            # Count funding categories
            if allocated == requested:
                fully_funded += 1
            elif allocated > 0:
                partially_funded += 1
            else:
                unfunded += 1
            
            total_allocated += allocated
            
            # Only include in results if something was allocated
            if allocated > 0:
                results.append({
                    "request_id": vr["request_id"],
                    "region_id": vr["region_id"],
                    "specialty_id": vr["specialty_id"],
                    "region_name": vr["region_name"],
                    "specialty_name": vr["specialty_name"],
                    "allocated_quota": allocated,
                    "requested_quota": requested,
                    "priority_score": round(vr["priority_score"], 2),
                    "historical_deficit": vr["historical_deficit"],
                    "request_uz_amount": vr["request_uz_amount"],
                    "fulfillment_rate": round(fulfillment_rate, 1)
                })
        
        # Sort by priority score (highest first)
        results.sort(key=lambda x: x["priority_score"], reverse=True)
        
        # Calculate statistics
        avg_fulfillment = sum(r['fulfillment_rate'] for r in results) / len(results) if results else 0
        
        print(f"\n=== Allocation Results ===")
        print(f"Total Allocated: {total_allocated} / {total_grants}")
        print(f"Grants Remaining: {total_grants - total_allocated}")
        print(f"Requests Funded: {len(results)} / {len(valid_requests)}")
        print(f"  - Fully Funded: {fully_funded}")
        print(f"  - Partially Funded: {partially_funded}")
        print(f"  - Unfunded: {unfunded}")
        print(f"Average Fulfillment Rate: {avg_fulfillment:.1f}%")
        
        # Update allocation run with final statistics
        run_end = datetime.utcnow()
        duration = (run_end - run_start).total_seconds()
        
        supabase.table("allocation_runs").update({
            "status": "completed",
            "completed_at": run_end.isoformat(),
            "duration_seconds": int(duration),
            "total_allocated": total_allocated,
            "total_requested": total_requested_slots,
            "requests_count": len(valid_requests),
            "requests_fully_funded": fully_funded,
            "requests_partially_funded": partially_funded,
            "requests_unfunded": unfunded,
            "average_fulfillment_rate": round(avg_fulfillment, 2)
        }).eq("batch_id", batch_id).execute()
        
        return results, batch_id
        
    except Exception as e:
        # Log error and update run status
        error_msg = str(e)
        print(f"CRITICAL ERROR: {error_msg}")
        
        try:
            supabase.table("allocation_runs").update({
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "error_message": error_msg
            }).eq("batch_id", batch_id).execute()
        except:
            pass
        
        raise


# --- 4. Enhanced Endpoint ---
@app.post("/run-allocation")
async def trigger_allocation(payload: AllocationRequest):
    """
    Run the scholarship allocation algorithm and save results.
    
    Process:
    1. Creates allocation_run record
    2. Fetches specialties, regions, and pending demands
    3. Runs min-cost flow optimization
    4. Saves allocations to database
    5. Updates demand_requests status
    6. Returns detailed results and statistics
    """
    try:
        # Validate weights sum to 1.0
        total_weight = payload.specialty_weight + payload.region_weight + payload.deficit_weight
        if abs(total_weight - 1.0) > 0.01:
            raise HTTPException(
                status_code=400,
                detail=f"Weights must sum to 1.0 (currently: {total_weight})"
            )
        
        # Run the allocation algorithm
        result = run_allocation_logic(payload)
        
        if not result:
            return {
                "status": "warning",
                "message": "No allocations made - check demand_requests table for pending requests",
                "allocations": [],
                "statistics": {
                    "total_budget": payload.budget,
                    "grants_used": 0,
                    "grants_remaining": payload.budget,
                    "requests_funded": 0
                }
            }
        
        results, batch_id = result
        
        if not results:
            return {
                "status": "warning", 
                "message": "Algorithm completed but no allocations were made",
                "allocations": [],
                "batch_id": batch_id
            }
        
        # Save results to allocations table
        now = datetime.utcnow().isoformat()
        storage_results = [
            {
                "request_id": r["request_id"],
                "region_id": r["region_id"],
                "specialty_id": r["specialty_id"],
                "region_name": r["region_name"],
                "specialty_name": r["specialty_name"],
                "allocated_count": r["allocated_quota"],
                "requested_count": r["requested_quota"],
                "priority_score": r["priority_score"],
                "historical_deficit": r["historical_deficit"],
                "request_uz_amount": r["request_uz_amount"],
                "allocation_date": now,
                "allocation_batch_id": batch_id,
                "allocation_year": payload.year or datetime.utcnow().year,
                "total_budget_for_run": payload.budget,
                "algorithm_version": "v2.0",
                "status": "allocated"
            }
            for r in results
        ]
        
        # Insert allocations
        ins_resp = supabase.table("allocations").insert(storage_results).execute()
        if hasattr(ins_resp, 'error') and ins_resp.error:
            print("Supabase allocations insert error:", ins_resp.error)
            return {"status": "error", "error": str(ins_resp.error)}
        
        # Update demand_requests status to 'allocated'
        request_ids = [r["request_id"] for r in results]
        update_resp = supabase.table("demand_requests").update({
            "status": "allocated",
            "archived": True,
            "archived_at": now
        }).in_("id", request_ids).execute()
        
        if hasattr(update_resp, 'error') and update_resp.error:
            print("Warning: Could not update demand_requests status:", update_resp.error)
        
        # Calculate final statistics
        total_allocated = sum(r['allocated_quota'] for r in results)
        total_requested = sum(r['requested_quota'] for r in results)
        avg_fulfillment = sum(r['fulfillment_rate'] for r in results) / len(results)
        
        return {
            "status": "success",
            "message": f"Successfully allocated {total_allocated} scholarships across {len(results)} requests",
            "batch_id": batch_id,
            "allocations": results,
            "statistics": {
                "total_budget": payload.budget,
                "grants_used": total_allocated,
                "grants_remaining": payload.budget - total_allocated,
                "total_requested": total_requested,
                "requests_total": len(results),
                "requests_fully_funded": sum(1 for r in results if r['fulfillment_rate'] >= 99.9),
                "requests_partially_funded": sum(1 for r in results if 0 < r['fulfillment_rate'] < 99.9),
                "average_fulfillment_rate": round(avg_fulfillment, 1),
                "weights_used": {
                    "specialty": payload.specialty_weight,
                    "region": payload.region_weight,
                    "deficit": payload.deficit_weight
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("ERROR:", str(e))
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "scholarship-allocation-api", "version": "v2.0"}


@app.get("/allocation-runs")
async def get_allocation_runs(limit: int = 10):
    """Get recent allocation runs with statistics"""
    try:
        response = supabase.table("allocation_runs").select("*").order("started_at", desc=True).limit(limit).execute()
        
        runs = response.data or []
        
        return {
            "status": "success",
            "runs": runs,
            "count": len(runs)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/allocation-runs/{batch_id}")
async def get_allocation_run_details(batch_id: str):
    """Get details of a specific allocation run"""
    try:
        # Get run metadata
        run_response = supabase.table("allocation_runs").select("*").eq("batch_id", batch_id).execute()
        
        if not run_response.data:
            raise HTTPException(status_code=404, detail="Allocation run not found")
        
        run = run_response.data[0]
        
        # Get allocations for this run
        alloc_response = supabase.table("allocations").select("*").eq("allocation_batch_id", batch_id).execute()
        
        allocations = alloc_response.data or []
        
        return {
            "status": "success",
            "run": run,
            "allocations": allocations,
            "allocation_count": len(allocations)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/allocation-stats")
async def get_allocation_stats(year: Optional[int] = None):
    """Get statistics about allocations for a specific year or current year"""
    try:
        target_year = year or datetime.now().year
        
        # Get allocations for the year
        alloc_resp = supabase.table("allocations").select("*").eq("allocation_year", target_year).execute()
        allocations = alloc_resp.data or []
        
        if not allocations:
            return {
                "status": "no_data",
                "message": f"No allocations found for year {target_year}",
                "year": target_year
            }
        
        # Calculate statistics
        total_allocated = sum(a.get('allocated_count', 0) for a in allocations)
        total_requested = sum(a.get('requested_count', 0) for a in allocations)
        
        # Group by region
        by_region = {}
        for a in allocations:
            region = a.get('region_name', 'Unknown')
            if region not in by_region:
                by_region[region] = {"allocated": 0, "requested": 0, "count": 0}
            by_region[region]["allocated"] += a.get('allocated_count', 0)
            by_region[region]["requested"] += a.get('requested_count', 0)
            by_region[region]["count"] += 1
        
        # Group by specialty
        by_specialty = {}
        for a in allocations:
            specialty = a.get('specialty_name', 'Unknown')
            if specialty not in by_specialty:
                by_specialty[specialty] = {"allocated": 0, "requested": 0, "count": 0}
            by_specialty[specialty]["allocated"] += a.get('allocated_count', 0)
            by_specialty[specialty]["requested"] += a.get('requested_count', 0)
            by_specialty[specialty]["count"] += 1
        
        # Calculate fulfillment rates
        for region_data in by_region.values():
            if region_data["requested"] > 0:
                region_data["fulfillment_rate"] = round(
                    (region_data["allocated"] / region_data["requested"]) * 100, 1
                )
        
        for specialty_data in by_specialty.values():
            if specialty_data["requested"] > 0:
                specialty_data["fulfillment_rate"] = round(
                    (specialty_data["allocated"] / specialty_data["requested"]) * 100, 1
                )
        
        return {
            "status": "success",
            "year": target_year,
            "summary": {
                "total_allocated": total_allocated,
                "total_requested": total_requested,
                "total_requests": len(allocations),
                "overall_fulfillment_rate": round((total_allocated / total_requested * 100), 1) if total_requested > 0 else 0
            },
            "by_region": by_region,
            "by_specialty": by_specialty
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pending-demands")
async def get_pending_demands():
    """Get all pending demand requests that haven't been allocated yet"""
    try:
        response = supabase.table("demand_requests").select("*").eq("status", "pending").eq("archived", False).order("priority_score", desc=True).execute()
        
        demands = response.data or []
        
        total_requested = sum(d.get('historical_deficit', 0) + d.get('request_uz_amount', 0) for d in demands)
        
        return {
            "status": "success",
            "demands": demands,
            "count": len(demands),
            "total_slots_requested": total_requested
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/allocations/{batch_id}")
async def delete_allocation_batch(batch_id: str):
    """Delete an entire allocation batch (use with caution!)"""
    try:
        # First check if batch exists
        run_resp = supabase.table("allocation_runs").select("id").eq("batch_id", batch_id).execute()
        
        if not run_resp.data:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        # Get request IDs from this batch to un-archive them
        alloc_resp = supabase.table("allocations").select("request_id").eq("allocation_batch_id", batch_id).execute()
        request_ids = [a['request_id'] for a in (alloc_resp.data or []) if a.get('request_id')]
        
        # Delete allocations
        supabase.table("allocations").delete().eq("allocation_batch_id", batch_id).execute()
        
        # Delete allocation run
        supabase.table("allocation_runs").delete().eq("batch_id", batch_id).execute()
        
        # Un-archive demand requests
        if request_ids:
            supabase.table("demand_requests").update({
                "status": "pending",
                "archived": False,
                "archived_at": None
            }).in_("id", request_ids).execute()
        
        return {
            "status": "success",
            "message": f"Deleted allocation batch {batch_id}",
            "requests_restored": len(request_ids)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))