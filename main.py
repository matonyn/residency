@app.delete("/allocations")
async def delete_allocations():
    try:
        resp = supabase.table("allocations").delete().neq("id", "").execute()
        return {"status": "success", "message": "All allocations deleted"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
import os
import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
import uuid
import traceback

# --- 1. Setup & Config ---
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_SERVICE_KEY")

print(f"Supabase URL: {url}")
print(f"Supabase Key exists: {bool(key)}")

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
    budget: int
    specialty_weight: Optional[float] = 0.4
    region_weight: Optional[float] = 0.3
    deficit_weight: Optional[float] = 0.3
    run_by: Optional[str] = "system"
    notes: Optional[str] = None
    year: Optional[int] = None


def safe_db_call(operation_name: str, db_call):
    """Wrapper for database calls with error handling"""
    try:
        print(f"📊 Executing: {operation_name}")
        result = db_call()
        
        # Check for Supabase errors
        if hasattr(result, 'error') and result.error:
            print(f"❌ Database error in {operation_name}: {result.error}")
            return None
        
        data = result.data if hasattr(result, 'data') else result
        print(f"✅ {operation_name} successful - returned {len(data) if data else 0} records")
        return data
        
    except Exception as e:
        print(f"❌ Exception in {operation_name}: {str(e)}")
        print(traceback.format_exc())
        return None


def calculate_priority_score(
    specialty_score: int,
    region_coefficient: int,
    historical_deficit: int,
    specialty_weight: float = 0.4,
    region_weight: float = 0.3,
    deficit_weight: float = 0.3
) -> float:
    """Calculate weighted priority score"""
    deficit_score = min(100, (historical_deficit / 100.0) * 100) if historical_deficit > 0 else 0
    
    final_score = (
        specialty_score * specialty_weight +
        region_coefficient * region_weight +
        deficit_score * deficit_weight
    )
    
    return final_score


def run_allocation_logic(payload: AllocationRequest):
    """
    Main allocation algorithm with extensive debugging
    """
    print("\n" + "="*60)
    print("🚀 STARTING ALLOCATION RUN")
    print("="*60)
    
    total_grants = payload.budget
    allocation_year = payload.year or datetime.now().year
    batch_id = str(uuid.uuid4())
    run_start = datetime.utcnow()
    
    print(f"📋 Configuration:")
    print(f"   Budget: {total_grants}")
    print(f"   Year: {allocation_year}")
    print(f"   Batch ID: {batch_id}")
    print(f"   Weights: S={payload.specialty_weight}, R={payload.region_weight}, D={payload.deficit_weight}")
    
    try:
        # Step 1: Create allocation run record
        print("\n📝 Step 1: Creating allocation run record...")
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
            "algorithm_version": "v2.0-debug"
        }
        
        run_response = safe_db_call(
            "Create allocation_runs record",
            lambda: supabase.table("allocation_runs").insert(run_record).execute()
        )
        
        # Step 2: Fetch specialties
        print("\n📚 Step 2: Fetching specialties...")
        specialties_data = safe_db_call(
            "Fetch specialties",
            lambda: supabase.table("specialties").select("*").eq("archived", False).execute()
        )
        
        if specialties_data is None:
            # Try without archived filter in case column doesn't exist
            print("⚠️  Trying without archived filter...")
            specialties_data = safe_db_call(
                "Fetch specialties (no filter)",
                lambda: supabase.table("specialties").select("*").execute()
            )
        
        if not specialties_data:
            print("⚠️  No specialties found - will use defaults")
            specialties_map = {}
        else:
            specialties_map = {s['name']: s for s in specialties_data}
            print(f"   Found {len(specialties_map)} specialties: {list(specialties_map.keys())[:5]}...")
        
        # Step 3: Fetch regions
        print("\n🗺️  Step 3: Fetching regions...")
        regions_data = safe_db_call(
            "Fetch regions",
            lambda: supabase.table("regions").select("*").eq("archived", False).execute()
        )
        
        if regions_data is None:
            # Try without archived filter
            print("⚠️  Trying without archived filter...")
            regions_data = safe_db_call(
                "Fetch regions (no filter)",
                lambda: supabase.table("regions").select("*").execute()
            )
        
        if not regions_data:
            print("⚠️  No regions found - will use defaults")
            regions_map = {}
        else:
            regions_map = {r['name']: r for r in regions_data}
            print(f"   Found {len(regions_map)} regions: {list(regions_map.keys())[:5]}...")
        
        # Step 4: Fetch demand requests
        print("\n📥 Step 4: Fetching demand requests...")
        requests = safe_db_call(
            "Fetch demand_requests",
            lambda: supabase.table("demand_requests").select("*").eq("status", "pending").eq("archived", False).execute()
        )
        
        if requests is None:
            # Try without status filter
            print("⚠️  Trying without status/archived filters...")
            requests = safe_db_call(
                "Fetch demand_requests (no filters)",
                lambda: supabase.table("demand_requests").select("*").execute()
            )
        
        if not requests:
            print("❌ No demand requests found!")
            print("   Check your demand_requests table has data")
            
            # Update run as completed with no allocations
            safe_db_call(
                "Update run status (no requests)",
                lambda: supabase.table("allocation_runs").update({
                    "status": "completed",
                    "completed_at": datetime.utcnow().isoformat(),
                    "total_allocated": 0,
                    "total_requested": 0,
                    "requests_count": 0,
                    "error_message": "No pending demand requests found"
                }).eq("batch_id", batch_id).execute()
            )
            
            return [], batch_id
        
        print(f"   Found {len(requests)} demand requests")
        
        # Step 5: Build graph
        print("\n🕸️  Step 5: Building flow network graph...")
        G = nx.DiGraph()
        G.add_node("SOURCE", demand=-total_grants)
        G.add_node("SINK", demand=total_grants)
        
        total_requested_slots = 0
        valid_requests = []
        skipped_requests = 0
        
        for idx, req in enumerate(requests):
            region_name = req.get('region', 'Unknown')
            specialty_name = req.get('specialty', 'Unknown')
            request_id = req.get('id', f'unknown-{idx}')
            
            # Calculate total need
            historical_deficit = req.get('historical_deficit') or 0
            request_uz_amount = req.get('request_uz_amount') or 0
            slots_needed = historical_deficit + request_uz_amount
            
            if slots_needed < 1:
                skipped_requests += 1
                continue
            
            total_requested_slots += slots_needed
            
            # Get specialty and region data
            specialty = specialties_map.get(specialty_name, {})
            region = regions_map.get(region_name, {})
            
            specialty_score = specialty.get('normalized_score', 50)
            region_coefficient = region.get('coefficient', 50)
            
            # Calculate priority score
            priority_score = req.get('priority_score') or 0
            if priority_score == 0:
                priority_score = calculate_priority_score(
                    specialty_score=specialty_score,
                    region_coefficient=region_coefficient,
                    historical_deficit=historical_deficit,
                    specialty_weight=payload.specialty_weight,
                    region_weight=payload.region_weight,
                    deficit_weight=payload.deficit_weight
                )
            
            # Create node
            node_id = f"{region_name}|{specialty_name}|{request_id}"
            G.add_node(node_id, demand=0)
            
            # Add edges
            cost = 1000 - int(priority_score)
            G.add_edge("SOURCE", node_id, capacity=slots_needed, weight=cost)
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
        
        print(f"   Valid requests: {len(valid_requests)}")
        print(f"   Skipped requests: {skipped_requests}")
        print(f"   Total slots requested: {total_requested_slots}")
        print(f"   Budget: {total_grants}")
        
        if total_requested_slots > total_grants:
            print(f"   ⚠️  Over-subscribed by {total_requested_slots - total_grants} slots")
        else:
            print(f"   ℹ️  Under-subscribed by {total_grants - total_requested_slots} slots")
        
        if not valid_requests:
            print("❌ No valid requests after filtering!")
            
            safe_db_call(
                "Update run status (no valid requests)",
                lambda: supabase.table("allocation_runs").update({
                    "status": "completed",
                    "completed_at": datetime.utcnow().isoformat(),
                    "total_allocated": 0,
                    "total_requested": total_requested_slots,
                    "requests_count": 0,
                    "error_message": "No valid requests with slots_needed > 0"
                }).eq("batch_id", batch_id).execute()
            )
            
            return [], batch_id
        
        # Step 6: Solve flow problem
        print("\n🔄 Step 6: Solving min-cost max-flow problem...")
        try:
            flow_dict = nx.min_cost_flow(G)
            print("   ✅ Flow solution found!")
        except nx.NetworkXUnfeasible as e:
            error_msg = f"Graph infeasible: {str(e)}"
            print(f"   ❌ {error_msg}")
            
            safe_db_call(
                "Update run status (infeasible)",
                lambda: supabase.table("allocation_runs").update({
                    "status": "failed",
                    "completed_at": datetime.utcnow().isoformat(),
                    "error_message": error_msg
                }).eq("batch_id", batch_id).execute()
            )
            
            return [], batch_id
        
        # Step 7: Parse results
        print("\n📊 Step 7: Parsing allocation results...")
        results = []
        total_allocated = 0
        fully_funded = 0
        partially_funded = 0
        unfunded = 0
        
        for vr in valid_requests:
            node_id = vr["node_id"]
            allocated = flow_dict.get("SOURCE", {}).get(node_id, 0)
            
            requested = vr["slots_needed"]
            fulfillment_rate = (allocated / requested * 100) if requested > 0 else 0
            
            if allocated == requested:
                fully_funded += 1
            elif allocated > 0:
                partially_funded += 1
            else:
                unfunded += 1
            
            total_allocated += allocated
            
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
        
        results.sort(key=lambda x: x["priority_score"], reverse=True)
        
        print(f"\n📈 Results Summary:")
        print(f"   Total allocated: {total_allocated} / {total_grants}")
        print(f"   Fully funded: {fully_funded}")
        print(f"   Partially funded: {partially_funded}")
        print(f"   Unfunded: {unfunded}")
        
        # Step 8: Update run record
        print("\n💾 Step 8: Updating allocation run record...")
        run_end = datetime.utcnow()
        duration = (run_end - run_start).total_seconds()
        avg_fulfillment = sum(r['fulfillment_rate'] for r in results) / len(results) if results else 0
        
        safe_db_call(
            "Update allocation_runs (final)",
            lambda: supabase.table("allocation_runs").update({
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
        )
        
        print("\n✅ Allocation run completed successfully!")
        print("="*60 + "\n")
        
        return results, batch_id
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ CRITICAL ERROR: {error_msg}")
        print(traceback.format_exc())
        
        try:
            safe_db_call(
                "Update run status (error)",
                lambda: supabase.table("allocation_runs").update({
                    "status": "failed",
                    "completed_at": datetime.utcnow().isoformat(),
                    "error_message": error_msg
                }).eq("batch_id", batch_id).execute()
            )
        except:
            pass
        
        raise


@app.post("/run-allocation")
async def trigger_allocation(payload: AllocationRequest):
    """
    Run allocation with extensive debugging
    """
    print("\n" + "🎯"*30)
    print("API ENDPOINT: /run-allocation called")
    print("🎯"*30)
    
    try:
        # Validate weights
        total_weight = payload.specialty_weight + payload.region_weight + payload.deficit_weight
        if abs(total_weight - 1.0) > 0.01:
            raise HTTPException(
                status_code=400,
                detail=f"Weights must sum to 1.0 (currently: {total_weight})"
            )
        
        # Run allocation
        result = run_allocation_logic(payload)
        
        if not result:
            return {
                "status": "warning",
                "message": "No allocations made",
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
                "batch_id": batch_id,
                "allocations": []
            }
        
        # Step 9: Save to allocations table
        print("\n💾 Step 9: Saving allocations to database...")
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
                "algorithm_version": "v2.0-debug",
                "status": "allocated"
            }
            for r in results
        ]
        
        ins_resp = safe_db_call(
            f"Insert {len(storage_results)} allocations",
            lambda: supabase.table("allocations").insert(storage_results).execute()
        )
        
        if ins_resp is None:
            print("⚠️  Warning: Could not save allocations to database")
        
        # Step 10: Update demand_requests
        print("\n📝 Step 10: Updating demand_requests status...")
        request_ids = [r["request_id"] for r in results]
        
        if request_ids:
            update_resp = safe_db_call(
                f"Update {len(request_ids)} demand_requests",
                lambda: supabase.table("demand_requests").update({
                    "status": "allocated",
                    "archived": True,
                    "archived_at": now
                }).in_("id", request_ids).execute()
            )
            
            if update_resp is None:
                print("⚠️  Warning: Could not update demand_requests status")
        
        # Calculate statistics
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
                "average_fulfillment_rate": round(avg_fulfillment, 1)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ API ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check with database connectivity test"""
    try:
        # Test database connection
        test = safe_db_call(
            "Health check - test query",
            lambda: supabase.table("specialties").select("count").limit(1).execute()
        )
        
        db_status = "connected" if test is not None else "error"
        
        return {
            "status": "ok",
            "service": "scholarship-allocation-api",
            "version": "v2.0-debug",
            "database": db_status
        }
    except Exception as e:
        return {
            "status": "error",
            "service": "scholarship-allocation-api",
            "version": "v2.0-debug",
            "database": "disconnected",
            "error": str(e)
        }


@app.get("/debug/database")
async def debug_database():
    """Debug endpoint to check database tables"""
    results = {}
    
    # Check each table
    tables = ["specialties", "regions", "demand_requests", "allocations", "allocation_runs"]
    
    for table in tables:
        try:
            count_resp = supabase.table(table).select("*", count="exact").limit(0).execute()
            results[table] = {
                "status": "ok",
                "count": count_resp.count if hasattr(count_resp, 'count') else "unknown"
            }
        except Exception as e:
            results[table] = {
                "status": "error",
                "error": str(e)
            }
    
    return results


@app.get("/debug/pending-demands")
async def debug_pending_demands():
    """Debug endpoint to see pending demands"""
    try:
        # Try with filters
        demands = safe_db_call(
            "Get pending demands (with filters)",
            lambda: supabase.table("demand_requests").select("*").eq("status", "pending").eq("archived", False).execute()
        )
        
        if demands is None:
            # Try without filters
            demands = safe_db_call(
                "Get all demands (no filters)",
                lambda: supabase.table("demand_requests").select("*").execute()
            )
        
        return {
            "status": "success",
            "count": len(demands) if demands else 0,
            "demands": demands[:5] if demands else []  # First 5 for debugging
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
