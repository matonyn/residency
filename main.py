"""
Complete Allocation API with Excel Parser
Supports both Excel (.xlsx) and PDF uploads
"""

import os
import uuid
from datetime import datetime
from typing import List, Optional
import sys

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel

# --- Setup ---

# Load environment variables from .env.local
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env.local"))

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(url, key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---

class CreateSessionRequest(BaseModel):
    session_name: str
    year: int
    total_budget: int
    uploaded_by: Optional[str] = "system"


class AdjustAllocationRequest(BaseModel):
    demand_request_id: str
    new_allocation: int
    notes: Optional[str] = None
    changed_by: Optional[str] = "user"


class BulkAdjustRequest(BaseModel):
    adjustments: List[AdjustAllocationRequest]


class FinalizeSessionRequest(BaseModel):
    session_id: str
    finalized_by: str


# --- Excel Parser Helper ---
import asyncio
import io
import openpyxl

def parse_excel_demands(file_content):
    """
    Parse Excel file and extract demand data rows.
    Returns a list of dicts for each demand row.
    """
    wb = openpyxl.load_workbook(io.BytesIO(file_content), data_only=True)
    ws = wb.active
    headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        row_dict = dict(zip(headers, row))
        rows.append(row_dict)
    return rows


def _normalize_header(h):
    """Normalize Excel header for flexible matching."""
    if h is None:
        return ""
    s = str(h).strip().lower().replace(" ", "_").replace("-", "_")
    return s


def parse_excel_graduates(file_content, sheet_name=None, default_year=None):
    """
    Parse Excel file and extract yearly graduate data.
    Expects columns: year (or use default_year), region, specialty, graduate_count (or graduates/count).
    Optional: use sheet_name to read a specific sheet (e.g. "Graduates" or "Выпускники"); else first sheet.
    Returns a list of dicts: { year, region, specialty, graduate_count }.
    """
    wb = openpyxl.load_workbook(io.BytesIO(file_content), data_only=True)
    ws = wb[sheet_name] if sheet_name and sheet_name in wb.sheetnames else wb.active
    raw_headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    headers = [_normalize_header(h) for h in raw_headers]
    # Map common column names to canonical keys
    year_col = None
    region_col = None
    specialty_col = None
    count_col = None
    for i, h in enumerate(headers):
        if h in ("year", "год", "г"):
            year_col = i
        elif h in ("region", "регион", "область", "region_name"):
            region_col = i
        elif h in ("specialty", "speciality", "специальность", "specialty_name"):
            specialty_col = i
        elif h in ("graduate_count", "graduates", "count", "количество", "number", "num"):
            count_col = i
    if region_col is None or specialty_col is None or count_col is None:
        raise ValueError(
            "Excel must have columns for region, specialty, and graduate count. "
            "Accepted names: region/Region, specialty/Specialty, graduate_count/graduates/count."
        )
    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        region = row[region_col] if region_col is not None else None
        specialty = row[specialty_col] if specialty_col is not None else None
        count_val = row[count_col] if count_col is not None else None
        if region is None or specialty is None:
            continue
        region = str(region).strip() if region else ""
        specialty = str(specialty).strip() if specialty else ""
        if not region or not specialty:
            continue
        year_val = default_year
        if year_col is not None and row[year_col] is not None:
            try:
                year_val = int(float(row[year_col]))
            except (TypeError, ValueError):
                pass
        if year_val is None:
            raise ValueError("Year is required: either include a 'year' column in Excel or pass default_year.")
        try:
            graduate_count = int(float(count_val)) if count_val is not None else 0
        except (TypeError, ValueError):
            graduate_count = 0
        rows.append({
            "year": year_val,
            "region": region,
            "specialty": specialty,
            "graduate_count": max(0, graduate_count),
        })
    return rows

async def process_uploaded_excel(content, session_id, supabase):
    """
    Process uploaded Excel file, insert demands, and generate suggestions.
    Returns dict with success, counts, and metadata.
    """
    try:
        demands = parse_excel_demands(content)
        inserted = 0
        suggestions = 0
        metadata = {"columns": list(demands[0].keys()) if demands else []}
        for d in demands:
            # Prepare demand data
            demand_data = {
                "session_id": session_id,
                "region": d.get("region") or d.get("Region"),
                "specialty": d.get("specialty") or d.get("Specialty"),
                "historical_deficit": d.get("historical_deficit") or d.get("Historical Deficit") or 0,
                "current_request": d.get("current_request") or d.get("Current Request") or 0,
                "initial_allocation": d.get("initial_allocation") or d.get("Initial Allocation") or 0,
                "notes": d.get("notes") or d.get("Notes") or ""
            }
            # Insert demand
            resp = supabase.table("demand_requests").insert(demand_data).execute()
            if resp.data:
                inserted += 1
                # Optionally, generate a suggestion (dummy logic)
                suggestion_data = {
                    "demand_request_id": resp.data[0]["id"],
                    "suggestion_type": "none",
                    "suggested_reduction": 0,
                    "reason": "",
                    "highlight_color": None
                }
                supabase.table("adjustment_suggestions").insert(suggestion_data).execute()
                suggestions += 1
        return {
            "success": True,
            "demands_inserted": inserted,
            "suggestions_generated": suggestions,
            "metadata": metadata
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def process_uploaded_graduates(content, supabase_client, session_id=None, default_year=None, file_name=None, sheet_name=None):
    """
    Process uploaded Excel file with graduate data and upsert into yearly_graduates.
    default_year: used when Excel has no year column.
    sheet_name: optional sheet name (e.g. "Graduates"); else first sheet.
    """
    try:
        rows = parse_excel_graduates(content, sheet_name=sheet_name, default_year=default_year)
        if not rows:
            return {"success": True, "inserted": 0, "updated": 0, "message": "No rows to import"}
        inserted = 0
        for r in rows:
            payload = {
                "year": r["year"],
                "region": r["region"],
                "specialty": r["specialty"],
                "graduate_count": r["graduate_count"],
                "source_file_name": file_name,
                "uploaded_at": datetime.utcnow().isoformat(),
            }
            if session_id:
                payload["session_id"] = session_id
            result = supabase_client.table("yearly_graduates").upsert(
                payload,
                on_conflict="year,region,specialty",
            ).execute()
            if result.data:
                inserted += 1
        return {
            "success": True,
            "inserted": inserted,
            "total_rows": len(rows),
            "metadata": {"columns_used": ["year", "region", "specialty", "graduate_count"]},
        }
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}


# --- API Endpoints ---

@app.post("/sessions/create")
async def create_session(request: CreateSessionRequest):
    """
    Create a new allocation session
    """
    try:
        session_data = {
            "session_name": request.session_name,
            "year": request.year,
            "total_budget": request.total_budget,
            "uploaded_by": request.uploaded_by,
            "status": "draft",
            "uploaded_at": datetime.utcnow().isoformat()
        }
        
        result = supabase.table("allocation_sessions").insert(session_data).execute()
        
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create session")
        
        session = result.data[0]
        
        return {
            "status": "success",
            "message": "Session created successfully",
            "session": session
        }
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/upload-file")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    """
    Upload Excel or PDF file and extract demand data
    Automatically detects file type and uses appropriate parser
    """
    try:
        # Read file content
        content = await file.read()
        
        # Determine file type
        file_extension = file.filename.lower().split('.')[-1]
        
        if file_extension in ['xlsx', 'xls']:
            # Use Excel parser
            print(f"Processing Excel file: {file.filename}")
            result = await process_uploaded_excel(content, session_id, supabase)
            
            if not result['success']:
                raise HTTPException(status_code=400, detail=result.get('error', 'Failed to process Excel'))
            
            # Update session with file info
            supabase.table("allocation_sessions").update({
                "source_file_name": file.filename,
                "status": "in_review"
            }).eq("id", session_id).execute()
            
            return {
                "status": "success",
                "message": f"Successfully processed {result['demands_inserted']} demands from Excel",
                "session_id": session_id,
                "file_name": file.filename,
                "file_type": "excel",
                "demands_count": result['demands_inserted'],
                "suggestions_count": result['suggestions_generated'],
                "metadata": result['metadata']
            }
            
        elif file_extension == 'pdf':
            # Use PDF parser (implement if needed)
            raise HTTPException(
                status_code=400,
                detail="PDF parsing not yet implemented. Please upload Excel (.xlsx) file."
            )
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Please upload .xlsx or .pdf"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("Upload error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/review-table")
async def get_review_table(session_id: str, page: int = 1, page_size: int = 50):
    """
    Get the Excel-like review table with highlighting and pagination
    """
    try:
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Get total count
        count_resp = supabase.table("demand_requests").select("*", count="exact").eq("session_id", session_id).execute()
        total_count = count_resp.count if hasattr(count_resp, 'count') else 0
        
        # Get paginated data with suggestions
        demands_resp = supabase.table("demand_requests").select("*").eq("session_id", session_id).range(offset, offset + page_size - 1).order("region", desc=False).order("specialty", desc=False).execute()
        
        demands = demands_resp.data or []
        
        # Get suggestions for these demands
        demand_ids = [d['id'] for d in demands]
        suggestions_resp = supabase.table("adjustment_suggestions").select("*").in_("demand_request_id", demand_ids).execute()
        
        suggestions_map = {}
        if suggestions_resp.data:
            for s in suggestions_resp.data:
                suggestions_map[s['demand_request_id']] = s
        
        # Merge data
        review_data = []
        for demand in demands:
            suggestion = suggestions_map.get(demand['id'], {})
            
            review_data.append({
                **demand,
                'suggestion_type': suggestion.get('suggestion_type'),
                'suggested_reduction': suggestion.get('suggested_reduction'),
                'suggestion_reason': suggestion.get('reason'),
                'highlight_color': suggestion.get('highlight_color'),
                'final_allocation': demand.get('user_allocation') or demand.get('initial_allocation', 0),
                'review_status': 'reviewed' if demand.get('user_allocation') is not None else 'pending'
            })
        
        # Get session info
        session_resp = supabase.table("allocation_sessions").select("*").eq("id", session_id).execute()
        session = session_resp.data[0] if session_resp.data else None
        
        # Calculate summary statistics
        all_demands = supabase.table("demand_requests").select("*").eq("session_id", session_id).execute()
        total_initial = sum(d.get('initial_allocation', 0) for d in all_demands.data)
        total_final = sum(d.get('user_allocation') or d.get('initial_allocation', 0) for d in all_demands.data)
        reviewed_count = sum(1 for d in all_demands.data if d.get('user_allocation') is not None)
        
        return {
            "status": "success",
            "session": session,
            "review_table": review_data,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": (total_count + page_size - 1) // page_size
            },
            "summary": {
                "total_budget": session.get('total_budget', 0) if session else 0,
                "total_initial_allocation": total_initial,
                "total_final_allocation": total_final,
                "budget_remaining": (session.get('total_budget', 0) if session else 0) - total_final,
                "reviewed_count": reviewed_count,
                "pending_count": total_count - reviewed_count,
                "review_progress_pct": round((reviewed_count / total_count * 100), 1) if total_count > 0 else 0
            }
        }
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/review-table-full")
async def get_full_review_table(session_id: str):
    """
    Get the complete review table without pagination (for export)
    WARNING: Use with caution on large datasets
    """
    try:
        # Get all demands
        demands_resp = supabase.table("demand_requests").select("*").eq("session_id", session_id).order("region", desc=False).order("specialty", desc=False).execute()
        
        demands = demands_resp.data or []
        
        # Get all suggestions
        demand_ids = [d['id'] for d in demands]
        if demand_ids:
            suggestions_resp = supabase.table("adjustment_suggestions").select("*").in_("demand_request_id", demand_ids).execute()
            
            suggestions_map = {}
            if suggestions_resp.data:
                for s in suggestions_resp.data:
                    suggestions_map[s['demand_request_id']] = s
        else:
            suggestions_map = {}
        
        # Merge data
        review_data = []
        for demand in demands:
            suggestion = suggestions_map.get(demand['id'], {})
            
            review_data.append({
                'id': demand['id'],
                'region': demand['region'],
                'specialty': demand['specialty'],
                'historical_deficit': demand['historical_deficit'],
                'current_request': demand['current_request'],
                'max_need': demand['historical_deficit'] + demand['current_request'],
                'initial_allocation': demand.get('initial_allocation', 0),
                'user_allocation': demand.get('user_allocation'),
                'final_allocation': demand.get('user_allocation') or demand.get('initial_allocation', 0),
                'deduction_amount': demand.get('initial_allocation', 0) - (demand.get('user_allocation') or demand.get('initial_allocation', 0)),
                'notes': demand.get('notes', ''),
                'suggestion_type': suggestion.get('suggestion_type'),
                'suggested_reduction': suggestion.get('suggested_reduction'),
                'suggestion_reason': suggestion.get('reason'),
                'highlight_color': suggestion.get('highlight_color'),
                'review_status': 'reviewed' if demand.get('user_allocation') is not None else 'pending'
            })
        
        return {
            "status": "success",
            "review_table": review_data,
            "total_count": len(review_data)
        }
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/demands/{demand_id}/adjust")
async def adjust_allocation(demand_id: str, request: AdjustAllocationRequest):
    """
    Adjust a single allocation
    """
    try:
        update_data = {
            "user_allocation": request.new_allocation,
            "notes": request.notes,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        result = supabase.table("demand_requests").update(update_data).eq("id", demand_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Demand request not found")
        
        return {
            "status": "success",
            "message": "Allocation adjusted successfully",
            "demand": result.data[0]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/demands/bulk-adjust")
async def bulk_adjust_allocations(request: BulkAdjustRequest):
    """
    Bulk adjust multiple allocations at once
    """
    try:
        results = []
        errors = []
        
        for adj in request.adjustments:
            try:
                result = supabase.table("demand_requests").update({
                    "user_allocation": adj.new_allocation,
                    "notes": adj.notes,
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("id", adj.demand_request_id).execute()
                
                if result.data:
                    results.append(result.data[0])
            except Exception as e:
                errors.append({
                    "demand_id": adj.demand_request_id,
                    "error": str(e)
                })
        
        return {
            "status": "success" if not errors else "partial",
            "message": f"Adjusted {len(results)} allocations",
            "updated_count": len(results),
            "error_count": len(errors),
            "updated_demands": results,
            "errors": errors if errors else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/finalize")
async def finalize_session(session_id: str, request: FinalizeSessionRequest):
    """
    Finalize the allocation session (lock it)
    """
    try:
        # Check if all demands have been reviewed
        demands = supabase.table("demand_requests").select("user_allocation").eq("session_id", session_id).execute()
        
        if demands.data:
            unreviewed = sum(1 for d in demands.data if d.get('user_allocation') is None)
            
            if unreviewed > 0:
                return {
                    "status": "warning",
                    "message": f"{unreviewed} allocations have not been reviewed. Are you sure you want to finalize?",
                    "unreviewed_count": unreviewed,
                    "total_count": len(demands.data)
                }
        
        # Update session status
        result = supabase.table("allocation_sessions").update({
            "status": "finalized",
            "finalized_at": datetime.utcnow().isoformat()
        }).eq("id", session_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Log audit entry
        supabase.table("audit_log").insert({
            "session_id": session_id,
            "action": "session_finalized",
            "changed_by": request.finalized_by,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        
        return {
            "status": "success",
            "message": "Session finalized successfully",
            "session": result.data[0]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def list_sessions(year: Optional[int] = None, status: Optional[str] = None):
    """
    List all allocation sessions with filters
    """
    try:
        query = supabase.table("allocation_sessions").select("*")
        
        if year:
            query = query.eq("year", year)
        if status:
            query = query.eq("status", status)
        
        result = query.order("created_at", desc=True).execute()
        
        return {
            "status": "success",
            "sessions": result.data,
            "count": len(result.data) if result.data else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/summary")
async def get_session_summary(session_id: str):
    """
    Get detailed session summary with statistics
    """
    try:
        # Get session
        session_resp = supabase.table("allocation_sessions").select("*").eq("id", session_id).execute()
        
        if not session_resp.data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = session_resp.data[0]
        
        # Get all demands
        demands_resp = supabase.table("demand_requests").select("*").eq("session_id", session_id).execute()
        demands = demands_resp.data or []
        
        # Calculate statistics
        total_deficit = sum(d.get('historical_deficit', 0) for d in demands)
        total_requested = sum(d.get('current_request', 0) for d in demands)
        total_initial = sum(d.get('initial_allocation', 0) for d in demands)
        total_final = sum(d.get('user_allocation') or d.get('initial_allocation', 0) for d in demands)
        total_deductions = sum(d.get('initial_allocation', 0) - (d.get('user_allocation') or d.get('initial_allocation', 0)) for d in demands)
        
        reviewed_count = sum(1 for d in demands if d.get('user_allocation') is not None)
        
        # Group by region
        by_region = {}
        for d in demands:
            region = d['region']
            if region not in by_region:
                by_region[region] = {
                    'deficit': 0,
                    'requested': 0,
                    'allocated': 0,
                    'count': 0
                }
            by_region[region]['deficit'] += d.get('historical_deficit', 0)
            by_region[region]['requested'] += d.get('current_request', 0)
            by_region[region]['allocated'] += d.get('user_allocation') or d.get('initial_allocation', 0)
            by_region[region]['count'] += 1
        
        # Group by specialty
        by_specialty = {}
        for d in demands:
            specialty = d['specialty']
            if specialty not in by_specialty:
                by_specialty[specialty] = {
                    'deficit': 0,
                    'requested': 0,
                    'allocated': 0,
                    'count': 0
                }
            by_specialty[specialty]['deficit'] += d.get('historical_deficit', 0)
            by_specialty[specialty]['requested'] += d.get('current_request', 0)
            by_specialty[specialty]['allocated'] += d.get('user_allocation') or d.get('initial_allocation', 0)
            by_specialty[specialty]['count'] += 1
        
        return {
            "status": "success",
            "session": session,
            "summary": {
                "total_demands": len(demands),
                "total_deficit": total_deficit,
                "total_requested": total_requested,
                "total_initial_allocation": total_initial,
                "total_final_allocation": total_final,
                "total_deductions": total_deductions,
                "budget": session['total_budget'],
                "budget_remaining": session['total_budget'] - total_final,
                "budget_utilization_pct": round((total_final / session['total_budget'] * 100), 1) if session['total_budget'] > 0 else 0,
                "reviewed_count": reviewed_count,
                "pending_count": len(demands) - reviewed_count,
                "review_progress_pct": round((reviewed_count / len(demands) * 100), 1) if len(demands) > 0 else 0
            },
            "by_region": by_region,
            "by_specialty": by_specialty
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/export")
async def export_session(session_id: str, format: str = "json"):
    """
    Export session data (JSON or CSV)
    """
    try:
        # Get all demands
        demands_resp = supabase.table("demand_requests").select("*").eq("session_id", session_id).execute()
        
        if not demands_resp.data:
            raise HTTPException(status_code=404, detail="No data found for session")
        
        # Format export data
        export_data = []
        for d in demands_resp.data:
            export_data.append({
                "region": d['region'],
                "specialty": d['specialty'],
                "historical_deficit": d['historical_deficit'],
                "current_request": d['current_request'],
                "max_need": d['historical_deficit'] + d['current_request'],
                "initial_allocation": d.get('initial_allocation', 0),
                "final_allocation": d.get('user_allocation') or d.get('initial_allocation', 0),
                "deduction": d.get('initial_allocation', 0) - (d.get('user_allocation') or d.get('initial_allocation', 0)),
                "notes": d.get('notes', '')
            })
        
        if format == "csv":
            import io
            import csv
            
            output = io.StringIO()
            if export_data:
                writer = csv.DictWriter(output, fieldnames=export_data[0].keys())
                writer.writeheader()
                writer.writerows(export_data)
            
            return {
                "status": "success",
                "format": "csv",
                "data": output.getvalue()
            }
        
        return {
            "status": "success",
            "format": "json",
            "data": export_data,
            "count": len(export_data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session (cascade deletes all related data)
    """
    try:
        result = supabase.table("allocation_sessions").delete().eq("id", session_id).execute()
        
        return {
            "status": "success",
            "message": "Session deleted successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Yearly Graduates ---

@app.post("/graduates/upload")
async def upload_graduates(
    file: UploadFile = File(...),
    session_id: Optional[str] = None,
    year: Optional[int] = None,
    sheet_name: Optional[str] = None,
):
    """
    Upload Excel file with yearly graduate data.
    Excel columns: region, specialty, graduate_count (or graduates/count). Optional: year (or pass ?year=2024).
    Optional: session_id to link this data to an allocation session; sheet_name to read a specific sheet.
    """
    try:
        content = await file.read()
        ext = file.filename.lower().split(".")[-1]
        if ext not in ("xlsx", "xls"):
            raise HTTPException(status_code=400, detail="Only Excel (.xlsx, .xls) files are supported.")
        result = await process_uploaded_graduates(
            content,
            supabase,
            session_id=session_id,
            default_year=year,
            file_name=file.filename,
            sheet_name=sheet_name,
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to process file"))
        return {
            "status": "success",
            "message": f"Imported {result.get('inserted', 0)} graduate records",
            "inserted": result.get("inserted", 0),
            "total_rows": result.get("total_rows", 0),
            "metadata": result.get("metadata"),
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graduates")
async def list_graduates(year: Optional[int] = None, session_id: Optional[str] = None):
    """
    List yearly graduate records. Filter by year and/or session_id.
    """
    try:
        query = supabase.table("yearly_graduates").select("*").order("year", desc=True).order("region").order("specialty")
        if year is not None:
            query = query.eq("year", year)
        if session_id is not None:
            query = query.eq("session_id", session_id)
        result = query.execute()
        data = result.data or []
        return {
            "status": "success",
            "graduates": data,
            "count": len(data),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "allocation-api-v3",
        "version": "3.0",
        "features": ["excel_upload", "pdf_upload", "regional_allocation", "yearly_graduates"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
