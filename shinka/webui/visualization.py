#!/usr/bin/env python3
"""
Shinka Visualization Module - PostgreSQL Backend

This module provides visualization capabilities for Shinka evolution results
using PostgreSQL as the database backend. All SQLite code has been removed.
"""

import argparse
import base64
import http.server
import json
import markdown
import os
import re
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None

# We'll use a simple text-to-PDF approach instead of complex dependencies
WEASYPRINT_AVAILABLE = False

DEFAULT_PORT = 8000
CACHE_EXPIRATION_SECONDS = 5  # Cache data for 5 seconds
db_cache: Dict[str, Tuple[float, Any]] = {}

# Default PostgreSQL configuration
DEFAULT_PG_CONFIG = {
    "host": os.environ.get("SHINKA_PG_HOST", "localhost"),
    "port": int(os.environ.get("SHINKA_PG_PORT", "5433")),
    "database": os.environ.get("SHINKA_PG_DATABASE", "shinka"),
    "user": os.environ.get("SHINKA_PG_USER", "shinka"),
    "password": os.environ.get("SHINKA_PG_PASSWORD", "shinka_dev"),
}


class PostgreSQLConnectionPool:
    """Manages a pool of PostgreSQL connections for the visualization server."""
    
    _instance = None
    _pool = None
    
    @classmethod
    def get_instance(cls, config: Optional[Dict[str, Any]] = None):
        """Get or create the singleton connection pool."""
        if cls._instance is None:
            cls._instance = cls(config or DEFAULT_PG_CONFIG)
        return cls._instance
    
    def __init__(self, config: Dict[str, Any]):
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 is required for PostgreSQL backend. "
                            "Install it with: pip install psycopg2-binary")
        self.config = config
        self._pool = None
        self._connect()
    
    def _connect(self):
        """Create the connection pool."""
        try:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                host=self.config["host"],
                port=self.config["port"],
                database=self.config["database"],
                user=self.config["user"],
                password=self.config["password"],
            )
            print(f"[SERVER] Connected to PostgreSQL: {self.config['database']} "
                  f"at {self.config['host']}:{self.config['port']}")
        except psycopg2.Error as e:
            print(f"[SERVER] Failed to connect to PostgreSQL: {e}")
            raise
    
    def get_connection(self):
        """Get a connection from the pool."""
        if self._pool is None:
            self._connect()
        return self._pool.getconn()
    
    def return_connection(self, conn):
        """Return a connection to the pool."""
        if self._pool:
            self._pool.putconn(conn)
    
    def close(self):
        """Close all connections in the pool."""
        if self._pool:
            self._pool.closeall()
            self._pool = None


class DatabaseRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler for database visualization endpoints."""
    
    def __init__(self, *args, search_root=None, pg_config=None, **kwargs):
        self.search_root = search_root or os.getcwd()
        self.pg_config = pg_config or DEFAULT_PG_CONFIG
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        """Override to provide more detailed logging."""
        print(f"\n[SERVER] {format % args}")

    def _get_db_connection(self):
        """Get a database connection from the pool."""
        pool = PostgreSQLConnectionPool.get_instance(self.pg_config)
        return pool.get_connection()
    
    def _return_db_connection(self, conn):
        """Return a database connection to the pool."""
        pool = PostgreSQLConnectionPool.get_instance(self.pg_config)
        pool.return_connection(conn)

    def do_GET(self):
        print(f"\n[SERVER] Received GET request for: {self.path}")
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        if path == "/list_databases":
            return self.handle_list_databases()

        if path == "/get_db_stats":
            # Support both db_path (frontend) and run_id (new API)
            run_id = query.get("run_id", query.get("db_path", [None]))[0]
            return self.handle_get_db_stats(run_id)

        if path == "/get_programs":
            # Support both db_path (frontend) and run_id (new API)
            run_id = query.get("run_id", query.get("db_path", [None]))[0]
            # Support pagination and filtering for large datasets
            limit = int(query.get("limit", [0])[0])  # 0 = no limit
            offset = int(query.get("offset", [0])[0])
            min_gen = int(query.get("min_gen", [0])[0])
            max_gen = query.get("max_gen", [None])[0]
            max_gen = int(max_gen) if max_gen else None
            # top_k: return paths to top k best programs (0 = disabled)
            top_k = int(query.get("top_k", [0])[0])
            # Legacy support: best_path_only=true is equivalent to top_k=1
            if query.get("best_path_only", ["false"])[0].lower() == "true":
                top_k = 1
            return self.handle_get_programs(
                run_id, limit=limit, offset=offset, 
                min_gen=min_gen, max_gen=max_gen,
                top_k=top_k
            )

        # Lightweight endpoint for tree rendering - returns minimal fields
        if path == "/get_programs_summary":
            run_id = query.get("run_id", query.get("db_path", [None]))[0]
            return self.handle_get_programs_summary(run_id)

        # Full details for a single program - fetched on demand when node is selected
        if path == "/get_program_details":
            program_id = query.get("id", [None])[0]
            return self.handle_get_program_details(program_id)

        if path == "/get_meta_files":
            run_id = query.get("run_id", query.get("db_path", [None]))[0]
            return self.handle_get_meta_files(run_id)

        if path == "/get_meta_content" and "generation" in query:
            run_id = query.get("run_id", [None])[0]
            generation = query["generation"][0]
            return self.handle_get_meta_content(run_id, generation)

        if path == "/download_meta_pdf" and "generation" in query:
            run_id = query.get("run_id", [None])[0]
            generation = query["generation"][0]
            return self.handle_download_meta_pdf(run_id, generation)

        if path == "/":
            print("[SERVER] Root path requested, serving viz_tree.html")
            self.path = "/viz_tree.html"

        # Handle objective function GET
        if path == "/api/objective_function":
            run_id = query.get("run_id", [None])[0]
            return self.handle_get_objective_function(run_id)

        # Handle preprompt GET
        if path == "/api/preprompt":
            run_id = query.get("run_id", [None])[0]
            return self.handle_get_preprompt(run_id)

        # Handle focus_node GET
        if path == "/api/focus_node":
            run_id = query.get("run_id", [None])[0]
            return self.handle_get_focus_node(run_id)

        # Serve static files from the webui directory
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        """Handle POST requests for modifying evolution state."""
        print(f"\n[SERVER] Received POST request for: {self.path}")
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        # Read POST body
        content_length = int(self.headers.get('Content-Length', 0))
        post_body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else "{}"
        
        try:
            post_data = json.loads(post_body)
        except json.JSONDecodeError:
            self._send_json_response({"error": "Invalid JSON"}, status_code=400)
            return
        
        if path == "/api/objective_function":
            return self.handle_set_objective_function(post_data)
        
        if path == "/api/preprompt":
            return self.handle_set_preprompt(post_data)
        
        if path == "/api/focus_node":
            return self.handle_set_focus_node(post_data)
        
        # Unknown endpoint
        self._send_json_response({"error": "Unknown endpoint"}, status_code=404)

    def handle_get_objective_function(self, run_id: Optional[str]):
        """Get the current objective function expression."""
        print(f"[SERVER] Getting objective function for run: {run_id}")
        
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get current objective function from metadata_store
            cursor.execute(
                "SELECT value FROM metadata_store WHERE key = 'objective_function'"
            )
            row = cursor.fetchone()
            
            expression = row["value"] if row else "ppl_score"
            
            # Check if raw_ppl_score column exists
            cursor.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'programs' AND column_name = 'raw_ppl_score'
            """)
            has_raw_columns = cursor.fetchone() is not None
            
            samples = []
            if has_raw_columns:
                # Get sample raw metrics for preview
                cursor.execute("""
                    SELECT raw_ppl_score, raw_code_size, raw_exec_time
                    FROM programs
                    WHERE raw_ppl_score IS NOT NULL
                    ORDER BY combined_score DESC
                    LIMIT 5
                """)
                samples = cursor.fetchall()
            
            self._send_json_response({
                "expression": expression,
                "samples": [dict(s) for s in samples],
                "has_raw_columns": has_raw_columns
            })
            
        except Exception as e:
            print(f"[SERVER] Error getting objective function: {e}")
            self._send_json_response({"error": str(e)}, status_code=500)
        finally:
            if conn:
                self._return_db_connection(conn)

    def handle_set_objective_function(self, data: Dict[str, Any]):
        """Set a new objective function and recalculate all scores."""
        expression = data.get("expression", "").strip()
        
        if not expression:
            self._send_json_response({"error": "Expression is required"}, status_code=400)
            return
        
        print(f"[SERVER] Setting objective function: {expression}")
        
        # Validate expression with sample values
        try:
            test_vars = {"ppl_score": 100.0, "code_size": 1000, "exec_time": 1.0}
            eval(expression, {"__builtins__": {}}, test_vars)
        except Exception as e:
            self._send_json_response({
                "error": f"Invalid expression: {e}"
            }, status_code=400)
            return
        
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Check if raw_ppl_score column exists
            cursor.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'programs' AND column_name = 'raw_ppl_score'
            """)
            has_raw_columns = cursor.fetchone() is not None
            
            if not has_raw_columns:
                self._send_json_response({
                    "error": "Database does not have raw metric columns. Please run migration first.",
                    "has_raw_columns": False
                }, status_code=400)
                return
            
            # Store the expression
            cursor.execute("""
                INSERT INTO metadata_store (key, value) 
                VALUES ('objective_function', %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """, (expression,))
            
            # Recalculate all scores
            cursor.execute("""
                SELECT id, raw_ppl_score, raw_code_size, raw_exec_time 
                FROM programs 
                WHERE raw_ppl_score IS NOT NULL
            """)
            rows = cursor.fetchall()
            
            updated = 0
            for row in rows:
                try:
                    ppl_score = float(row["raw_ppl_score"]) if row["raw_ppl_score"] else 0.0
                    code_size = int(row["raw_code_size"]) if row["raw_code_size"] else 0
                    exec_time = float(row["raw_exec_time"]) if row["raw_exec_time"] else 0.0
                    
                    local_vars = {
                        "ppl_score": ppl_score,
                        "code_size": code_size,
                        "exec_time": exec_time,
                    }
                    new_score = eval(expression, {"__builtins__": {}}, local_vars)
                    
                    cursor.execute(
                        "UPDATE programs SET combined_score = %s WHERE id = %s",
                        (float(new_score), row["id"])
                    )
                    updated += 1
                except Exception as e:
                    print(f"[SERVER] Failed to recalculate for {row['id']}: {e}")
            
            conn.commit()
            
            self._send_json_response({
                "success": True,
                "expression": expression,
                "programs_updated": updated
            })
            
        except Exception as e:
            print(f"[SERVER] Error setting objective function: {e}")
            if conn:
                conn.rollback()
            self._send_json_response({"error": str(e)}, status_code=500)
        finally:
            if conn:
                self._return_db_connection(conn)

    def handle_get_preprompt(self, run_id: Optional[str]):
        """Get the current preprompt for LLM generation."""
        print(f"[SERVER] Getting preprompt for run: {run_id}")
        
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get current preprompt from metadata_store
            cursor.execute(
                "SELECT value FROM metadata_store WHERE key = 'preprompt'"
            )
            row = cursor.fetchone()
            
            preprompt = row["value"] if row else ""
            
            self._send_json_response({
                "preprompt": preprompt
            })
            
        except Exception as e:
            print(f"[SERVER] Error getting preprompt: {e}")
            self._send_json_response({"error": str(e)}, status_code=500)
        finally:
            if conn:
                self._return_db_connection(conn)

    def handle_set_preprompt(self, data: Dict[str, Any]):
        """Set a new preprompt for LLM generation."""
        preprompt = data.get("preprompt", "").strip()
        
        print(f"[SERVER] Setting preprompt: {preprompt[:100]}...")
        
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Store the preprompt
            cursor.execute("""
                INSERT INTO metadata_store (key, value) 
                VALUES ('preprompt', %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """, (preprompt,))
            
            conn.commit()
            
            self._send_json_response({
                "success": True,
                "preprompt": preprompt
            })
            
        except Exception as e:
            print(f"[SERVER] Error setting preprompt: {e}")
            if conn:
                conn.rollback()
            self._send_json_response({"error": str(e)}, status_code=500)
        finally:
            if conn:
                self._return_db_connection(conn)

    def handle_get_focus_node(self, run_id: Optional[str]):
        """Get the current focus node for mutations (subtree focus)."""
        print(f"[SERVER] Getting focus node for run: {run_id}")
        
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get current focus node from metadata_store
            cursor.execute(
                "SELECT value FROM metadata_store WHERE key = 'focus_node'"
            )
            row = cursor.fetchone()
            
            focus_node_id = row["value"] if row else None
            
            # If we have a focus node, get its details
            node_info = None
            if focus_node_id:
                cursor.execute("""
                    SELECT id, metadata->>'patch_name' as agent_name, combined_score, generation
                    FROM programs
                    WHERE id = %s
                """, (focus_node_id,))
                node_row = cursor.fetchone()
                if node_row:
                    node_info = dict(node_row)
            
            self._send_json_response({
                "focus_node_id": focus_node_id,
                "node_info": node_info
            })
            
        except Exception as e:
            print(f"[SERVER] Error getting focus node: {e}")
            self._send_json_response({"error": str(e)}, status_code=500)
        finally:
            if conn:
                self._return_db_connection(conn)

    def handle_set_focus_node(self, data: Dict[str, Any]):
        """Set the focus node for mutations (subtree focus)."""
        focus_node_id = data.get("focus_node_id", None)
        
        if focus_node_id:
            print(f"[SERVER] Setting focus node: {focus_node_id[:8]}...")
        else:
            print(f"[SERVER] Clearing focus node")
        
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            if focus_node_id:
                # Store the focus node
                cursor.execute("""
                    INSERT INTO metadata_store (key, value) 
                    VALUES ('focus_node', %s)
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """, (focus_node_id,))
            else:
                # Clear the focus node
                cursor.execute("""
                    DELETE FROM metadata_store WHERE key = 'focus_node'
                """)
            
            conn.commit()
            
            self._send_json_response({
                "success": True,
                "focus_node_id": focus_node_id
            })
            
        except Exception as e:
            print(f"[SERVER] Error setting focus node: {e}")
            if conn:
                conn.rollback()
            self._send_json_response({"error": str(e)}, status_code=500)
        finally:
            if conn:
                self._return_db_connection(conn)

    def handle_list_databases(self):
        """List available evolution runs from PostgreSQL metadata table.
        
        Returns data in a format compatible with the frontend, which expects:
        - path: unique identifier for the database/run
        - name: display name
        """
        print(f"[SERVER] Listing evolution runs from PostgreSQL")
        
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # First, try to get run_id and task_name from metadata_store
            stored_run_id = None
            stored_task_name = None
            try:
                cursor.execute("SELECT key, value FROM metadata_store WHERE key IN ('run_id', 'task_name')")
                for row in cursor.fetchall():
                    if row["key"] == "run_id":
                        stored_run_id = row["value"]
                    elif row["key"] == "task_name":
                        stored_task_name = row["value"]
            except Exception as e:
                print(f"[SERVER] Could not read metadata_store: {e}")
            
            # Query the runs or metadata table for available evolution runs
            cursor.execute("""
                SELECT DISTINCT 
                    COALESCE(metadata->>'run_id', 'default') as run_id,
                    COALESCE(metadata->>'task_name', 'unknown') as task_name,
                    MIN(timestamp) as start_time,
                    MAX(timestamp) as end_time,
                    COUNT(*) as program_count,
                    MAX(generation) as max_generation
                FROM programs
                GROUP BY metadata->>'run_id', metadata->>'task_name'
                ORDER BY MAX(timestamp) DESC
            """)
            
            runs = []
            for row in cursor.fetchall():
                # Use stored values from metadata_store if program metadata is missing
                run_id = row["run_id"] if row["run_id"] != "default" else (stored_run_id or "default")
                task_name = row["task_name"] if row["task_name"] != "unknown" else (stored_task_name or "unknown")
                # Create a path that looks like the SQLite paths for frontend compatibility
                # Format: task_name/run_id/evolution_db
                path = f"{task_name}/{run_id}/evolution_db"
                
                run_info = {
                    # Frontend expects 'path' field
                    "path": path,
                    "name": f"{task_name} - {run_id[:16] if len(run_id) > 16 else run_id}",
                    # Additional metadata
                    "run_id": run_id,
                    "task_name": task_name,
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "program_count": row["program_count"],
                    "max_generation": row["max_generation"],
                }
                runs.append(run_info)
            
            if not runs:
                print("[SERVER] No evolution runs found in database.")
                # Return at least one default entry if table has data
                cursor.execute("SELECT COUNT(*) as count FROM programs")
                count = cursor.fetchone()["count"]
                if count > 0:
                    run_id = stored_run_id or "default"
                    task_name = stored_task_name or "unknown"
                    runs = [{
                        "path": f"{task_name}/{run_id}/evolution_db",
                        "name": f"{task_name} - {run_id[:16] if len(run_id) > 16 else run_id}",
                        "run_id": run_id,
                        "task_name": task_name,
                        "program_count": count,
                    }]
            
            self.send_json_response(runs)
            print(f"[SERVER] Found {len(runs)} evolution runs")
            
        except psycopg2.Error as e:
            print(f"[SERVER] Database error: {e}")
            self.send_error(500, f"Database error: {str(e)}")
        finally:
            if conn:
                self._return_db_connection(conn)

    def _parse_run_id(self, path_or_run_id: Optional[str]) -> Optional[str]:
        """Parse run_id from either a direct run_id or a path like 'task/run_id/db'.
        
        The frontend sends paths like 'circle_packing/abc123/evolution_db'
        but we need just the run_id part.
        """
        if not path_or_run_id:
            return None
        
        # If it looks like a path (contains /), extract the run_id part
        if '/' in path_or_run_id:
            parts = path_or_run_id.split('/')
            if len(parts) >= 2:
                # Format is: task_name/run_id/evolution_db
                return parts[1] if parts[1] != 'evolution_db' else parts[0]
        
        return path_or_run_id

    def handle_get_db_stats(self, run_id: Optional[str]):
        """Get database statistics for a specific run."""
        # Parse run_id from path if needed
        run_id = self._parse_run_id(run_id)
        print(f"[SERVER] Getting stats for run: {run_id}")
        
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Build run filter
            run_filter = ""
            params = []
            if run_id and run_id != "default":
                run_filter = "WHERE metadata->>'run_id' = %s"
                params = [run_id]
            
            # Get basic counts
            cursor.execute(f"SELECT COUNT(*) as total FROM programs {run_filter}", params)
            total_count = cursor.fetchone()["total"]
            
            cursor.execute(f"SELECT MAX(generation) as max_gen FROM programs {run_filter}", params)
            result = cursor.fetchone()
            max_gen = result["max_gen"] if result["max_gen"] is not None else 0
            
            where_prefix = "WHERE" if not run_filter else "AND"
            correct_query = f"""
                SELECT COUNT(*) as correct_count 
                FROM programs 
                {run_filter} {where_prefix if run_filter else 'WHERE'} correct = true
            """
            cursor.execute(correct_query.replace("WHERE AND", "WHERE"), params)
            correct_count = cursor.fetchone()["correct_count"]
            
            best_query = f"""
                SELECT MAX(combined_score) as best_score 
                FROM programs 
                {run_filter} {where_prefix if run_filter else 'WHERE'} correct = true
            """
            cursor.execute(best_query.replace("WHERE AND", "WHERE"), params)
            best_score = cursor.fetchone()["best_score"]
            
            # Get generation distribution
            gen_query = f"""
                SELECT generation, COUNT(*) as count 
                FROM programs 
                {run_filter}
                GROUP BY generation 
                ORDER BY generation
            """
            cursor.execute(gen_query, params)
            gen_distribution = {row["generation"]: row["count"] for row in cursor.fetchall()}
            
            stats = {
                "total_count": total_count,
                "max_generation": max_gen,
                "correct_count": correct_count,
                "best_score": best_score,
                "generation_distribution": gen_distribution,
                "recommended_mode": "paginated" if total_count > 2000 else "full"
            }
            
            self.send_json_response(stats)
            print(f"[SERVER] Stats: {total_count} programs, max gen {max_gen}")
            
        except psycopg2.Error as e:
            print(f"[SERVER] Database error: {e}")
            self.send_error(500, f"Database error: {str(e)}")
        finally:
            if conn:
                self._return_db_connection(conn)

    def handle_get_programs(
        self,
        run_id: Optional[str],
        limit: int = 0,
        offset: int = 0,
        min_gen: int = 0,
        max_gen: Optional[int] = None,
        top_k: int = 0
    ):
        """Fetch programs from PostgreSQL with optional filtering."""
        # Parse run_id from path if needed
        run_id = self._parse_run_id(run_id)
        print(f"[SERVER] Fetching programs (run_id={run_id}, limit={limit}, "
              f"offset={offset}, min_gen={min_gen}, max_gen={max_gen}, top_k={top_k})")
        
        # Check if we need filtering
        needs_filtering = (min_gen > 0 or max_gen is not None or top_k > 0 or limit > 0)
        
        # Check cache first (only for full, unfiltered requests)
        cache_key = f"{run_id}:{min_gen}:{max_gen}:{top_k}:{limit}:{offset}"
        use_cache = not needs_filtering
        
        if use_cache and cache_key in db_cache:
            last_fetch_time, cached_data = db_cache[cache_key]
            if time.time() - last_fetch_time < CACHE_EXPIRATION_SECONDS:
                print(f"[SERVER] Serving from cache")
                self.send_json_response(cached_data)
                return
        
        conn = None
        try:
            conn = self._get_db_connection()
            
            if top_k > 0:
                programs_dict, total_count = self._get_top_k_paths_pg(
                    conn, run_id, top_k, min_gen, max_gen
                )
            else:
                programs_dict, total_count = self._get_programs_filtered_pg(
                    conn, run_id, min_gen, max_gen, limit, offset
                )
            
            # Build response
            if needs_filtering:
                response_data = {
                    "programs": programs_dict,
                    "total_count": total_count,
                    "filtered_count": len(programs_dict),
                    "limit": limit,
                    "offset": offset
                }
                self.send_json_response(response_data)
            else:
                db_cache[cache_key] = (time.time(), programs_dict)
                self.send_json_response(programs_dict)
            
            print(f"[SERVER] Successfully served {len(programs_dict)} programs "
                  f"[filtered from {total_count}]")
            
        except psycopg2.Error as e:
            print(f"[SERVER] Database error: {e}")
            self.send_error(500, f"Database error: {str(e)}")
        finally:
            if conn:
                self._return_db_connection(conn)

    def handle_get_programs_summary(self, run_id: Optional[str]):
        """Lightweight endpoint returning only fields needed for tree rendering.
        
        This dramatically reduces payload size for initial load:
        - Full response: ~3.6MB for 275 programs
        - Summary response: ~100KB for 275 programs (35x smaller)
        """
        run_id = self._parse_run_id(run_id)
        print(f"[SERVER] Fetching programs summary (run_id={run_id})")
        
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Build WHERE clause
            where_sql = ""
            params = []
            if run_id and run_id != "default":
                where_sql = "WHERE p.metadata->>'run_id' = %s"
                params.append(run_id)
            
            # Only fetch fields needed for tree rendering
            query = f"""
                SELECT p.id, p.parent_id, p.generation, p.timestamp,
                       p.combined_score, p.correct, p.island_idx,
                       p.embedding_pca_2d, p.embedding_pca_3d, p.embedding_cluster_id,
                       p.metadata->>'patch_name' as patch_name,
                       p.metadata->>'model_name' as model_name,
                       CASE WHEN a.program_id IS NOT NULL THEN true ELSE false END as in_archive
                FROM programs p
                LEFT JOIN archive a ON p.id = a.program_id
                {where_sql}
                ORDER BY p.generation, p.timestamp
            """
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to lightweight dicts
            programs = []
            for row in rows:
                programs.append({
                    'id': row['id'],
                    'parent_id': row['parent_id'],
                    'generation': row['generation'],
                    'timestamp': row['timestamp'],
                    'combined_score': row['combined_score'],
                    'correct': row['correct'] or False,
                    'island_idx': row['island_idx'],
                    'embedding_pca_2d': row['embedding_pca_2d'] or [],
                    'embedding_pca_3d': row['embedding_pca_3d'] or [],
                    'embedding_cluster_id': row['embedding_cluster_id'],
                    'patch_name': row['patch_name'],
                    'model_name': row['model_name'],
                    'in_archive': row['in_archive'] or False,
                })
            
            self.send_json_response(programs)
            print(f"[SERVER] Served {len(programs)} program summaries")
            
        except psycopg2.Error as e:
            print(f"[SERVER] Database error: {e}")
            self.send_error(500, f"Database error: {str(e)}")
        finally:
            if conn:
                self._return_db_connection(conn)

    def handle_get_program_details(self, program_id: Optional[str]):
        """Fetch full details for a single program - called when node is selected."""
        if not program_id:
            self.send_error(400, "Missing program id parameter")
            return
        
        print(f"[SERVER] Fetching program details for: {program_id[:20]}...")
        
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Fetch all fields for this single program
            query = """
                SELECT p.*, 
                       CASE WHEN a.program_id IS NOT NULL THEN true ELSE false END as in_archive
                FROM programs p
                LEFT JOIN archive a ON p.id = a.program_id
                WHERE p.id = %s
            """
            
            cursor.execute(query, (program_id,))
            row = cursor.fetchone()
            
            if not row:
                self.send_error(404, f"Program not found: {program_id}")
                return
            
            # Convert to dict - include full metadata for single program details
            program = self._row_to_dict(row, include_full_metadata=True)
            self.send_json_response(program)
            print(f"[SERVER] Served details for program {program_id[:20]}...")
            
        except psycopg2.Error as e:
            print(f"[SERVER] Database error: {e}")
            self.send_error(500, f"Database error: {str(e)}")
        finally:
            if conn:
                self._return_db_connection(conn)

    def _get_programs_filtered_pg(
        self,
        conn,
        run_id: Optional[str],
        min_gen: int,
        max_gen: Optional[int],
        limit: int,
        offset: int
    ) -> Tuple[List[Dict], int]:
        """Fetch filtered programs from PostgreSQL."""
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Build WHERE clauses
        where_clauses = []
        params = []
        
        if run_id and run_id != "default":
            where_clauses.append("p.metadata->>'run_id' = %s")
            params.append(run_id)
        if min_gen > 0:
            where_clauses.append("p.generation >= %s")
            params.append(min_gen)
        if max_gen is not None:
            where_clauses.append("p.generation <= %s")
            params.append(max_gen)
        
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        
        # Get total count
        count_query = f"SELECT COUNT(*) as count FROM programs p {where_sql}"
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()["count"]
        
        # Get programs with archive info - exclude large embedding column for performance
        # The visualization UI uses embedding_pca_2d/3d for display, not the full embedding
        query = f"""
            SELECT p.id, p.code, p.language, p.parent_id, p.archive_inspiration_ids,
                   p.top_k_inspiration_ids, p.generation, p.timestamp, p.code_diff,
                   p.combined_score, p.public_metrics, p.private_metrics,
                   p.text_feedback, p.complexity, p.embedding_pca_2d, p.embedding_pca_3d,
                   p.embedding_cluster_id, p.correct, p.children_count, p.metadata,
                   p.island_idx, p.migration_history, p.raw_ppl_score, p.raw_code_size,
                   p.raw_exec_time, p.objective_function_used, p.preprompt,
                   CASE WHEN a.program_id IS NOT NULL THEN true ELSE false END as in_archive
            FROM programs p
            LEFT JOIN archive a ON p.id = a.program_id
            {where_sql}
            ORDER BY p.generation, p.timestamp
        """
        
        if limit > 0:
            query += f" LIMIT {limit} OFFSET {offset}"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Convert rows to dicts
        programs_dict = [self._row_to_dict(row) for row in rows]
        programs_dict = [p for p in programs_dict if p is not None]
        
        return programs_dict, total_count

    def _get_top_k_paths_pg(
        self,
        conn,
        run_id: Optional[str],
        top_k: int,
        min_gen: int,
        max_gen: Optional[int]
    ) -> Tuple[List[Dict], int]:
        """Get programs on paths to top k best solutions using PostgreSQL.
        
        Uses indexed queries to efficiently find top k correct programs,
        then traces ancestry using recursive CTE.
        """
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Build run filter
        run_filter = ""
        run_params = []
        if run_id and run_id != "default":
            run_filter = "AND metadata->>'run_id' = %s"
            run_params = [run_id]
        
        # Get total count first
        count_query = f"SELECT COUNT(*) as count FROM programs WHERE true {run_filter}"
        cursor.execute(count_query, run_params)
        total_count = cursor.fetchone()["count"]
        
        # Step 1: Find top k correct programs efficiently
        # PostgreSQL can use index on (correct, combined_score DESC)
        top_k_query = f"""
            SELECT id, parent_id, combined_score 
            FROM programs 
            WHERE correct = true 
              AND combined_score IS NOT NULL
              {run_filter}
            ORDER BY combined_score DESC
            LIMIT %s
        """
        cursor.execute(top_k_query, run_params + [top_k])
        top_k_rows = cursor.fetchall()
        
        if not top_k_rows:
            print(f"[SERVER] top_k={top_k}: No correct programs found")
            return [], total_count
        
        # Step 2: Use recursive CTE to get all ancestors efficiently
        top_k_ids = [row["id"] for row in top_k_rows]
        
        # Create a recursive CTE to trace all ancestry paths
        placeholders = ",".join(["%s"] * len(top_k_ids))
        ancestry_query = f"""
            WITH RECURSIVE ancestry AS (
                -- Base case: the top k programs
                SELECT id, parent_id, 1 as depth
                FROM programs 
                WHERE id IN ({placeholders})
                
                UNION
                
                -- Recursive case: parents of programs in ancestry
                SELECT p.id, p.parent_id, a.depth + 1
                FROM programs p
                INNER JOIN ancestry a ON p.id = a.parent_id
                WHERE a.parent_id IS NOT NULL
            )
            SELECT DISTINCT id FROM ancestry
        """
        cursor.execute(ancestry_query, top_k_ids)
        needed_ids = [row["id"] for row in cursor.fetchall()]
        
        print(f"[SERVER] top_k={top_k}: Found {len(top_k_rows)} best programs, "
              f"need {len(needed_ids)} total IDs on paths")
        
        if not needed_ids:
            return [], total_count
        
        # Step 3: Build generation filter
        gen_clauses = []
        gen_params = []
        if min_gen > 0:
            gen_clauses.append("p.generation >= %s")
            gen_params.append(min_gen)
        if max_gen is not None:
            gen_clauses.append("p.generation <= %s")
            gen_params.append(max_gen)
        gen_filter = (" AND " + " AND ".join(gen_clauses)) if gen_clauses else ""
        
        # Step 4: Fetch full program data for needed IDs
        placeholders = ",".join(["%s"] * len(needed_ids))
        data_query = f"""
            SELECT p.*,
                   CASE WHEN a.program_id IS NOT NULL THEN true ELSE false END as in_archive
            FROM programs p
            LEFT JOIN archive a ON p.id = a.program_id
            WHERE p.id IN ({placeholders}){gen_filter}
            ORDER BY p.generation, p.timestamp
        """
        cursor.execute(data_query, needed_ids + gen_params)
        rows = cursor.fetchall()
        
        # Convert to dicts
        programs_dict = [self._row_to_dict(row) for row in rows]
        programs_dict = [p for p in programs_dict if p is not None]
        
        print(f"[SERVER] top_k={top_k}: Returning {len(programs_dict)} programs on paths")
        return programs_dict, total_count

    def _row_to_dict(self, row: Dict, include_full_metadata: bool = False) -> Optional[Dict]:
        """Convert a PostgreSQL RealDictCursor row to Program dict format.
        
        Args:
            row: Database row to convert
            include_full_metadata: If True, include llm_result and other large fields in metadata.
                                   Set to True when fetching single program details.
        """
        try:
            # PostgreSQL returns JSONB as dicts/lists directly
            def ensure_list(val):
                if val is None:
                    return []
                if isinstance(val, list):
                    return val
                if isinstance(val, str):
                    try:
                        return json.loads(val)
                    except:
                        return []
                return []
            
            def ensure_dict(val):
                if val is None:
                    return {}
                if isinstance(val, dict):
                    return val
                if isinstance(val, str):
                    try:
                        return json.loads(val)
                    except:
                        return {}
                return {}
            
            # Strip large fields from metadata to reduce response size (for list views)
            # llm_result can be 20KB+ per program
            raw_metadata = ensure_dict(row.get('metadata'))
            if include_full_metadata:
                # For single program details, include everything
                stripped_metadata = raw_metadata
            else:
                # For list views, strip large fields for performance
                stripped_metadata = {k: v for k, v in raw_metadata.items() 
                                   if k not in ('llm_result', 'stdout_log', 'stderr_log', 'new_msg_history')}
            
            prog_dict = {
                'id': row['id'],
                'code': row['code'],
                'language': row['language'],
                'parent_id': row.get('parent_id'),
                'archive_inspiration_ids': ensure_list(row.get('archive_inspiration_ids')),
                'top_k_inspiration_ids': ensure_list(row.get('top_k_inspiration_ids')),
                'island_idx': row.get('island_idx'),
                'generation': row['generation'],
                'timestamp': row['timestamp'],
                'code_diff': row.get('code_diff'),
                'combined_score': row.get('combined_score'),
                'public_metrics': ensure_dict(row.get('public_metrics')),
                'private_metrics': ensure_dict(row.get('private_metrics')),
                'text_feedback': row.get('text_feedback') or '',
                'correct': row.get('correct', False),
                'children_count': row.get('children_count', 0),
                'complexity': row.get('complexity'),
                'embedding': [],  # Excluded from query for performance - use embedding_pca_* for viz
                'embedding_pca_2d': ensure_list(row.get('embedding_pca_2d')),
                'embedding_pca_3d': ensure_list(row.get('embedding_pca_3d')),
                'embedding_cluster_id': row.get('embedding_cluster_id'),
                'migration_history': ensure_list(row.get('migration_history')),
                'metadata': stripped_metadata,  # Stripped of large fields for performance
                'in_archive': row.get('in_archive', False),
                # Raw metrics for dynamic objective function
                'raw_ppl_score': row.get('raw_ppl_score'),
                'raw_code_size': row.get('raw_code_size'),
                'raw_exec_time': row.get('raw_exec_time'),
                'objective_function_used': row.get('objective_function_used'),
                # Preprompt used for LLM generation
                'preprompt': row.get('preprompt'),
            }
            return prog_dict
        except Exception as e:
            print(f"[SERVER] Error converting row to dict: {e}")
            return None

    def handle_get_meta_files(self, run_id: Optional[str]):
        """List available meta files for a given run."""
        print(f"[SERVER] Listing meta files for run: {run_id}")
        
        # Meta files are stored on filesystem, not in PostgreSQL
        # We need to find them based on run_id mapping to filesystem path
        # For now, return empty list as this needs filesystem integration
        self.send_json_response([])

    def handle_get_meta_content(self, run_id: Optional[str], generation: str):
        """Get the content of a specific meta file."""
        print(f"[SERVER] Meta content not implemented for PostgreSQL-only mode")
        self.send_error(501, "Meta file access not implemented for PostgreSQL mode")

    def handle_download_meta_pdf(self, run_id: Optional[str], generation: str):
        """Convert meta file to PDF."""
        print(f"[SERVER] Meta PDF not implemented for PostgreSQL-only mode")
        self.send_error(501, "Meta PDF access not implemented for PostgreSQL mode")

    def send_json_response(self, data, status_code: int = 200):
        """Helper to send a JSON response."""
        payload = json.dumps(data, default=self._json_encoder).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    
    def _send_json_response(self, data, status_code: int = 200):
        """Alias for send_json_response with status_code support."""
        return self.send_json_response(data, status_code)

    def _json_encoder(self, obj):
        """Custom JSON encoder to handle NaN and Inf values."""
        import math
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def create_handler_factory(search_root: str, pg_config: Dict[str, Any]):
    """Create a handler factory that passes config to handler."""
    def handler_factory(*args, **kwargs):
        return DatabaseRequestHandler(
            *args, 
            search_root=search_root, 
            pg_config=pg_config, 
            **kwargs
        )
    return handler_factory


def start_server(
    port: int, 
    search_root: str, 
    pg_config: Optional[Dict[str, Any]] = None
):
    """Start the HTTP server."""
    webui_dir = os.path.dirname(__file__)
    webui_dir = os.path.abspath(webui_dir)

    if not os.path.exists(webui_dir):
        raise FileNotFoundError(f"Webui directory not found: {webui_dir}")

    os.chdir(webui_dir)
    print(f"[DEBUG] Server root directory: {webui_dir}")
    print(f"[DEBUG] Search root directory: {search_root}")
    print(f"[DEBUG] Using PostgreSQL backend")

    # Initialize connection pool
    config = pg_config or DEFAULT_PG_CONFIG
    PostgreSQLConnectionPool.get_instance(config)

    # Create handler factory
    handler_factory = create_handler_factory(search_root, config)

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("", port), handler_factory) as httpd:
        msg = f"\n[*] Serving http://0.0.0.0:{port}  (Ctrl+C to stop)"
        print(msg)
        httpd.serve_forever()


def main():
    """Main entry point for PostgreSQL visualization server."""
    description = "Serve the Shinka visualization UI using PostgreSQL backend."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "root_directory",
        nargs="?",
        default=os.getcwd(),
        help="Root directory for meta files (default: current working directory)",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port to listen on (default: 8000).",
    )
    parser.add_argument(
        "--pg-host",
        type=str,
        default=DEFAULT_PG_CONFIG["host"],
        help="PostgreSQL host",
    )
    parser.add_argument(
        "--pg-port",
        type=int,
        default=DEFAULT_PG_CONFIG["port"],
        help="PostgreSQL port",
    )
    parser.add_argument(
        "--pg-database",
        type=str,
        default=DEFAULT_PG_CONFIG["database"],
        help="PostgreSQL database name",
    )
    parser.add_argument(
        "--pg-user",
        type=str,
        default=DEFAULT_PG_CONFIG["user"],
        help="PostgreSQL user",
    )
    parser.add_argument(
        "--pg-password",
        type=str,
        default=DEFAULT_PG_CONFIG["password"],
        help="PostgreSQL password",
    )
    parser.add_argument(
        "--open",
        dest="open_browser",
        action="store_true",
        help="Open browser on the local machine",
    )
    args = parser.parse_args()

    search_root = os.path.abspath(args.root_directory)
    
    pg_config = {
        "host": args.pg_host,
        "port": args.pg_port,
        "database": args.pg_database,
        "user": args.pg_user,
        "password": args.pg_password,
    }

    print(f"[INFO] Connecting to PostgreSQL: {pg_config['database']} "
          f"at {pg_config['host']}:{pg_config['port']}")

    # Start server in daemon thread
    server_thread = threading.Thread(
        target=start_server,
        args=(args.port, search_root, pg_config),
        daemon=True,
    )
    server_thread.start()
    time.sleep(0.8)

    viz_url = f"http://localhost:{args.port}/viz_tree.html"

    if args.open_browser:
        try:
            webbrowser.open_new_tab(viz_url)
            print(f"→ Opening {viz_url} in browser")
        except Exception as e:
            print(f"→ Could not open browser: {e}")
            print(f"→ Visit {viz_url}")
    else:
        print(f"→ Visit {viz_url}")
        print("(remember to forward the port if this is a remote host)")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Shutting down.")
        PostgreSQLConnectionPool.get_instance().close()


if __name__ == "__main__":
    main()
