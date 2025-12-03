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
    "port": int(os.environ.get("SHINKA_PG_PORT", "5432")),
    "database": os.environ.get("SHINKA_PG_DATABASE", "shinka"),
    "user": os.environ.get("SHINKA_PG_USER", "postgres"),
    "password": os.environ.get("SHINKA_PG_PASSWORD", ""),
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

        # Serve static files from the webui directory
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

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
                run_id = row["run_id"] or "default"
                task_name = row["task_name"] or "unknown"
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
                    runs = [{
                        "path": "unknown/default/evolution_db",
                        "name": "Default Run",
                        "run_id": "default",
                        "task_name": "unknown",
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
        
        # Get programs with archive info
        query = f"""
            SELECT p.*,
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

    def _row_to_dict(self, row: Dict) -> Optional[Dict]:
        """Convert a PostgreSQL RealDictCursor row to Program dict format."""
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
                'embedding': ensure_list(row.get('embedding')),
                'embedding_pca_2d': ensure_list(row.get('embedding_pca_2d')),
                'embedding_pca_3d': ensure_list(row.get('embedding_pca_3d')),
                'embedding_cluster_id': row.get('embedding_cluster_id'),
                'migration_history': ensure_list(row.get('migration_history')),
                'metadata': ensure_dict(row.get('metadata')),
                'in_archive': row.get('in_archive', False),
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

    def send_json_response(self, data):
        """Helper to send a JSON response."""
        payload = json.dumps(data, default=self._json_encoder).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

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
