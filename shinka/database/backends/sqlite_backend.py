"""
SQLite database backend implementation.

This module provides the SQLite implementation of the DatabaseBackend interface.
SQLite is the default backend for Shinka and works well for single-machine deployments.
"""

import json
import logging
import sqlite3
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import DatabaseBackend, BackendType

logger = logging.getLogger(__name__)


def sqlite_retry(max_retries=5, initial_delay=0.1, backoff_factor=2):
    """
    A decorator to retry database operations on specific SQLite errors.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (
                    sqlite3.OperationalError,
                    sqlite3.DatabaseError,
                    sqlite3.IntegrityError,
                ) as e:
                    if i == max_retries - 1:
                        logger.error(
                            f"DB operation {func.__name__} failed after "
                            f"{max_retries} retries: {e}"
                        )
                        raise
                    logger.warning(
                        f"DB operation {func.__name__} failed with "
                        f"{type(e).__name__}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
            raise RuntimeError(
                f"DB retry logic failed for function {func.__name__} without "
                "raising an exception."
            )
        return wrapper
    return decorator


class SQLiteBackend(DatabaseBackend):
    """
    SQLite implementation of the DatabaseBackend interface.
    
    This backend is ideal for:
    - Single-machine deployments
    - Development and testing
    - Small to medium-sized datasets
    - When simplicity and zero configuration are priorities
    
    Attributes:
        db_path: Path to the SQLite database file (or ":memory:" for in-memory)
        timeout: Connection timeout in seconds
        read_only: Whether to open in read-only mode
    """
    
    def __init__(
        self,
        db_path: str = ":memory:",
        timeout: float = 30.0,
        read_only: bool = False,
        check_same_thread: bool = True,
    ):
        """
        Initialize the SQLite backend.
        
        Args:
            db_path: Path to database file or ":memory:" for in-memory
            timeout: Connection timeout in seconds
            read_only: Open in read-only mode
            check_same_thread: If True, raise error when connection is used in
                             different thread (set False for thread-safe operations)
        """
        super().__init__(read_only=read_only)
        self.db_path = db_path
        self.timeout = timeout
        self.check_same_thread = check_same_thread
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
        self._in_transaction = False
    
    def connect(self) -> None:
        """Establish connection to SQLite database."""
        if self._connected:
            return
        
        if self.db_path != ":memory:":
            db_file = Path(self.db_path).resolve()
            
            if not self.read_only:
                # Handle WAL recovery for unclean shutdown
                self._handle_wal_recovery(db_file)
                db_file.parent.mkdir(parents=True, exist_ok=True)
                self.conn = sqlite3.connect(
                    str(db_file),
                    timeout=self.timeout,
                    check_same_thread=self.check_same_thread,
                )
                logger.debug(f"Connected to SQLite database: {db_file}")
            else:
                if not db_file.exists():
                    raise FileNotFoundError(
                        f"Database file not found for read-only connection: {db_file}"
                    )
                db_uri = f"file:{db_file}?mode=ro"
                self.conn = sqlite3.connect(
                    db_uri,
                    uri=True,
                    timeout=self.timeout,
                    check_same_thread=self.check_same_thread,
                )
                logger.debug(f"Connected to SQLite database (read-only): {db_file}")
        else:
            self.conn = sqlite3.connect(
                ":memory:",
                check_same_thread=self.check_same_thread,
            )
            logger.info("Initialized in-memory SQLite database.")
        
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self._connected = True
        
        # Apply default pragmas for better performance
        if not self.read_only:
            self._apply_performance_pragmas()
    
    def _handle_wal_recovery(self, db_file: Path) -> None:
        """Handle WAL file cleanup for unclean shutdown recovery."""
        db_wal_file = Path(f"{db_file}-wal")
        db_shm_file = Path(f"{db_file}-shm")
        
        if (
            db_file.exists()
            and db_file.stat().st_size == 0
            and (db_wal_file.exists() or db_shm_file.exists())
        ):
            logger.warning(
                f"Database file {db_file} is empty but WAL/SHM files "
                "exist. This may indicate an unclean shutdown. "
                "Removing WAL/SHM files to attempt recovery."
            )
            if db_wal_file.exists():
                db_wal_file.unlink()
            if db_shm_file.exists():
                db_shm_file.unlink()
    
    def _apply_performance_pragmas(self) -> None:
        """Apply SQLite pragmas for better performance and stability."""
        if not self.cursor:
            return
        
        pragmas = [
            "PRAGMA journal_mode = WAL;",
            "PRAGMA busy_timeout = 30000;",
            "PRAGMA wal_autocheckpoint = 1000;",
            "PRAGMA synchronous = NORMAL;",
            "PRAGMA cache_size = -64000;",  # 64MB cache
            "PRAGMA temp_store = MEMORY;",
            "PRAGMA foreign_keys = ON;",
        ]
        
        for pragma in pragmas:
            self.cursor.execute(pragma)
        
        self.conn.commit()
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
            self._connected = False
            logger.debug("SQLite connection closed.")
    
    def commit(self) -> None:
        """Commit the current transaction."""
        if self.conn:
            self.conn.commit()
            self._in_transaction = False
    
    def rollback(self) -> None:
        """Rollback the current transaction."""
        if self.conn:
            self.conn.rollback()
            self._in_transaction = False
    
    @sqlite_retry()
    def execute(self, query: str, params: Optional[Tuple] = None) -> sqlite3.Cursor:
        """Execute a query with optional parameters."""
        if not self.cursor:
            raise ConnectionError("Database not connected.")
        
        if params:
            return self.cursor.execute(query, params)
        return self.cursor.execute(query)
    
    @sqlite_retry()
    def executemany(self, query: str, params_list: List[Tuple]) -> sqlite3.Cursor:
        """Execute a query multiple times with different parameters."""
        if not self.cursor:
            raise ConnectionError("Database not connected.")
        return self.cursor.executemany(query, params_list)
    
    def fetchone(self) -> Optional[Dict[str, Any]]:
        """Fetch one row from the last executed query."""
        if not self.cursor:
            return None
        row = self.cursor.fetchone()
        return dict(row) if row else None
    
    def fetchall(self) -> List[Dict[str, Any]]:
        """Fetch all rows from the last executed query."""
        if not self.cursor:
            return []
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]
    
    def fetchmany(self, size: int) -> List[Dict[str, Any]]:
        """Fetch specified number of rows from the last executed query."""
        if not self.cursor:
            return []
        rows = self.cursor.fetchmany(size)
        return [dict(row) for row in rows]
    
    def create_tables(self, schema: str) -> None:
        """Create tables based on the provided schema."""
        if not self.cursor or not self.conn:
            raise ConnectionError("Database not connected.")
        
        # Split schema into individual statements and execute each
        statements = [s.strip() for s in schema.split(';') if s.strip()]
        for statement in statements:
            self.cursor.execute(statement)
        
        self.conn.commit()
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        if not self.cursor:
            return False
        
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return self.cursor.fetchone() is not None
    
    def column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table."""
        if not self.cursor:
            return False
        
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in self.cursor.fetchall()]
        return column_name in columns
    
    def add_column(
        self,
        table_name: str,
        column_name: str,
        column_type: str,
        default: Optional[str] = None
    ) -> None:
        """Add a column to an existing table."""
        if not self.cursor or not self.conn:
            raise ConnectionError("Database not connected.")
        
        default_clause = f" DEFAULT {default}" if default is not None else ""
        query = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}{default_clause}"
        
        self.cursor.execute(query)
        self.conn.commit()
    
    def get_placeholder(self) -> str:
        """Get the parameter placeholder for SQLite."""
        return "?"
    
    def get_json_extract(self, column: str, key: str) -> str:
        """Get SQLite-specific JSON extraction syntax."""
        return f"json_extract({column}, '$.{key}')"
    
    def serialize_json(self, data: Any) -> str:
        """Serialize data to JSON string."""
        return json.dumps(data)
    
    def deserialize_json(self, json_str: str) -> Any:
        """Deserialize JSON string."""
        if not json_str:
            return None
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to deserialize JSON: {json_str[:100]}...")
            return None
    
    def begin_transaction(self) -> None:
        """Begin a new transaction."""
        if self.conn and not self._in_transaction:
            self.conn.execute("BEGIN TRANSACTION")
            self._in_transaction = True
    
    def create_savepoint(self, name: str) -> None:
        """Create a savepoint within the current transaction."""
        if self.conn:
            self.conn.execute(f"SAVEPOINT {name}")
    
    def rollback_to_savepoint(self, name: str) -> None:
        """Rollback to a savepoint."""
        if self.conn:
            self.conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
    
    def release_savepoint(self, name: str) -> None:
        """Release a savepoint."""
        if self.conn:
            self.conn.execute(f"RELEASE SAVEPOINT {name}")
    
    def get_backend_type(self) -> BackendType:
        """Get the backend type."""
        return BackendType.SQLITE
    
    def create_connection_for_thread(self) -> 'SQLiteBackend':
        """Create a new connection for use in a different thread."""
        new_backend = SQLiteBackend(
            db_path=self.db_path,
            timeout=self.timeout,
            read_only=self.read_only,
            check_same_thread=False,  # Allow cross-thread usage
        )
        new_backend.connect()
        return new_backend
    
    def set_busy_timeout(self, timeout_ms: int) -> None:
        """Set the busy timeout for the database connection."""
        if self.cursor:
            self.cursor.execute(f"PRAGMA busy_timeout = {timeout_ms};")
    
    def optimize_for_bulk_insert(self) -> None:
        """Apply optimizations for bulk insert operations."""
        if self.cursor and not self.read_only:
            self.cursor.execute("PRAGMA synchronous = OFF;")
            self.cursor.execute("PRAGMA journal_mode = MEMORY;")
    
    def restore_after_bulk_insert(self) -> None:
        """Restore normal operation after bulk inserts."""
        if self.cursor and not self.read_only:
            self.cursor.execute("PRAGMA synchronous = NORMAL;")
            self.cursor.execute("PRAGMA journal_mode = WAL;")
    
    def get_last_insert_rowid(self) -> Optional[int]:
        """Get the row ID of the last inserted row."""
        if self.cursor:
            return self.cursor.lastrowid
        return None
    
    def vacuum(self) -> None:
        """Vacuum the database to reclaim space."""
        if self.conn and not self.read_only:
            self.conn.execute("VACUUM")
    
    def analyze(self) -> None:
        """Analyze the database to update statistics."""
        if self.conn and not self.read_only:
            self.conn.execute("ANALYZE")
    
    def get_raw_connection(self) -> sqlite3.Connection:
        """Get the raw SQLite connection for advanced operations."""
        if not self.conn:
            raise ConnectionError("Database not connected.")
        return self.conn
    
    def get_raw_cursor(self) -> sqlite3.Cursor:
        """Get the raw SQLite cursor for advanced operations."""
        if not self.cursor:
            raise ConnectionError("Database not connected.")
        return self.cursor
