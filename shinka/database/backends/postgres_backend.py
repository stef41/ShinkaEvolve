"""
PostgreSQL database backend implementation.

This module provides the PostgreSQL implementation of the DatabaseBackend interface.
PostgreSQL is recommended for production deployments with:
- High concurrency requirements
- Large datasets
- Distributed/multi-machine setups
- Advanced embedding storage with pgvector

Requires: psycopg2-binary or psycopg2
Optional: pgvector extension for vector similarity search
"""

import json
import logging
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
    from psycopg2 import sql
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None

from .base import DatabaseBackend, BackendType

logger = logging.getLogger(__name__)


def postgres_retry(max_retries=5, initial_delay=0.1, backoff_factor=2):
    """
    A decorator to retry database operations on transient PostgreSQL errors.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not PSYCOPG2_AVAILABLE:
                raise ImportError("psycopg2 is required for PostgreSQL backend")
            
            delay = initial_delay
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except psycopg2.OperationalError as e:
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
                except psycopg2.IntegrityError as e:
                    # Don't retry integrity errors
                    logger.error(f"Integrity error in {func.__name__}: {e}")
                    raise
            raise RuntimeError(
                f"DB retry logic failed for function {func.__name__}"
            )
        return wrapper
    return decorator


class PostgresCursorWrapper:
    """
    Wrapper around psycopg2 cursor that automatically adapts SQLite-style queries.
    
    This allows existing code that uses SQLite syntax (?, INSERT OR REPLACE, etc.)
    to work with PostgreSQL without modification.
    """
    
    def __init__(self, cursor, backend: 'PostgreSQLBackend'):
        self._cursor = cursor
        self._backend = backend
    
    def execute(self, query, params=None):
        """Execute a query, adapting SQLite syntax to PostgreSQL."""
        adapted_query = self._backend.adapt_query(query)
        if params:
            return self._cursor.execute(adapted_query, params)
        return self._cursor.execute(adapted_query)
    
    def executemany(self, query, params_list):
        """Execute a query multiple times with different parameters."""
        adapted_query = self._backend.adapt_query(query)
        return psycopg2.extras.execute_batch(self._cursor, adapted_query, params_list)
    
    def fetchone(self):
        return self._cursor.fetchone()
    
    def fetchall(self):
        return self._cursor.fetchall()
    
    def fetchmany(self, size=None):
        return self._cursor.fetchmany(size) if size else self._cursor.fetchmany()
    
    def close(self):
        return self._cursor.close()
    
    def __getattr__(self, name):
        # Delegate any other attribute access to the underlying cursor
        return getattr(self._cursor, name)


class PostgreSQLBackend(DatabaseBackend):
    """
    PostgreSQL implementation of the DatabaseBackend interface.
    
    This backend is ideal for:
    - Production deployments
    - High concurrency scenarios
    - Large datasets
    - Distributed systems
    - When JSONB and advanced indexing are needed
    
    Features:
    - Connection pooling for better performance
    - JSONB support for efficient JSON querying
    - Optional pgvector extension for embedding similarity search
    - Full ACID compliance with serializable isolation
    
    Attributes:
        host: Database server host
        port: Database server port
        database: Database name
        user: Database user
        password: Database password
        connection_string: Full connection string (alternative to individual params)
        min_connections: Minimum connections in pool
        max_connections: Maximum connections in pool
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "shinka",
        user: str = "postgres",
        password: str = "",
        connection_string: Optional[str] = None,
        min_connections: int = 1,
        max_connections: int = 10,
        read_only: bool = False,
        use_pgvector: bool = True,
    ):
        """
        Initialize the PostgreSQL backend.
        
        Args:
            host: Database server host
            port: Database server port
            database: Database name
            user: Database user
            password: Database password
            connection_string: Full connection string (overrides individual params)
            min_connections: Minimum connections in pool
            max_connections: Maximum connections in pool
            read_only: Open in read-only mode
            use_pgvector: Whether to use pgvector extension for embeddings
        """
        if not PSYCOPG2_AVAILABLE:
            raise ImportError(
                "PostgreSQL backend requires psycopg2. "
                "Install it with: pip install psycopg2-binary"
            )
        
        super().__init__(read_only=read_only)
        
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.connection_string = connection_string
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.use_pgvector = use_pgvector
        
        self._pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        self.conn: Optional[Any] = None
        self.cursor: Optional[Any] = None
        self._in_transaction = False
        self._pgvector_available = False
    
    def _get_connection_params(self) -> Dict[str, Any]:
        """Get connection parameters as a dictionary."""
        if self.connection_string:
            # Parse connection string
            parsed = urlparse(self.connection_string)
            return {
                "host": parsed.hostname or "localhost",
                "port": parsed.port or 5432,
                "database": parsed.path.lstrip("/") or "shinka",
                "user": parsed.username or "postgres",
                "password": parsed.password or "",
            }
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
        }
    
    def connect(self) -> None:
        """Establish connection to PostgreSQL database."""
        if self._connected:
            return
        
        params = self._get_connection_params()
        
        try:
            # Create connection pool
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                self.min_connections,
                self.max_connections,
                **params
            )
            
            # Get a connection from the pool
            self.conn = self._pool.getconn()
            self.conn.autocommit = False
            
            # Create cursor with dictionary-like row factory
            self.cursor = self.conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            )
            
            self._connected = True
            logger.info(
                f"Connected to PostgreSQL database: {params['database']} "
                f"at {params['host']}:{params['port']}"
            )
            
            # Check for pgvector extension
            if self.use_pgvector:
                self._check_pgvector()
            
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def _check_pgvector(self) -> None:
        """Check if pgvector extension is available and enable it."""
        try:
            self.cursor.execute(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            )
            result = self.cursor.fetchone()
            
            if result and result.get('exists'):
                self._pgvector_available = True
                logger.info("pgvector extension is available")
            else:
                # Try to create the extension
                if not self.read_only:
                    try:
                        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                        self.conn.commit()
                        self._pgvector_available = True
                        logger.info("pgvector extension created successfully")
                    except psycopg2.Error as e:
                        logger.warning(
                            f"Could not create pgvector extension: {e}. "
                            "Embeddings will be stored as JSONB."
                        )
                        self.conn.rollback()
                else:
                    logger.info(
                        "pgvector not available and read-only mode. "
                        "Embeddings will be stored as JSONB."
                    )
        except psycopg2.Error as e:
            logger.warning(f"Error checking for pgvector: {e}")
            self.conn.rollback()
    
    def close(self) -> None:
        """Close the database connection and return it to the pool."""
        if self.cursor:
            self.cursor.close()
            self.cursor = None
        
        if self.conn and self._pool:
            self._pool.putconn(self.conn)
            self.conn = None
        
        if self._pool:
            self._pool.closeall()
            self._pool = None
        
        self._connected = False
        logger.debug("PostgreSQL connection closed.")
    
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
    
    @postgres_retry()
    def execute(self, query: str, params: Optional[Tuple] = None) -> Any:
        """Execute a query with optional parameters."""
        if not self.cursor:
            raise ConnectionError("Database not connected.")
        
        # Convert SQLite-style placeholders to PostgreSQL style
        adapted_query = self.adapt_query(query)
        
        if params:
            self.cursor.execute(adapted_query, params)
        else:
            self.cursor.execute(adapted_query)
        
        return self.cursor
    
    @postgres_retry()
    def executemany(self, query: str, params_list: List[Tuple]) -> Any:
        """Execute a query multiple times with different parameters."""
        if not self.cursor:
            raise ConnectionError("Database not connected.")
        
        adapted_query = self.adapt_query(query)
        
        # Use execute_batch for better performance
        psycopg2.extras.execute_batch(self.cursor, adapted_query, params_list)
        
        return self.cursor
    
    def fetchone(self) -> Optional[Dict[str, Any]]:
        """Fetch one row from the last executed query."""
        if not self.cursor:
            return None
        result = self.cursor.fetchone()
        return dict(result) if result else None
    
    def fetchall(self) -> List[Dict[str, Any]]:
        """Fetch all rows from the last executed query."""
        if not self.cursor:
            return []
        results = self.cursor.fetchall()
        return [dict(row) for row in results]
    
    def fetchmany(self, size: int) -> List[Dict[str, Any]]:
        """Fetch specified number of rows from the last executed query."""
        if not self.cursor:
            return []
        results = self.cursor.fetchmany(size)
        return [dict(row) for row in results]
    
    def create_tables(self, schema: str) -> None:
        """Create tables based on the provided schema."""
        if not self.cursor or not self.conn:
            raise ConnectionError("Database not connected.")
        
        # Adapt schema for PostgreSQL
        adapted_schema = self._adapt_schema_for_postgres(schema)
        
        # Split schema into individual statements and execute each
        statements = [s.strip() for s in adapted_schema.split(';') if s.strip()]
        for statement in statements:
            self.cursor.execute(statement)
        
        self.conn.commit()
    
    def _adapt_schema_for_postgres(self, schema: str) -> str:
        """Convert SQLite schema to PostgreSQL schema."""
        adapted = schema
        
        # Replace AUTOINCREMENT with SERIAL
        adapted = adapted.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
        adapted = adapted.replace("INTEGER PRIMARY KEY", "SERIAL PRIMARY KEY")
        
        # Replace BOOLEAN with actual BOOLEAN (SQLite uses INTEGER for booleans)
        # PostgreSQL has native BOOLEAN support
        
        # Replace TEXT PRIMARY KEY with proper syntax
        adapted = adapted.replace("TEXT PRIMARY KEY", "TEXT PRIMARY KEY")
        
        # Replace SQLite's PRAGMA statements (they're not needed in PostgreSQL)
        import re
        adapted = re.sub(r'PRAGMA\s+[^;]+;?', '', adapted)
        
        # Replace CREATE INDEX IF NOT EXISTS
        adapted = adapted.replace(
            "CREATE INDEX IF NOT EXISTS",
            "CREATE INDEX IF NOT EXISTS"
        )
        
        return adapted
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        if not self.cursor:
            return False
        
        self.cursor.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            )
            """,
            (table_name,)
        )
        result = self.cursor.fetchone()
        return result.get('exists', False) if result else False
    
    def column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table."""
        if not self.cursor:
            return False
        
        self.cursor.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = %s 
                AND column_name = %s
            )
            """,
            (table_name, column_name)
        )
        result = self.cursor.fetchone()
        return result.get('exists', False) if result else False
    
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
        
        # Adapt column type for PostgreSQL
        pg_type = self._adapt_type_for_postgres(column_type)
        
        default_clause = f" DEFAULT {default}" if default is not None else ""
        query = f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {column_name} {pg_type}{default_clause}"
        
        self.cursor.execute(query)
        self.conn.commit()
    
    def _adapt_type_for_postgres(self, sqlite_type: str) -> str:
        """Convert SQLite type to PostgreSQL type."""
        type_mapping = {
            "TEXT": "TEXT",
            "INTEGER": "INTEGER",
            "REAL": "DOUBLE PRECISION",
            "BLOB": "BYTEA",
            "BOOLEAN": "BOOLEAN",
        }
        
        upper_type = sqlite_type.upper()
        return type_mapping.get(upper_type, sqlite_type)
    
    def get_placeholder(self) -> str:
        """Get the parameter placeholder for PostgreSQL."""
        return "%s"
    
    def adapt_query(self, query: str) -> str:
        """
        Convert SQLite-style query to PostgreSQL-style.
        
        Converts:
        - '?' placeholders to '%s'
        - SQLite-specific functions to PostgreSQL equivalents
        - INSERT OR REPLACE/IGNORE to PostgreSQL ON CONFLICT
        """
        import re
        adapted = query
        
        # Replace ? placeholders with %s
        adapted = adapted.replace("?", "%s")
        
        # Replace SQLite's json_extract with PostgreSQL's JSONB operators
        adapted = re.sub(
            r"json_extract\((\w+),\s*'\$\.(\w+)'\)",
            r"\1::jsonb->'\2'",
            adapted
        )
        
        # Replace INSERT OR REPLACE with INSERT ... ON CONFLICT DO UPDATE
        # Handle metadata_store case: INSERT OR REPLACE INTO metadata_store (key, value)
        if re.search(r"INSERT\s+OR\s+REPLACE\s+INTO\s+metadata_store", adapted, re.IGNORECASE):
            adapted = re.sub(
                r"INSERT\s+OR\s+REPLACE\s+INTO\s+metadata_store\s*\((.*?)\)\s*VALUES\s*\((.*?)\)",
                r"INSERT INTO metadata_store (\1) VALUES (\2) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                adapted,
                flags=re.IGNORECASE | re.DOTALL
            )
        
        # Replace INSERT OR IGNORE with INSERT ... ON CONFLICT DO NOTHING
        if re.search(r"INSERT\s+OR\s+IGNORE\s+INTO", adapted, re.IGNORECASE):
            adapted = re.sub(
                r"INSERT\s+OR\s+IGNORE\s+INTO\s+(\w+)\s*\((.*?)\)\s*VALUES\s*\((.*?)\)",
                r"INSERT INTO \1 (\2) VALUES (\3) ON CONFLICT DO NOTHING",
                adapted,
                flags=re.IGNORECASE | re.DOTALL
            )
        
        # Convert SQLite integer boolean comparisons to PostgreSQL BOOLEAN
        # correct = 1 -> correct = TRUE, correct = 0 -> correct = FALSE
        adapted = re.sub(r'\bcorrect\s*=\s*1\b', 'correct = TRUE', adapted)
        adapted = re.sub(r'\bcorrect\s*=\s*0\b', 'correct = FALSE', adapted)
        
        return adapted
    
    def get_json_extract(self, column: str, key: str) -> str:
        """Get PostgreSQL-specific JSONB extraction syntax."""
        return f"{column}::jsonb->>'{key}'"
    
    def serialize_json(self, data: Any) -> str:
        """Serialize data to JSON string."""
        return json.dumps(data)
    
    def deserialize_json(self, json_str: Union[str, dict]) -> Any:
        """Deserialize JSON string or dict."""
        if isinstance(json_str, dict):
            return json_str  # Already deserialized (PostgreSQL JSONB)
        if not json_str:
            return None
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to deserialize JSON: {str(json_str)[:100]}...")
            return None
    
    def begin_transaction(self) -> None:
        """Begin a new transaction."""
        if self.conn and not self._in_transaction:
            # PostgreSQL starts transactions automatically
            self._in_transaction = True
    
    def create_savepoint(self, name: str) -> None:
        """Create a savepoint within the current transaction."""
        if self.cursor:
            self.cursor.execute(f"SAVEPOINT {name}")
    
    def rollback_to_savepoint(self, name: str) -> None:
        """Rollback to a savepoint."""
        if self.cursor:
            self.cursor.execute(f"ROLLBACK TO SAVEPOINT {name}")
    
    def release_savepoint(self, name: str) -> None:
        """Release a savepoint."""
        if self.cursor:
            self.cursor.execute(f"RELEASE SAVEPOINT {name}")
    
    def get_backend_type(self) -> BackendType:
        """Get the backend type."""
        return BackendType.POSTGRES
    
    def create_connection_for_thread(self) -> 'PostgreSQLBackend':
        """Create a new connection for use in a different thread."""
        if not self._pool:
            raise ConnectionError("Connection pool not initialized.")
        
        # Create a new backend that will use the same pool
        new_backend = PostgreSQLBackend(
            connection_string=self.connection_string,
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
            min_connections=1,
            max_connections=1,
            read_only=self.read_only,
            use_pgvector=self.use_pgvector,
        )
        new_backend.connect()
        return new_backend
    
    def set_busy_timeout(self, timeout_ms: int) -> None:
        """Set statement timeout for the connection."""
        if self.cursor:
            self.cursor.execute(f"SET statement_timeout = {timeout_ms}")
    
    def optimize_for_bulk_insert(self) -> None:
        """Apply optimizations for bulk insert operations."""
        if self.cursor and not self.read_only:
            # Disable synchronous commit for bulk operations
            self.cursor.execute("SET synchronous_commit = OFF")
            # Increase work_mem for better performance
            self.cursor.execute("SET work_mem = '256MB'")
    
    def restore_after_bulk_insert(self) -> None:
        """Restore normal operation after bulk inserts."""
        if self.cursor and not self.read_only:
            self.cursor.execute("SET synchronous_commit = ON")
            self.cursor.execute("RESET work_mem")
    
    def get_raw_connection(self) -> Any:
        """Get the raw psycopg2 connection for advanced operations."""
        if not self.conn:
            raise ConnectionError("Database not connected.")
        return self.conn
    
    def get_raw_cursor(self) -> Any:
        """
        Get a cursor wrapper that adapts SQLite-style queries to PostgreSQL.
        
        This allows code using SQLite syntax (?, INSERT OR REPLACE, etc.)
        to work with PostgreSQL transparently.
        """
        if not self.cursor:
            raise ConnectionError("Database not connected.")
        return PostgresCursorWrapper(self.cursor, self)
    
    def is_pgvector_available(self) -> bool:
        """Check if pgvector extension is available."""
        return self._pgvector_available
    
    def create_vector_index(
        self,
        table_name: str,
        column_name: str,
        dimensions: int,
        index_type: str = "ivfflat"
    ) -> None:
        """
        Create a vector similarity index using pgvector.
        
        Args:
            table_name: Name of the table
            column_name: Name of the vector column
            dimensions: Number of dimensions in the vector
            index_type: Type of index ('ivfflat' or 'hnsw')
        """
        if not self._pgvector_available:
            logger.warning("pgvector not available, skipping index creation")
            return
        
        if not self.cursor or not self.conn:
            raise ConnectionError("Database not connected.")
        
        # First, alter the column to vector type if needed
        # This is a simplified version; actual implementation would need
        # to check current column type
        
        index_name = f"idx_{table_name}_{column_name}_vector"
        
        if index_type == "ivfflat":
            self.cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {table_name}
                USING ivfflat ({column_name} vector_cosine_ops)
                WITH (lists = 100)
                """
            )
        elif index_type == "hnsw":
            self.cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {table_name}
                USING hnsw ({column_name} vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                """
            )
        
        self.conn.commit()
        logger.info(f"Created {index_type} vector index on {table_name}.{column_name}")
    
    def vector_similarity_search(
        self,
        table_name: str,
        column_name: str,
        query_vector: List[float],
        limit: int = 10,
        where_clause: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using pgvector.
        
        Args:
            table_name: Name of the table
            column_name: Name of the vector column
            query_vector: Query vector for similarity search
            limit: Maximum number of results
            where_clause: Optional WHERE clause for filtering
            
        Returns:
            List of matching rows with similarity scores
        """
        if not self._pgvector_available:
            logger.warning("pgvector not available, using fallback")
            return []
        
        if not self.cursor:
            raise ConnectionError("Database not connected.")
        
        vector_str = f"[{','.join(map(str, query_vector))}]"
        
        where_part = f"WHERE {where_clause}" if where_clause else ""
        
        query = f"""
            SELECT *, 1 - ({column_name} <=> %s::vector) as similarity
            FROM {table_name}
            {where_part}
            ORDER BY {column_name} <=> %s::vector
            LIMIT %s
        """
        
        self.cursor.execute(query, (vector_str, vector_str, limit))
        return self.fetchall()
    
    def vacuum_analyze(self, table_name: Optional[str] = None) -> None:
        """Run VACUUM ANALYZE on the database or a specific table."""
        if self.read_only:
            return
        
        # VACUUM requires autocommit mode
        old_autocommit = self.conn.autocommit
        self.conn.autocommit = True
        
        try:
            if table_name:
                self.cursor.execute(f"VACUUM ANALYZE {table_name}")
            else:
                self.cursor.execute("VACUUM ANALYZE")
        finally:
            self.conn.autocommit = old_autocommit
