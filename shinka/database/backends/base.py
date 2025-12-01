"""
Abstract base class for database backends.

This module defines the interface that all database backends must implement.
"""

import json
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Supported database backend types."""
    SQLITE = "sqlite"
    POSTGRES = "postgres"


class DatabaseBackend(ABC):
    """
    Abstract base class for database backends.
    
    All database backends must implement these methods to ensure
    compatibility with the ProgramDatabase class.
    """
    
    def __init__(self, read_only: bool = False):
        self.read_only = read_only
        self._connected = False
    
    @property
    def is_connected(self) -> bool:
        """Check if the backend is connected."""
        return self._connected
    
    @abstractmethod
    def connect(self) -> None:
        """Establish a connection to the database."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        pass
    
    @abstractmethod
    def commit(self) -> None:
        """Commit the current transaction."""
        pass
    
    @abstractmethod
    def rollback(self) -> None:
        """Rollback the current transaction."""
        pass
    
    @abstractmethod
    def execute(self, query: str, params: Optional[Tuple] = None) -> Any:
        """
        Execute a query with optional parameters.
        
        Args:
            query: SQL query string (may use backend-specific placeholders)
            params: Query parameters
            
        Returns:
            Cursor or result object
        """
        pass
    
    @abstractmethod
    def executemany(self, query: str, params_list: List[Tuple]) -> Any:
        """
        Execute a query multiple times with different parameters.
        
        Args:
            query: SQL query string
            params_list: List of parameter tuples
            
        Returns:
            Cursor or result object
        """
        pass
    
    @abstractmethod
    def fetchone(self) -> Optional[Dict[str, Any]]:
        """Fetch one row from the last executed query."""
        pass
    
    @abstractmethod
    def fetchall(self) -> List[Dict[str, Any]]:
        """Fetch all rows from the last executed query."""
        pass
    
    @abstractmethod
    def fetchmany(self, size: int) -> List[Dict[str, Any]]:
        """Fetch specified number of rows from the last executed query."""
        pass
    
    @abstractmethod
    def create_tables(self, schema: str) -> None:
        """
        Create tables based on the provided schema.
        
        Args:
            schema: SQL schema definition
        """
        pass
    
    @abstractmethod
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        pass
    
    @abstractmethod
    def column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table."""
        pass
    
    @abstractmethod
    def add_column(
        self, 
        table_name: str, 
        column_name: str, 
        column_type: str,
        default: Optional[str] = None
    ) -> None:
        """Add a column to an existing table."""
        pass
    
    @abstractmethod
    def get_placeholder(self) -> str:
        """
        Get the parameter placeholder for this backend.
        
        Returns:
            '?' for SQLite, '%s' for PostgreSQL
        """
        pass
    
    @abstractmethod
    def get_json_extract(self, column: str, key: str) -> str:
        """
        Get backend-specific JSON extraction syntax.
        
        Args:
            column: Column name containing JSON data
            key: Key to extract from JSON
            
        Returns:
            SQL expression for JSON extraction
        """
        pass
    
    @abstractmethod
    def serialize_json(self, data: Any) -> str:
        """
        Serialize data to JSON string for storage.
        
        Args:
            data: Data to serialize
            
        Returns:
            JSON string
        """
        pass
    
    @abstractmethod
    def deserialize_json(self, json_str: str) -> Any:
        """
        Deserialize JSON string from storage.
        
        Args:
            json_str: JSON string
            
        Returns:
            Deserialized data
        """
        pass
    
    @abstractmethod
    def begin_transaction(self) -> None:
        """Begin a new transaction."""
        pass
    
    @abstractmethod
    def create_savepoint(self, name: str) -> None:
        """Create a savepoint within the current transaction."""
        pass
    
    @abstractmethod
    def rollback_to_savepoint(self, name: str) -> None:
        """Rollback to a savepoint."""
        pass
    
    @abstractmethod
    def release_savepoint(self, name: str) -> None:
        """Release a savepoint."""
        pass
    
    @abstractmethod
    def get_backend_type(self) -> BackendType:
        """Get the backend type."""
        pass
    
    @abstractmethod
    def create_connection_for_thread(self) -> 'DatabaseBackend':
        """
        Create a new connection for use in a different thread.
        
        This is needed for thread-safe operations.
        
        Returns:
            A new DatabaseBackend instance with its own connection
        """
        pass
    
    def adapt_query(self, query: str) -> str:
        """
        Adapt a query for this backend's placeholder style.
        
        By default, assumes queries use '?' as placeholder.
        Backends that use different placeholders should override this.
        
        Args:
            query: Query with '?' placeholders
            
        Returns:
            Query with backend-appropriate placeholders
        """
        return query
    
    def set_busy_timeout(self, timeout_ms: int) -> None:
        """
        Set the busy timeout for the database connection.
        
        Args:
            timeout_ms: Timeout in milliseconds
        """
        # Default implementation does nothing
        # Backends that support this should override
        pass
    
    def optimize_for_bulk_insert(self) -> None:
        """
        Apply optimizations for bulk insert operations.
        
        This may include disabling constraints, changing
        transaction mode, etc.
        """
        # Default implementation does nothing
        pass
    
    def restore_after_bulk_insert(self) -> None:
        """
        Restore normal operation after bulk inserts.
        """
        # Default implementation does nothing
        pass


class QueryBuilder:
    """
    Helper class to build database-agnostic queries.
    
    This class provides methods to construct SQL queries that work
    across different database backends.
    """
    
    def __init__(self, backend: DatabaseBackend):
        self.backend = backend
        self.placeholder = backend.get_placeholder()
    
    def select(
        self,
        table: str,
        columns: List[str] = None,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Tuple[str, Tuple]:
        """
        Build a SELECT query.
        
        Args:
            table: Table name
            columns: Columns to select (default: *)
            where: WHERE conditions as dict
            order_by: ORDER BY clause
            limit: LIMIT value
            offset: OFFSET value
            
        Returns:
            Tuple of (query_string, parameters)
        """
        cols = ", ".join(columns) if columns else "*"
        query = f"SELECT {cols} FROM {table}"
        params = []
        
        if where:
            conditions = []
            for key, value in where.items():
                conditions.append(f"{key} = {self.placeholder}")
                params.append(value)
            query += " WHERE " + " AND ".join(conditions)
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        if limit is not None:
            query += f" LIMIT {self.placeholder}"
            params.append(limit)
        
        if offset is not None:
            query += f" OFFSET {self.placeholder}"
            params.append(offset)
        
        return query, tuple(params)
    
    def insert(
        self,
        table: str,
        data: Dict[str, Any],
    ) -> Tuple[str, Tuple]:
        """
        Build an INSERT query.
        
        Args:
            table: Table name
            data: Column-value pairs to insert
            
        Returns:
            Tuple of (query_string, parameters)
        """
        columns = list(data.keys())
        values = list(data.values())
        
        cols_str = ", ".join(columns)
        placeholders = ", ".join([self.placeholder] * len(columns))
        
        query = f"INSERT INTO {table} ({cols_str}) VALUES ({placeholders})"
        
        return query, tuple(values)
    
    def update(
        self,
        table: str,
        data: Dict[str, Any],
        where: Dict[str, Any],
    ) -> Tuple[str, Tuple]:
        """
        Build an UPDATE query.
        
        Args:
            table: Table name
            data: Column-value pairs to update
            where: WHERE conditions
            
        Returns:
            Tuple of (query_string, parameters)
        """
        set_parts = []
        params = []
        
        for key, value in data.items():
            set_parts.append(f"{key} = {self.placeholder}")
            params.append(value)
        
        where_parts = []
        for key, value in where.items():
            where_parts.append(f"{key} = {self.placeholder}")
            params.append(value)
        
        query = f"UPDATE {table} SET {', '.join(set_parts)} WHERE {' AND '.join(where_parts)}"
        
        return query, tuple(params)
    
    def delete(
        self,
        table: str,
        where: Dict[str, Any],
    ) -> Tuple[str, Tuple]:
        """
        Build a DELETE query.
        
        Args:
            table: Table name
            where: WHERE conditions
            
        Returns:
            Tuple of (query_string, parameters)
        """
        where_parts = []
        params = []
        
        for key, value in where.items():
            where_parts.append(f"{key} = {self.placeholder}")
            params.append(value)
        
        query = f"DELETE FROM {table} WHERE {' AND '.join(where_parts)}"
        
        return query, tuple(params)
