"""
Database backend abstraction for Shinka.

This module provides a unified interface for different database backends,
allowing the system to work with SQLite (default) or PostgreSQL.
"""

from .base import DatabaseBackend, BackendType
from .sqlite_backend import SQLiteBackend

# PostgreSQL backend is optional and requires psycopg2
try:
    from .postgres_backend import PostgreSQLBackend
    POSTGRES_AVAILABLE = True
except ImportError:
    PostgreSQLBackend = None
    POSTGRES_AVAILABLE = False


from typing import Union


def create_backend(
    backend_type: Union[str, BackendType] = "sqlite",
    **kwargs
) -> DatabaseBackend:
    """
    Factory function to create the appropriate database backend.
    
    Args:
        backend_type: Type of backend ('sqlite', 'postgres', or BackendType enum)
        **kwargs: Backend-specific configuration options
            For SQLite:
                - db_path: Path to SQLite database file
                - read_only: Whether to open in read-only mode
            For PostgreSQL:
                - host: Database host
                - port: Database port
                - database: Database name
                - user: Database user
                - password: Database password
                - connection_string: Full connection string (alternative to individual params)
    
    Returns:
        DatabaseBackend instance
    
    Raises:
        ValueError: If backend_type is not recognized
        ImportError: If PostgreSQL backend is requested but psycopg2 is not installed
    """
    # Normalize backend_type to string
    if isinstance(backend_type, BackendType):
        backend_type = backend_type.value
    
    if backend_type == "sqlite":
        return SQLiteBackend(**kwargs)
    elif backend_type == "postgres" or backend_type == "postgresql":
        if not POSTGRES_AVAILABLE:
            raise ImportError(
                "PostgreSQL backend requires psycopg2. "
                "Install it with: pip install psycopg2-binary"
            )
        return PostgreSQLBackend(**kwargs)
    else:
        raise ValueError(
            f"Unknown backend type: {backend_type}. "
            f"Supported backends: sqlite, postgres"
        )


__all__ = [
    "DatabaseBackend",
    "BackendType",
    "SQLiteBackend",
    "PostgreSQLBackend",
    "POSTGRES_AVAILABLE",
    "create_backend",
]
