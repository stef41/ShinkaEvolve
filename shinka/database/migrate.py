#!/usr/bin/env python3
"""
Database Migration Utility for Shinka.

This utility helps migrate data between different database backends,
particularly from SQLite to PostgreSQL.

Usage:
    python -m shinka.database.migrate --source sqlite:path/to/db.sqlite --target postgres://user:pass@host/db
    
    # Or use environment variables
    export SHINKA_SOURCE_DB=sqlite:path/to/db.sqlite
    export SHINKA_TARGET_DB=postgres://user:pass@host/db
    python -m shinka.database.migrate
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .backends import create_backend, BackendType, POSTGRES_AVAILABLE
from .dbase import DatabaseConfig, Program

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_connection_string(conn_str: str) -> Dict[str, Any]:
    """
    Parse a connection string into backend configuration.
    
    Supported formats:
    - sqlite:path/to/db.sqlite
    - sqlite:///path/to/db.sqlite
    - postgres://user:pass@host:port/database
    - postgresql://user:pass@host:port/database
    
    Returns:
        Dictionary with backend type and configuration
    """
    if conn_str.startswith("sqlite:"):
        # Handle SQLite connection strings
        # sqlite:path/to/db -> relative path
        # sqlite:/absolute/path -> absolute path
        # sqlite:///absolute/path -> absolute path (standard URI format)
        path = conn_str.replace("sqlite:", "")
        if path.startswith("///"):
            # Standard URI format: sqlite:///absolute/path
            path = path[2:]  # Remove leading // to get /absolute/path
        elif not path:
            raise ValueError("SQLite connection string must include a path")
        return {
            "backend_type": "sqlite",
            "db_path": path,
        }
    
    elif conn_str.startswith(("postgres://", "postgresql://")):
        # Handle PostgreSQL connection strings
        parsed = urlparse(conn_str)
        return {
            "backend_type": "postgres",
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 5432,
            "database": parsed.path.lstrip("/") or "shinka",
            "user": parsed.username or "postgres",
            "password": parsed.password or "",
            "connection_string": conn_str,
        }
    
    else:
        # Try to infer from path
        if conn_str.endswith((".sqlite", ".db")):
            return {
                "backend_type": "sqlite",
                "db_path": conn_str,
            }
        raise ValueError(
            f"Unknown connection string format: {conn_str}. "
            f"Use sqlite:path or postgres://user:pass@host/db"
        )


def migrate_database(
    source_conn: str,
    target_conn: str,
    batch_size: int = 1000,
    dry_run: bool = False,
) -> None:
    """
    Migrate data from source database to target database.
    
    Args:
        source_conn: Source database connection string
        target_conn: Target database connection string
        batch_size: Number of records to migrate at once
        dry_run: If True, only show what would be done
    """
    logger.info("=" * 60)
    logger.info("Shinka Database Migration")
    logger.info("=" * 60)
    
    # Parse connection strings
    source_config = parse_connection_string(source_conn)
    target_config = parse_connection_string(target_conn)
    
    logger.info(f"Source: {source_config['backend_type']} database")
    logger.info(f"Target: {target_config['backend_type']} database")
    
    if target_config["backend_type"] == "postgres" and not POSTGRES_AVAILABLE:
        raise ImportError(
            "PostgreSQL migration requires psycopg2. "
            "Install it with: pip install psycopg2-binary"
        )
    
    if dry_run:
        logger.info("DRY RUN - No data will be modified")
    
    # Create source backend
    source_backend = create_backend(**source_config)
    source_backend.connect()
    logger.info("Connected to source database")
    
    # Create target backend
    target_backend = create_backend(**target_config)
    target_backend.connect()
    logger.info("Connected to target database")
    
    try:
        # Create tables in target
        if not dry_run:
            _create_target_tables(target_backend)
        
        # Migrate programs table
        _migrate_programs(
            source_backend, target_backend, batch_size, dry_run
        )
        
        # Migrate archive table
        _migrate_archive(
            source_backend, target_backend, batch_size, dry_run
        )
        
        # Migrate metadata_store table
        _migrate_metadata(
            source_backend, target_backend, dry_run
        )
        
        logger.info("=" * 60)
        logger.info("Migration completed successfully!")
        logger.info("=" * 60)
        
    finally:
        source_backend.close()
        target_backend.close()


def _create_target_tables(target_backend) -> None:
    """Create tables in the target database."""
    logger.info("Creating tables in target database...")
    
    backend_type = target_backend.get_backend_type()
    
    if backend_type == BackendType.POSTGRES:
        # PostgreSQL schema with JSONB
        programs_table = """
            CREATE TABLE IF NOT EXISTS programs (
                id TEXT PRIMARY KEY,
                code TEXT NOT NULL,
                language TEXT NOT NULL,
                parent_id TEXT,
                archive_inspiration_ids JSONB,
                top_k_inspiration_ids JSONB,
                generation INTEGER NOT NULL,
                timestamp DOUBLE PRECISION NOT NULL,
                code_diff TEXT,
                combined_score DOUBLE PRECISION,
                public_metrics JSONB,
                private_metrics JSONB,
                text_feedback TEXT,
                complexity DOUBLE PRECISION,
                embedding JSONB,
                embedding_pca_2d JSONB,
                embedding_pca_3d JSONB,
                embedding_cluster_id INTEGER,
                correct BOOLEAN DEFAULT FALSE,
                children_count INTEGER NOT NULL DEFAULT 0,
                metadata JSONB,
                migration_history JSONB,
                island_idx INTEGER
            )
        """
    else:
        # SQLite schema
        programs_table = """
            CREATE TABLE IF NOT EXISTS programs (
                id TEXT PRIMARY KEY,
                code TEXT NOT NULL,
                language TEXT NOT NULL,
                parent_id TEXT,
                archive_inspiration_ids TEXT,
                top_k_inspiration_ids TEXT,
                generation INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                code_diff TEXT,
                combined_score REAL,
                public_metrics TEXT,
                private_metrics TEXT,
                text_feedback TEXT,
                complexity REAL,
                embedding TEXT,
                embedding_pca_2d TEXT,
                embedding_pca_3d TEXT,
                embedding_cluster_id INTEGER,
                correct BOOLEAN DEFAULT 0,
                children_count INTEGER NOT NULL DEFAULT 0,
                metadata TEXT,
                migration_history TEXT,
                island_idx INTEGER
            )
        """
    
    target_backend.execute(programs_table)
    
    # Create indices
    indices = [
        "CREATE INDEX IF NOT EXISTS idx_programs_generation ON programs(generation)",
        "CREATE INDEX IF NOT EXISTS idx_programs_timestamp ON programs(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_programs_complexity ON programs(complexity)",
        "CREATE INDEX IF NOT EXISTS idx_programs_parent_id ON programs(parent_id)",
        "CREATE INDEX IF NOT EXISTS idx_programs_children_count ON programs(children_count)",
        "CREATE INDEX IF NOT EXISTS idx_programs_island_idx ON programs(island_idx)",
    ]
    for idx in indices:
        target_backend.execute(idx)
    
    # Create archive table
    target_backend.execute("""
        CREATE TABLE IF NOT EXISTS archive (
            program_id TEXT PRIMARY KEY,
            FOREIGN KEY (program_id) REFERENCES programs(id)
                ON DELETE CASCADE
        )
    """)
    
    # Create metadata_store table
    target_backend.execute("""
        CREATE TABLE IF NOT EXISTS metadata_store (
            key TEXT PRIMARY KEY, value TEXT
        )
    """)
    
    target_backend.commit()
    logger.info("Tables created successfully")


def _migrate_programs(
    source, target, batch_size: int, dry_run: bool
) -> None:
    """Migrate the programs table."""
    logger.info("Migrating programs table...")
    
    # Count total programs
    source.execute("SELECT COUNT(*) as count FROM programs")
    result = source.fetchone()
    total_count = result['count'] if result else 0
    logger.info(f"Total programs to migrate: {total_count}")
    
    if total_count == 0:
        logger.info("No programs to migrate")
        return
    
    if dry_run:
        logger.info(f"Would migrate {total_count} programs")
        return
    
    # Get target backend type for proper serialization
    target_type = target.get_backend_type()
    source_type = source.get_backend_type()
    
    # Enable bulk insert optimizations
    target.optimize_for_bulk_insert()
    
    # JSON fields that need special handling
    json_fields = [
        'archive_inspiration_ids', 'top_k_inspiration_ids',
        'public_metrics', 'private_metrics', 'embedding',
        'embedding_pca_2d', 'embedding_pca_3d', 'metadata',
        'migration_history'
    ]
    
    # Migrate in batches
    offset = 0
    migrated = 0
    start_time = time.time()
    
    while offset < total_count:
        source.execute(
            f"SELECT * FROM programs ORDER BY id LIMIT {batch_size} OFFSET {offset}"
        )
        rows = source.fetchall()
        
        if not rows:
            break
        
        for row in rows:
            # Convert row to dict
            row_dict = dict(row)
            
            # Handle JSON field conversion between backends
            for field in json_fields:
                value = row_dict.get(field)
                if value is not None:
                    # Deserialize from source format
                    if source_type == BackendType.SQLITE and isinstance(value, str):
                        try:
                            value = json.loads(value) if value else None
                        except json.JSONDecodeError:
                            value = None
                    
                    # Serialize for target format
                    if target_type == BackendType.SQLITE:
                        row_dict[field] = json.dumps(value) if value else None
                    else:
                        # PostgreSQL JSONB accepts dict/list directly
                        row_dict[field] = json.dumps(value) if value else None
            
            # Build insert query based on target type
            columns = list(row_dict.keys())
            placeholders = ", ".join([target.get_placeholder()] * len(columns))
            columns_str = ", ".join(columns)
            
            # Handle upsert differently for each backend
            if target_type == BackendType.POSTGRES:
                insert_query = f"""
                    INSERT INTO programs ({columns_str}) 
                    VALUES ({placeholders})
                    ON CONFLICT (id) DO NOTHING
                """
            else:
                insert_query = f"""
                    INSERT OR IGNORE INTO programs ({columns_str}) 
                    VALUES ({placeholders})
                """
            
            target.execute(insert_query, tuple(row_dict.values()))
            migrated += 1
        
        target.commit()
        offset += batch_size
        
        # Progress update
        elapsed = time.time() - start_time
        rate = migrated / elapsed if elapsed > 0 else 0
        logger.info(
            f"Progress: {migrated}/{total_count} programs "
            f"({migrated * 100 / total_count:.1f}%) - "
            f"{rate:.1f} records/sec"
        )
    
    # Restore normal operation
    target.restore_after_bulk_insert()
    
    elapsed = time.time() - start_time
    logger.info(
        f"Migrated {migrated} programs in {elapsed:.1f} seconds "
        f"({migrated / elapsed:.1f} records/sec)"
    )


def _table_exists(backend, table_name: str) -> bool:
    """Check if a table exists in the database."""
    backend_type = backend.get_backend_type()
    
    if backend_type == BackendType.POSTGRES:
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            )
        """
        backend.execute(query, (table_name,))
    else:  # SQLite
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        backend.execute(query, (table_name,))
    
    result = backend.fetchone()
    if result is None:
        return False
    
    # For SQLite, we get a row if table exists
    # For PostgreSQL, we get a boolean in 'exists' column
    if isinstance(result, dict):
        return result.get('exists', result.get('name') is not None)
    return bool(result)


def _migrate_archive(
    source, target, batch_size: int, dry_run: bool
) -> None:
    """Migrate the archive table."""
    logger.info("Migrating archive table...")
    
    # Check if archive table exists in source
    if not _table_exists(source, 'archive'):
        logger.info("Archive table does not exist in source, skipping...")
        return
    
    source.execute("SELECT COUNT(*) as count FROM archive")
    result = source.fetchone()
    total_count = result['count'] if result else 0
    logger.info(f"Total archive entries to migrate: {total_count}")
    
    if total_count == 0 or dry_run:
        if dry_run and total_count > 0:
            logger.info(f"Would migrate {total_count} archive entries")
        return
    
    target_type = target.get_backend_type()
    
    source.execute("SELECT program_id FROM archive")
    rows = source.fetchall()
    
    for row in rows:
        program_id = row['program_id']
        
        if target_type == BackendType.POSTGRES:
            query = """
                INSERT INTO archive (program_id) VALUES (%s)
                ON CONFLICT (program_id) DO NOTHING
            """
        else:
            query = """
                INSERT OR IGNORE INTO archive (program_id) VALUES (?)
            """
        
        target.execute(query, (program_id,))
    
    target.commit()
    logger.info(f"Migrated {len(rows)} archive entries")


def _migrate_metadata(source, target, dry_run: bool) -> None:
    """Migrate the metadata_store table."""
    logger.info("Migrating metadata_store table...")
    
    # Check if metadata_store table exists in source
    if not _table_exists(source, 'metadata_store'):
        logger.info("Metadata_store table does not exist in source, skipping...")
        return
    
    source.execute("SELECT key, value FROM metadata_store")
    rows = source.fetchall()
    
    if not rows:
        logger.info("No metadata to migrate")
        return
    
    if dry_run:
        logger.info(f"Would migrate {len(rows)} metadata entries")
        return
    
    target_type = target.get_backend_type()
    
    for row in rows:
        key = row['key']
        value = row['value']
        
        if target_type == BackendType.POSTGRES:
            query = """
                INSERT INTO metadata_store (key, value) VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """
        else:
            query = """
                INSERT OR REPLACE INTO metadata_store (key, value) VALUES (?, ?)
            """
        
        target.execute(query, (key, value))
    
    target.commit()
    logger.info(f"Migrated {len(rows)} metadata entries")


def main():
    """Main entry point for the migration utility."""
    parser = argparse.ArgumentParser(
        description="Migrate Shinka database between backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate from SQLite to PostgreSQL
  python -m shinka.database.migrate \\
    --source sqlite:evolution_db.sqlite \\
    --target postgres://user:pass@localhost/shinka

  # Dry run to see what would be migrated
  python -m shinka.database.migrate \\
    --source sqlite:evolution_db.sqlite \\
    --target postgres://localhost/shinka \\
    --dry-run

  # Migrate between SQLite databases
  python -m shinka.database.migrate \\
    --source sqlite:old_db.sqlite \\
    --target sqlite:new_db.sqlite
        """
    )
    
    parser.add_argument(
        "--source",
        default=os.environ.get("SHINKA_SOURCE_DB"),
        help="Source database connection string (or set SHINKA_SOURCE_DB)",
    )
    
    parser.add_argument(
        "--target",
        default=os.environ.get("SHINKA_TARGET_DB"),
        help="Target database connection string (or set SHINKA_TARGET_DB)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of records to migrate at once (default: 1000)",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.source:
        parser.error("--source is required (or set SHINKA_SOURCE_DB)")
    
    if not args.target:
        parser.error("--target is required (or set SHINKA_TARGET_DB)")
    
    try:
        migrate_database(
            args.source,
            args.target,
            args.batch_size,
            args.dry_run,
        )
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
