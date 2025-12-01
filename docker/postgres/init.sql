-- Initialize Shinka PostgreSQL database
-- This script runs automatically when the container is first created

-- Enable pgvector extension for embedding similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the programs table
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
);

-- Create the archive table
CREATE TABLE IF NOT EXISTS archive (
    program_id TEXT PRIMARY KEY REFERENCES programs(id)
);

-- Create the metadata_store table
CREATE TABLE IF NOT EXISTS metadata_store (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_programs_generation ON programs(generation);
CREATE INDEX IF NOT EXISTS idx_programs_island_idx ON programs(island_idx);
CREATE INDEX IF NOT EXISTS idx_programs_combined_score ON programs(combined_score);
CREATE INDEX IF NOT EXISTS idx_programs_parent_id ON programs(parent_id);
CREATE INDEX IF NOT EXISTS idx_programs_timestamp ON programs(timestamp);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO shinka;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO shinka;
