-- Migration: Add raw metrics columns for dynamic objective function support
-- Run this on existing databases to add the new columns

-- Add raw metrics columns (used for on-the-fly score recalculation)
ALTER TABLE programs ADD COLUMN IF NOT EXISTS raw_ppl_score DOUBLE PRECISION;
ALTER TABLE programs ADD COLUMN IF NOT EXISTS raw_code_size INTEGER;
ALTER TABLE programs ADD COLUMN IF NOT EXISTS raw_exec_time DOUBLE PRECISION;

-- Add objective function tracking column
ALTER TABLE programs ADD COLUMN IF NOT EXISTS objective_function_used TEXT;

-- Add comment explaining the columns
COMMENT ON COLUMN programs.raw_ppl_score IS 'Primary performance score (e.g., game_score, accuracy) for dynamic objective function';
COMMENT ON COLUMN programs.raw_code_size IS 'Code size in bytes for dynamic objective function';
COMMENT ON COLUMN programs.raw_exec_time IS 'Execution time in seconds for dynamic objective function';
COMMENT ON COLUMN programs.objective_function_used IS 'The objective function formula used when this program was scored (e.g., "ppl_score - 0.01 * code_size")';
