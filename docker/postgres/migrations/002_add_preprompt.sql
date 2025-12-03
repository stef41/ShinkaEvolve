-- Migration: Add preprompt column for tracking LLM generation preprompts
-- Run this on existing databases to add the new column

-- Add preprompt column to programs table
ALTER TABLE programs ADD COLUMN IF NOT EXISTS preprompt TEXT;

-- Add comment explaining the column
COMMENT ON COLUMN programs.preprompt IS 'The preprompt text injected before the system message during LLM code generation';
