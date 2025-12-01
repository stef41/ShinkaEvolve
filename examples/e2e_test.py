#!/usr/bin/env python3
"""
End-to-end test with:
- gpt-oss-120b LLM via voltage-park (local model serving)
- Local sentence-transformers embeddings
- Dockerized PostgreSQL backend

This test uses NO external API keys - all models are locally hosted.
"""
import os
import sys

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set API key for voltage-park before importing shinka modules
os.environ["OPENAI_API_KEY"] = "sk-obelisk"

from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

# Voltage-park LiteLLM server endpoint
VOLTAGE_PARK_URL = "http://voltage-park-litellm.taila1eba.ts.net/v1"

# Use circle packing as a simple test task
os.chdir(os.path.join(os.path.dirname(__file__), "circle_packing"))

job_config = LocalJobConfig(eval_program_path="evaluate.py")

db_config = DatabaseConfig(
    # PostgreSQL backend (Docker)
    backend="postgres",
    pg_host="localhost",
    pg_port=5433,
    pg_database="shinka",
    pg_user="shinka",
    pg_password="shinka_dev",
    pg_use_pgvector=True,
    
    # Local embedding model (no API key needed)
    embedding_model="local:all-MiniLM-L6-v2",
    
    # Island configuration - more islands for longer run
    num_islands=4,
    archive_size=20,
    
    # Inspiration parameters
    elite_selection_ratio=0.3,
    num_archive_inspirations=3,
    num_top_k_inspirations=2,
    
    # Parent selection
    parent_selection_strategy="weighted",
    parent_selection_lambda=10.0,
)

task_sys_msg = """You are an expert mathematician solving circle packing problems.
Pack circles in a unit square to maximize the sum of radii.
Make sure all circles are disjoint and inside the unit square."""

evo_config = EvolutionConfig(
    task_sys_msg=task_sys_msg,
    patch_types=["diff", "full"],
    patch_type_probs=[0.7, 0.3],
    num_generations=20,  # Longer run for thorough testing
    max_parallel_jobs=2,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="python",
    
    # Use gpt-oss-120b via voltage-park (local model, no API key needed)
    llm_models=[f"openai/gpt-oss-120b@{VOLTAGE_PARK_URL}"],
    llm_kwargs=dict(
        temperatures=[0.7],
        max_tokens=4096,
    ),
    
    # Disable meta and novelty LLMs for this test
    meta_rec_interval=0,
    
    # Local embeddings
    embedding_model="local:all-MiniLM-L6-v2",
    
    init_program_path="initial.py",
    results_dir="results_e2e_test",
)


def main():
    print("=" * 60)
    print("End-to-End Test Configuration:")
    print("=" * 60)
    print(f"  LLM Model: {evo_config.llm_models}")
    print(f"  Embedding Model: {evo_config.embedding_model}")
    print(f"  Database Backend: {db_config.backend}")
    print(f"  PostgreSQL: {db_config.pg_host}:{db_config.pg_port}/{db_config.pg_database}")
    print(f"  Generations: {evo_config.num_generations}")
    print("=" * 60)
    print()
    
    evo_runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    evo_runner.run()
    
    print()
    print("=" * 60)
    print("End-to-End Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
