#!/usr/bin/env python3
from pathlib import Path
from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from shinka.core import EvolutionRunner


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    print("Experiment configurations:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    job_cfg = hydra.utils.instantiate(cfg.job_config)
    db_cfg = hydra.utils.instantiate(cfg.db_config)
    evo_cfg = hydra.utils.instantiate(cfg.evo_config)
    
    # Pass run identification info to database config for visualization
    # Use run_name from config (or generate from results_dir)
    run_id = cfg.get("run_name", None)
    if not run_id and hasattr(cfg, "output_dir"):
        # Extract run_id from output_dir path
        output_dir = str(cfg.output_dir)
        parts = output_dir.split("/")
        if len(parts) >= 2:
            run_id = parts[-1]  # Last part is usually the run identifier
    if run_id:
        db_cfg.run_id = run_id
    
    # Get task_name from exp_name in config
    task_name = cfg.get("exp_name", None)
    if task_name:
        db_cfg.task_name = task_name

    evo_runner = EvolutionRunner(
        evo_config=evo_cfg,
        job_config=job_cfg,
        db_config=db_cfg,
        verbose=cfg.verbose,
    )
    evo_runner.run()


if __name__ == "__main__":
    main()
