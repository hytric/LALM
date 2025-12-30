from src.train import main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
import hydra

@hydra.main(version_base=None, config_path="config", config_name="config")
def run_with_hydra(cfg: DictConfig):
    """
    Main entry point with Hydra.
    
    Usage:
        python main.py
        python main.py dataset=speechcraft_train
        python main.py experiment.name=my_experiment trainer.trainer.resume_from_checkpoint=path/to/ckpt
        python main.py model=base dataset=librispeech_train
    """
    main(cfg)

if __name__ == "__main__":
    run_with_hydra()