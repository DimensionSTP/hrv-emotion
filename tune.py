from omegaconf import DictConfig
import hydra

from src.pipelines.ml_pipeline import tune


@hydra.main(config_path="configs/", config_name="lgbm_tune.yaml")
def main(config: DictConfig,) -> None:
    return tune(config)


if __name__ == "__main__":
    main()