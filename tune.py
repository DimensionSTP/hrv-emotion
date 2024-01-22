from omegaconf import DictConfig
import hydra

from src.pipeline.ml_pipeline import tune


@hydra.main(config_path="configs/", config_name="stimulus_lgbm_classifier_tune.yaml")
def main(config: DictConfig,) -> None:
    return tune(config)


if __name__ == "__main__":
    main()