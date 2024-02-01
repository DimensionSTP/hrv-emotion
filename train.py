from omegaconf import DictConfig
import hydra

from src.pipelines.ml_pipeline import train


@hydra.main(config_path="configs/", config_name="lgbm_train.yaml")
def main(config: DictConfig,) -> None:
    return train(config)


if __name__ == "__main__":
    main()