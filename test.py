from omegaconf import DictConfig
import hydra

from src.pipelines.ml_pipeline import test


@hydra.main(config_path="configs/", config_name="lgbm_test.yaml")
def main(config: DictConfig,) -> None:
    return test(config)


if __name__ == "__main__":
    main()