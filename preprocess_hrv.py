from omegaconf import DictConfig
import hydra

from src.pipeline.hrv_pipeline import preprocess_hrv


@hydra.main(config_path="configs/", config_name="preprocess_hrv.yaml")
def main(config: DictConfig,) -> None:
    return preprocess_hrv(config)


if __name__ == "__main__":
    main()