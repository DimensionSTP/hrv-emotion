from omegaconf import DictConfig
import hydra

from src.pipelines.signal_pipeline import get_hrv


@hydra.main(config_path="configs/", config_name="signal_to_hrv.yaml")
def main(config: DictConfig,) -> None:
    return get_hrv(config)


if __name__ == "__main__":
    main()