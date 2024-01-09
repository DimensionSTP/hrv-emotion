import hydra

from src.pipeline.ml_pipeline import test


@hydra.main(config_path="configs/", config_name="ecg_basic_test.yaml")
def main(config):
    return test(config)


if __name__ == "__main__":
    main()