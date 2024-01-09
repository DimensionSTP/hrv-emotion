import hydra

from src.pipeline.ml_pipeline import train


@hydra.main(config_path="configs/", config_name="basic_train.yaml")
def main(config):
    return train(config)


if __name__ == "__main__":
    main()