from omegaconf import DictConfig

from ..utils.ml_setup import MLSetUp


def train(config: DictConfig,) -> None:
    ml_setup = MLSetUp(config)
    
    stimulus_dataset = ml_setup.get_stimulus_dataset()
    basic_classifier = ml_setup.get_basic_classifier()
    
    data, label = stimulus_dataset()
    basic_classifier.train(
        data=data,
        label=label,
        num_folds=config.num_folds,
        fold_seed=config.fold_seed,
        boosting_type=config.boosting_type,
        objective=config.objective,
        metric=config.metric,
        result_name=config.result_name,
        plt_save_path=config.plt_save_path,
    )

def test(config: DictConfig,) -> None:
    ml_setup = MLSetUp(config)
    
    stimulus_dataset = ml_setup.get_stimulus_dataset()
    basic_classifier = ml_setup.get_basic_classifier()
    
    data, label = stimulus_dataset()
    basic_classifier.test(
        data=data,
        label=label,
        result_name=config.result_name,
    )