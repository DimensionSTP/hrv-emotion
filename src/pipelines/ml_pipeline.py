from typing import Union

from omegaconf import DictConfig
from hydra.utils import instantiate

from ..utils.ml_setup import MLSetUp
from ..tuners.lgbm_tuner import LGBMTuner
from ..tuners.xgb_tuner import XGBTuner


def train(config: DictConfig,) -> None:
    ml_setup = MLSetUp(config)
    
    dataset = ml_setup.get_dataset()
    architecture = ml_setup.get_architecture()
    
    data, label = dataset()
    architecture.train(
        data=data,
        label=label,
        num_folds=config.num_folds,
        seed=config.seed,
        params_path=config.params_path,
        result_name=config.result_name,
        plt_save_path=config.plt_save_path,
    )

def test(config: DictConfig,) -> None:
    ml_setup = MLSetUp(config)
    
    dataset = ml_setup.get_dataset()
    architecture = ml_setup.get_architecture()
    
    data, label = dataset()
    architecture.test(
        data=data,
        label=label,
        result_name=config.result_name,
    )

def tune(config: DictConfig,) -> None:
    ml_setup = MLSetUp(config)
    
    dataset = ml_setup.get_dataset()
    
    data, label = dataset()
    tuner: Union[LGBMTuner, XGBTuner] = instantiate(
        config.tuner, data=data, label=label
    )
    tuner()