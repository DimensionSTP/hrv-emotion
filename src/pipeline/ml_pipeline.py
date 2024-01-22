from typing import Union

from omegaconf import DictConfig
from hydra.utils import instantiate

from ..utils.ml_setup import MLSetUp
from ..tuner_modules.lgbm_tunermodule import LGBMTunerModule
from ..tuner_modules.xgb_tunermodule import XGBTunerModule


def train(config: DictConfig,) -> None:
    ml_setup = MLSetUp(config)
    
    dataset = ml_setup.get_dataset()
    classifier = ml_setup.get_architecture_module()
    
    data, label = dataset()
    classifier.train(
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
    classifier = ml_setup.get_architecture_module()
    
    data, label = dataset()
    classifier.test(
        data=data,
        label=label,
        result_name=config.result_name,
    )

def tune(config: DictConfig,) -> None:
    ml_setup = MLSetUp(config)
    
    dataset = ml_setup.get_dataset()
    
    data, label = dataset()
    tuner_module: Union[LGBMTunerModule, XGBTunerModule] = instantiate(
        config.tuner_module, data=data, label=label
    )
    tuner_module()