from typing import Union

from omegaconf import DictConfig
from hydra.utils import instantiate

from ..dataset_modules.stimulus_dataset import StimulusDataset
from ..dataset_modules.survey_dataset import SurveyDataset
from ..architecture_modules.lgbm_archimodule import LGBMClassifierModule
from ..architecture_modules.xgb_archimodule import XGBClassifierModule


class MLSetUp:
    def __init__(self, config: DictConfig,) -> None:
        self.config = config

    def get_dataset(self) -> Union[StimulusDataset, SurveyDataset]:
        dataset: Union[StimulusDataset, SurveyDataset] = instantiate(self.config.dataset_module)
        return dataset

    def get_architecture_module(self) -> Union[LGBMClassifierModule, XGBClassifierModule]:
        architecture_module: Union[LGBMClassifierModule, XGBClassifierModule] = instantiate(self.config.architecture_module)
        return architecture_module