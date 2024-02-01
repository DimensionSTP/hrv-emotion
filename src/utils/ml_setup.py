from typing import Union

from omegaconf import DictConfig
from hydra.utils import instantiate

from ..datasets.stimulus_dataset import StimulusDataset
from ..datasets.survey_dataset import SurveyDataset
from ..architectures.lgbm_architecture import LGBMArchitecture
from ..architectures.xgb_architecture import XGBArchitecture


class MLSetUp:
    def __init__(self, config: DictConfig):
        self.config = config

    def get_dataset(self) -> Union[StimulusDataset, SurveyDataset]:
        dataset: Union[StimulusDataset, SurveyDataset] = instantiate(self.config.dataset)
        return dataset

    def get_architecture(self) -> Union[LGBMArchitecture, XGBArchitecture]:
        architecture: Union[LGBMArchitecture, XGBArchitecture] = instantiate(self.config.architecture)
        return architecture