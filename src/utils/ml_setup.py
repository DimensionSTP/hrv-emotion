from omegaconf import DictConfig
from hydra.utils import instantiate

from ..dataset_modules.stimulus_dataset import StimulusDataset
from ..dataset_modules.survey_dataset import SurveyDataset
from ..architecture_modules.basic_archimodule import BasicClassifierModule


class MLSetUp:
    def __init__(self, config: DictConfig):
        self.config = config

    def get_stimulus_dataset(self) -> StimulusDataset:
        stimulus_dataset: StimulusDataset = instantiate(self.config.dataset_module)
        return stimulus_dataset

    def get_survey_dataset(self) -> SurveyDataset:
        survey_dataset: SurveyDataset = instantiate(self.config.dataset_module)
        return survey_dataset

    def get_basic_classifier(self) -> BasicClassifierModule:
        basic_archimodule: BasicClassifierModule = instantiate(self.config.architecture_module)
        return basic_archimodule