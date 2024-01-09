from omegaconf import DictConfig
from hydra.utils import instantiate

from ..hrv_preprocess.file_control import FileController
from ..hrv_preprocess.normalization import Normalization
from ..hrv_preprocess.average import Average
from ..hrv_preprocess.augmentation import Augmentation
from ..hrv_preprocess.statistical_analysis import StatisticalAnalysis


class HRVSetUp:
    def __init__(self, config: DictConfig):
        self.config = config

    def get_file_controller(self) -> FileController:
        file_controller: FileController = instantiate(self.config.hrv_preprocess.file_controller)
        return file_controller

    def get_normalization(self) -> Normalization:
        normalization: Normalization = instantiate(self.config.hrv_preprocess.normalization)
        return normalization

    def get_average(self) -> Average:
        average: Average = instantiate(self.config.hrv_preprocess.average)
        return average

    def get_augmentation(self) -> Augmentation:
        augmentation: Augmentation = instantiate(self.config.hrv_preprocess.augmentation)
        return augmentation

    def get_statistical_analysis(self) -> StatisticalAnalysis:
        statistical_analysis: StatisticalAnalysis = instantiate(self.config.hrv_preprocess.statistical_analysis)
        return statistical_analysis