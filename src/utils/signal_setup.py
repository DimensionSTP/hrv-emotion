from omegaconf import DictConfig
from hydra.utils import instantiate

from ..signal_preprocess.maxima_peak_detection import MaximaPeakDetection
from ..signal_preprocess.biopac_data_slicer import BioPacDataSlicer


class SignalSetUp:
    def __init__(self, config: DictConfig,) -> None:
        self.config = config

    def get_maxima_peak_detection(self) -> MaximaPeakDetection:
        maxima_peak_detection: MaximaPeakDetection = instantiate(self.config.signal_preprocess.maxima_peak_detection)
        return maxima_peak_detection

    def get_biopac_data_slicer(self) -> BioPacDataSlicer:
        biopac_data_slicer: BioPacDataSlicer = instantiate(self.config.signal_preprocess.biopac_data_slicer)
        return biopac_data_slicer