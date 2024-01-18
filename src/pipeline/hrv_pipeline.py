from omegaconf import DictConfig

from ..utils.hrv_setup import HRVSetUp


def preprocess_hrv(config: DictConfig,) -> None:
    hrv_setup = HRVSetUp(config)

    file_controller = hrv_setup.get_file_controller()
    file_controller.move_files_per_emotions()

    normalization = hrv_setup.get_normalization()
    normalization()
    
    average = hrv_setup.get_average()
    average()
    augmentation = hrv_setup.get_augmentation()
    augmentation()
    file_controller.make_test_template()

    statistical_analysis = hrv_setup.get_statistical_analysis()
    statistical_analysis()