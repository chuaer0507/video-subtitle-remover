import os
import shutil
import sys
from backend.config import config, BASE_DIR, VERSION
from backend.tools.common_tools import merge_big_file_if_not_exists
from backend.tools.constant import SubtitleDetectMode

_MODEL_NAME_MAP = {
    SubtitleDetectMode.PP_OCRv5_MOBILE: "PP-OCRv5_mobile_det",
    SubtitleDetectMode.PP_OCRv5_SERVER: "PP-OCRv5_server_det",
}

class ModelConfig:
    def __init__(self):
        self.LAMA_MODEL_DIR = self._writable_model_dir('big-lama', 'big-lama.pt')
        self.STTN_AUTO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sttn-auto', 'infer_model.pth')
        self.STTN_DET_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sttn-det', 'sttn.pth')
        self.PROPAINTER_MODEL_DIR = self._writable_model_dir('propainter', 'ProPainter.pth')
        if config.subtitleDetectMode.value == SubtitleDetectMode.PP_OCRv5_MOBILE:
            self.DET_MODEL_DIR = os.path.join(BASE_DIR,'models', 'V5', 'ch_det_fast')
        elif config.subtitleDetectMode.value == SubtitleDetectMode.PP_OCRv5_SERVER:
            self.DET_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'V5', 'ch_det')
        else:
            raise ValueError(f"Invalid subtitle detect mode: {config.subtitleDetectMode.value}")
        self.DET_MODEL_NAME = _MODEL_NAME_MAP[config.subtitleDetectMode.value]

        merge_big_file_if_not_exists(self.LAMA_MODEL_DIR, 'big-lama.pt')
        merge_big_file_if_not_exists(self.PROPAINTER_MODEL_DIR, 'ProPainter.pth')

    @staticmethod
    def _writable_model_dir(name, merged_file):
        source = os.path.join(BASE_DIR, 'models', name)
        if sys.platform != 'darwin' or not getattr(sys, 'frozen', False):
            return source
        target = os.path.expanduser(
            os.path.join('~/Library/Application Support/VideoSubtitleRemover',
                         'models', VERSION, name))
        if not os.path.isfile(os.path.join(target, merged_file)):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copytree(source, target, dirs_exist_ok=True)
        return target
