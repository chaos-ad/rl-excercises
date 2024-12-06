import logging
import logging.config
from pathlib import Path
from typing import Union, Optional

import yaml

##############################################################################

ROOT_PATH = Path(__file__).parents[1]
INFO_LOG_CONFIG_PATH = ROOT_PATH / "conf" / "logging" / "info.yaml"
DEBUG_LOG_CONFIG_PATH = ROOT_PATH / "conf" / "logging" / "debug.yaml"

##############################################################################

def setup(debug: bool = False) -> None:
    config_file = DEBUG_LOG_CONFIG_PATH if debug else INFO_LOG_CONFIG_PATH
    config = yaml.safe_load(Path(config_file).read_text())
    logging.config.dictConfig(config)

##############################################################################
