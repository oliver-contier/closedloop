import json
import numpy as np
from pathlib import Path
from os.path import join as pjoin
from os.path import pardir, dirname, exists
import importlib
import sys


def _dictinfo_to_string(d):
    return "_".join([f"{k}-{v}" for k, v in d.items()])


def set_up_pycortex(
    cfg_path:Path=pjoin(dirname(__file__), pardir, pardir, "data", 'pycortex_options.cfg'),
    filestore_path:Path=pjoin(dirname(__file__), pardir, pardir, "data", 'pycortex_filestore'),
    ):
    """
    Set path to pycortex config and file store. Requires to have the following set up
    - pycortex_filestore should be in the `data` directory
    - pycortex_options.cfg should be in `data` as well
    - the pycortex_options.cfg should point to a valid inkscape AppImage or installation
    """
    for f in [cfg_path, filestore_path]:
        assert exists(f), f"path to relevant pycortex file does not exist {f}"
    if 'cortex' not in sys.modules:  # check if pycortex is imported already
        cortex = importlib.import_module('cortex')
    else:
        cortex = sys.modules['cortex']
    cortex.options.usercfg = cfg_path
    cortex.database.default_filestore = filestore_path
    return None


class NpEncoder(json.JSONEncoder):
    """For saving dict data returned by retinaface
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)