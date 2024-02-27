import sys
if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

# init logger
import logging
from mmcv.utils import get_logger
logger = get_logger(name='flowDet', log_file=None, log_level=logging.INFO)

# new modules for mmdetection
from flowDet_mmdet.utils.gmmDet_modules import *
from flowDet_mmdet.roi_heads.feat_extract_bbox_heads import *
from flowDet_mmdet.roi_heads.msfeat_extract_roi_heads import *
from flowDet_mmdet.roi_heads.flowDet_bbox_heads import *
from flowDet_mmdet.roi_heads.flowDet_roi_heads import *
from flowDet_mmdet.detectors.frcnn_gtbbox import *
from flowDet_mmdet.losses.flowDet_losses import *