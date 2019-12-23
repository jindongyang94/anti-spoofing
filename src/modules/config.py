import os
import logging

# Logger --------------------------------------------------------------------------------------
try:
    import colorlog
    HAVE_COLORLOG = True
except ImportError:
    HAVE_COLORLOG = False


def create_logger():
    """
        Setup the logging environment
    """
    log = logging.getLogger()  # root logger
    log.setLevel(logging.INFO)
    format_str = '%(asctime)s - %(levelname)-8s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    if HAVE_COLORLOG and os.isatty(2):
        cformat = '%(log_color)s' + format_str
        colors = {'DEBUG': 'reset',
                  'INFO': 'reset',
                  'WARNING': 'bold_yellow',
                  'ERROR': 'bold_red',
                  'CRITICAL': 'bold_red'}
        formatter = colorlog.ColoredFormatter(cformat, date_format,
                                              log_colors=colors)
    else:
        formatter = logging.Formatter(format_str, date_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    return logging.getLogger(__name__)


logger = create_logger()

# S3 LOCATIONS -------------------------------------------------------------------------------------------
DATALAKE_NAME = 'hubble-datalake1'

PROFILEIMG_FOLDER = 'images/profile_pics'

# LOCAL WORKING DIRECTORIES --------------------------------------------------------------------------------
WORKING_DIR = '/Users/jindongyang/Documents/repos/hubble/hubble_projects/hubble_spoofing_detection'

EXTERNAL_DATA_DIR = os.path.join(WORKING_DIR, 'data/external')

INTERIM_DATA_DIR = os.path.join(WORKING_DIR, 'data/interim')

PROCESSED_DATA_DIR = os.path.join(WORKING_DIR, 'data/processed')

MODELS_DIR = os.path.join(WORKING_DIR, 'models')

NN_MODELS_DIR = os.path.join(MODELS_DIR, 'nn_models')

NN_WEIGHTS_DIR = os.path.join(MODELS_DIR, 'nn_pretrained_weights')

DETECTORS_DIR = os.path.join(MODELS_DIR, 'detectors')

LABELS_DIR = os.path.join(MODELS_DIR, 'labels')