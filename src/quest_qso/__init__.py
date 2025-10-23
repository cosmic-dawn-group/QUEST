# combination of several sources, most recent of which:
#  https://stackoverflow.com/a/68978292
#  https://stackoverflow.com/a/28743317

import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime as dt
from pathlib import Path

import torch
from cuda_selector import auto_cuda

# =========================================================================== #
# =========================== Convenience imports =========================== #
# =========================================================================== #


# =========================================================================== #
# ============= Minimum required parameters for model identifier ============ #
# ====================== and default dataset name =========================== #
# =========================================================================== #

EPOCHS = 500
REGULARIZATION_WEIGHT = 1e-1
LATENT_DIMENSIONS = 8
DEFAULT_DATASET_PARENT_DIR = "SDSS_DR16Q"
DEFAULT_DATASET = "SDSS_DR16Q_3650_9800_2300_2600_dv140_extended.npz"
DEFAULT_MAX_N_CPU = 8

# =========================================================================== #
# ================================= Logging ================================= #
# =========================================================================== #

MIN_LEVEL = logging.DEBUG  # min level of warning to be printed
FLT_LEVEL = logging.WARNING  # min level that goes to stderr

# =========================================================================== #


class LogFilter(logging.Filter):
    def __init__(self, level):
        self.level = level

    def filter(self, record):
        return record.levelno < self.level


# =========================================================================== #

CUR_LEVEL = logging.INFO  # level of the first child logger

# add this for year as well: `%Y-%m-%d - `
formatter = logging.Formatter(fmt="%(name)s - %(levelname)s: %(message)s")

# =========================================================================== #

# handlers, info and error
i_handler = logging.StreamHandler(sys.stdout)
e_handler = logging.StreamHandler(sys.stderr)

# file handler
home_dir = Path.home()
f_handler = logging.FileHandler(home_dir / "quest_qso.log")

# messages lower than WARNING go to stdout
# messages >= WARNING (and >= STDOUT_LOG_LEVEL) go to stderr
i_handler.addFilter(LogFilter(FLT_LEVEL))
e_handler.setLevel(max(MIN_LEVEL, FLT_LEVEL))
f_handler.setLevel(MIN_LEVEL)

# formatter
i_handler.setFormatter(formatter)
e_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)

# =========================================================================== #

# create root logger
root_logger = logging.getLogger()

# deals with IPython autoreload to avoid many many spam logging
if root_logger.hasHandlers():
    root_logger.handlers.clear()

root_logger.addHandler(i_handler)
root_logger.addHandler(e_handler)

# =========================================================================== #

# create child logger - to be used as 'main' logger
logger = logging.getLogger(__name__)
logger.setLevel(CUR_LEVEL)

# =========================================================================== #

# Logger to redirect warning to log, as I am essentially treating them as Info
#  at this point
logging.captureWarnings(True)  # this logs to py.warnings, which is created ad hoc

# =========================================================================== #


# In case I need to suppress things temporarily
# thanks to: https://gist.github.com/simon-weber/7853144
# If no logging this is the first offender to have a look at!
@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)


# =========================================================================== #
# Finally, uncomment this to deal with Astropy using Warning

# logging.captureWarnings(True)  # this logs to py.warnings, which is created ad hoc
# warnings_logger = logging.getLogger("py.warnings")
# warnings_logger.setLevel(logging.ERROR)

# =========================================================================== #
# ==================== Setup local path once and for all ==================== #
# =========================================================================== #

LOCAL_PATH = os.getenv("QUEST_LOCALPATH")

if LOCAL_PATH is None:
    LOCAL_PATH = Path.home() / ".QUEST_LOCAL_PATH"
    LOCAL_PATH.mkdir(parents=True, exist_ok=True)
else:
    LOCAL_PATH = Path(LOCAL_PATH)

logger.info(f"Using LOCAL_PATH = {LOCAL_PATH}.")

# =========================================================================== #
# ================= Add log to file if the user requests it ================= #
# =========================================================================== #

log_to_file = os.getenv("QUEST_LOG_TO_FILE", False)

if log_to_file in [True, "True", "true", "1"]:
    logger.info("Requested logging to file.")
    log_fld = LOCAL_PATH / "logs"
    log_fld.mkdir(parents=True, exist_ok=True)
    log_fname = log_fld / "quest_qso.log"

    logger.info(f"Logging to file {log_fname}.")
    if log_fname.exists():
        log_bak_timestamp = dt.fromtimestamp(log_fname.stat().st_mtime).strftime('%Y-%m-%dT%H-%M-%S')
        log_fname.rename(log_fld / f"quest_qso_{log_bak_timestamp}.log.bak")

    f_handler = logging.FileHandler(log_fname)
    f_handler.setLevel(MIN_LEVEL)
    f_handler.setFormatter(formatter)
    root_logger.addHandler(f_handler)

# =========================================================================== #
# ====================== Be nice to other users as well ===================== #
# =========================================================================== #

if os.getenv("AM_I_ON_SHARED_SERVER", False) in [True, "True", "true", "1"]:
    max_n_cpu = min(len(os.sched_getaffinity(0)) // 3, DEFAULT_MAX_N_CPU)
    torch.set_num_threads(max_n_cpu)
    logger.warning(
        f"Running on shared server, limiting the number of CPU cores to {max_n_cpu}."
    )

    # optionally also set a niceness level
    if os.environ.get("NICE_LVL"):
        nice_lvl = int(os.environ.get("NICE_LVL"))
        if nice_lvl <= 0:
            logger.warning(
                f"Nice level must be greater than zero, received {nice_lvl}."
            )
        else:
            os.nice(nice_lvl)
            logger.warning(f"Set the nice level to {nice_lvl}.")


# =========================================================================== #
# ========================== Be nice to myself too  ========================= #
# =========================================================================== #

# set device at init time, to avoid issues deriving from different parts of the
# code using different devices by accident

DEVICE = (
    auto_cuda()
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

logger.info(f"Using device: {DEVICE}.")

# =========================================================================== #
# ===================== Fix the seed for reproducibility ==================== #
# =========================================================================== #

# https://pytorch.org/docs/stable/notes/randomness.html
if os.getenv("TORCH_SEED", None):
    seed = int(os.environ.get("TORCH_SEED"))
    if seed > 0:
        logger.warning(f"Setting torch seed to {seed} for reproducibility.")
        torch.manual_seed(seed)
    else:
        logger.info("Not setting any torch seed.")
else:
    seed = 42
    logger.warning(f"Setting torch seed to {seed} for reproducibility.")
    torch.manual_seed(seed)

# =========================================================================== #
# =========================================================================== #
# =========================================================================== #

# Debug
if os.getenv("TORCH_DEBUG", False):
    logger.warning(
        "Setting torch.autograd.set_detect_anomaly(True). "
        "This will greatly slow down the code, enable this only if needed!."
    )
    torch.autograd.set_detect_anomaly(True)
