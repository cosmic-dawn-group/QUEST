# improve parsing and default arguments
import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

# =========================================================================== #
# ==================== Default values to use if not given =================== #
# =========================================================================== #
from quest_qso import (
    DEFAULT_DATASET,
    DEFAULT_DATASET_PARENT_DIR,
    EPOCHS,
    LATENT_DIMENSIONS,
    REGULARIZATION_WEIGHT,
)
from quest_qso.utils import resources

# Note: by default we use the same weight for MMD and KLD losses. They can be
#  adjusted separately though!

# =========================================================================== #
# =========================================================================== #
# =========================================================================== #

logger = logging.getLogger(__name__)
RESOURCE_PATH = resources.RESOURCE_PATH
RUNTIME_PARAMS = [
    "overwrite",
    "overwrite_cpu_defaults",
    "n_cpu",
    "niceness",
    "make_checkpoint_plot",
    "checkpoint_epoch",
    "dry_run",
    "no_qa",
    "tag",
    "json_full_path",
    "early_stopping",  # bit torn on this one
    "es_patience",
    "es_rel_delta",
    "debug",
    "save_src",
    "runtime_params",  # kind of funny
]

DEFAULT_RUNTIME_PARAMS = [
    False,
    False,
    8,
    None,
    False,
    None,
    False,
    False,
    "",
    None,
    False,
    None,
    None,
    False,
    False,
    RUNTIME_PARAMS,
]

# =========================================================================== #


def get_parser():
    # see this for tutorial: https://docs.python.org/3/howto/argparse.html#argparse-tutorial
    # for positional, just use the name without leading - or --
    # for optional, use --option (or short versions, as long as starts with -)
    # note: store_true defaults to false and vice versa!
    parser = argparse.ArgumentParser()

    # =========================================================================== #
    # ================================ Verbosity ================================ #
    # =========================================================================== #

    # TODO: change current logger to file
    # TODO: add second logger to stdout/stderr/whatever
    # TODO: change this so that verbosity changes the logger level of the stdout logger
    parser.add_argument(
        "-v",
        "--verbose",
        help="Make output more verbose. Currently unused! The logger is already set to high verbosity.",
        action="store_true",
    )

    # =========================================================================== #
    # ============================== General setup ============================== #
    # =========================================================================== #

    parser.add_argument(
        "-e",
        "--epochs",
        help=f"Set the number of epochs. Defaults to {EPOCHS}",
        default=EPOCHS,
        type=int,
    )

    parser.add_argument(
        "--latent-dim",
        help=f"Set the number of latent dimensions. Defaults to {LATENT_DIMENSIONS}",
        default=LATENT_DIMENSIONS,
        type=int,
    )

    parser.add_argument(
        "--loss-type",
        help="Set the loss type to use. Defaults to `chisq`. Valid options are `chisq` and `mse`.",
        default="chisq",
        type=str,
    )

    parser.add_argument(
        "--activation-function",
        help="Set the activation function to use. Defaults to `ReLu`. Valid options are `LeakyReLu` and `Speculator`. Case insensitive!",
        default="LeakyReLu",
        type=str,
    )

    parser.add_argument(
        "-o",
        "--overwrite",
        help="Force save the model and the output plots, even if it a valid saved model is available. "
        "A valid model is a model that has been trained with the same parameter combination.",
        action="store_true",
    )

    # =========================================================================== #
    # ============================== Masks related ============================== #
    # =========================================================================== #

    # =========================================================================== #
    # ========================== Batch sizes related ============================ #
    # =========================================================================== #

    parser.add_argument(
        "--train-bs",
        help="Batch size for training dataset. Defaults to 128.",
        default=128,
        type=int,
    )

    # =========================================================================== #

    parser.add_argument(
        "--valid-bs",
        help="Batch size for validation dataset. Defaults to 64.",
        default=64,
        type=int,
    )

    # =========================================================================== #

    parser.add_argument(
        "--upd-bs",
        help="Update the batch size during training. Defaults to False.",
        action="store_true",
    )

    # =========================================================================== #

    parser.add_argument(
        "--upd-bs-epoch",
        help="Epochs after which to update the batch size. Defaults to 250.",
        default=250,
        type=int,
    )

    # =========================================================================== #
    # ========================== Model hyperparameters ========================== #
    # =========================================================================== #

    def parse_hidden_dims(value):
        try:
            return [int(dim) for dim in value.split()]
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Hidden dimensions must be a space-separated list of integers, got {value}."
            )

    parser.add_argument(
        "--hidden-dims",
        help="Set the hidden dimensions (provided as a space-separated list of integers, such as `512 256 128`. "
        "Note that the decoder and the encoder will have the same structure, reversed.)",
        default="1024 512 256 128 64 32",
        type=parse_hidden_dims,
    )

    # =========================================================================== #

    parser.add_argument(
        "--reg-weight-rec",
        help="Set the regularization weight for the reconstruction loss. Defaults to 1.0.",
        default=1.0,
        type=float,
    )

    # =========================================================================== #

    parser.add_argument(
        "--reg-weight-mmd",
        help=f"Set the regularization weight for the MMD loss. Defaults to {REGULARIZATION_WEIGHT}.",
        default=REGULARIZATION_WEIGHT,
        type=float,
    )

    # =========================================================================== #

    parser.add_argument(
        "--reg-weight-kld",
        help=f"Set the regularization weight for the KLD loss. Defaults to {REGULARIZATION_WEIGHT}.",
        default=REGULARIZATION_WEIGHT,
        type=float,
    )

    # =========================================================================== #

    def check_rel_weight_mmd_kld(value):
        fvalue = float(value)
        if fvalue < 0.0 or fvalue > 1.0:
            raise argparse.ArgumentTypeError(
                f"The relative weight between KLD and MMD must be between 0.0 and 1.0, got {value}."
            )
        return fvalue

    parser.add_argument(
        "--rel-weight-mmd-kld",
        help="Set the regularization weight between MMD and KLD losses (where 1.0 is only MMD, 0.0 is only KLD). "
        "Defaults to 1.0 (MMD only).",
        default=1.0,
        type=check_rel_weight_mmd_kld,
    )

    # =========================================================================== #

    parser.add_argument(
        "--lr",
        help="Set the learning rate. Default to 0.001.",
        default=1e-3,
        type=float,
    )

    # =========================================================================== #
    # ============================ Line loss related ============================ #
    # =========================================================================== #

    parser.add_argument(
        "--train-on-lines",
        help="Global flag to indicate whether to run additional epochs where the "
        "reconstruction loss is only computed on emission lines.",
        action="store_true",
    )

    parser.add_argument(
        "--only-line-loss",
        help="Include only the line loss in the reconstruction loss. If `False`, "
        "the reconstruction loss also includes a contribution from the entire "
        "(valid) wavelength range.",
        action="store_true",
    )

    parser.add_argument(
        "--line-loss-weight",
        help="Set the line loss weight. Mostly useful if `--only-line-loss == False`.",
        default=-1.0,  # this is negative on purpose!
        type=float,
    )

    parser.add_argument(
        "--line-loss-epoch",
        help="Set the number of epochs for which one should train the model with "
        "added focus on emission lines in the reconstruction loss.",
        type=int,
        default=None,
    )

    # =========================================================================== #
    # ========================== Default dataset to use ========================= #
    # =========================================================================== #

    parser.add_argument(
        "--input-dataset-parent-dir",
        help="Parent directory where the input dataset is located. "
        f"Defaults to {DEFAULT_DATASET_PARENT_DIR}",
        default=DEFAULT_DATASET_PARENT_DIR,
        type=str,
    )

    parser.add_argument(
        "--input-dataset",
        help="Name of the input dataset from which to take training and validation data. "
        f"Defaults to {DEFAULT_DATASET}",
        default=DEFAULT_DATASET,
        type=str,
    )

    # =========================================================================== #
    # ========================== Schedulers and similar ========================= #
    # =========================================================================== #

    parser.add_argument(
        "--early-stopping",
        help="Stop training if the validation metric does not improve for a certain number of epochs.",
        action="store_true",
    )

    parser.add_argument(
        "--es-patience",
        help="Number of epochs to wait before stopping the training if the validation metric does not improve.",
        default=10,
        type=int,
    )

    parser.add_argument(
        "--es-rel-delta",
        help="Relative change in the validation metric to consider it an improvement.",
        default=0.01,
        type=float,
    )

    parser.add_argument(
        "--es-upd-lr",
        help="Update the learning rate when early stopping is triggered.",
        action="store_true",
    )

    # =========================================================================== #

    parser.add_argument(
        "--upd-lr",
        help="Update the learning rate when changing the batch size. "
        "The change is in the opposite direction compared to the change in batch size "
        "(i.e., if the batch size doubles, the learning rate is halved.).",
        action="store_true",
    )

    parser.add_argument(
        "--upd-lr-on-plateau",
        help="Update the learning rate if the validation metric plateaus. ",
        action="store_true",
    )

    parser.add_argument(
        "--do-warm-up",
        help="Warm up the model (i.e., allow a few epochs where only the reconstruction loss is computed). "
        "Regularisation loss is introduced gradually throught a sigmoid function.",
        action="store_true",
    )

    parser.add_argument(
        "--warm-up-epoch",
        help="Epoch at which the sigmoid function reaches a value of 0.5. ",
        type=int,
        default=None,
    )

    # =========================================================================== #
    # ========================== nCPU and niceness vals ========================= #
    # =========================================================================== #

    try:
        default_n_cpu = min(len(os.sched_getaffinity(0)) // 3, 8)
    except AttributeError:
        logger.warning(
            "OS does not support `os.sched_getaffinity`, using default value (8)."
        )
        default_n_cpu = 8

    parser.add_argument(
        "--n-cpu",
        help=f"Number of CPU cores to use. Defaults to {default_n_cpu}.",
        default=default_n_cpu,
        type=int,
    )

    parser.add_argument(
        "--niceness",
        help="Niceness value for the process.",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--overwrite-cpu-defaults",
        help="Use the command line arguments instead of the default set by __init__",
        action="store_true",
    )

    # =========================================================================== #
    # ========================= Scaling of input dataset ======================== #
    # =========================================================================== #

    parser.add_argument(
        "--scale-spectra",
        help="Scale the input spectra using the provided method.",
        action="store_true",
    )

    parser.add_argument(
        "--scale-method",
        help="Scaling method to use for scaling.",
        default="divide_by_median",
        type=str,
    )

    parser.add_argument(
        "--scale-by",
        help="Scaling factor to be used if method == `mult`. Note that spectra will be divided by this value.",
        default=None,
    )

    # =========================================================================== #
    # ================ Avoid clutter by not save things on disk ================= #
    # =========================================================================== #

    parser.add_argument(
        "-t",
        "--tag",
        help="Associate a name to this particular run for future identification.",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--dry-run",
        help="Make this a dry-run and avoid saving the results to disk, or record things in the database.",
        action="store_true",
    )

    parser.add_argument(
        "--no-qa",
        help="Do not produce QA plots.",
        action="store_true",
    )

    parser.add_argument(
        "--debug",
        help="Debug mode. Do not produce QA plots, do not save the model and open an IPython shell at the end of the training.",
        action="store_true",
    )

    parser.add_argument(
        "--make-checkpoint-plot",
        help="Compared sampled spectra against the validaton dataset.",
        action="store_true",
    )

    parser.add_argument(
        "--checkpoint-epoch",
        help="Epoch at which to make the checkpoint plot.",
        default=50,
        type=int,
    )

    parser.add_argument(
        "--save-src",
        help="Save the source code in the output directory.",
        action="store_true",
    )

    return parser


# =========================================================================== #
# ========================= Class to hold everything ======================== #
# ======= DO NOT CHANGE THESE VALUES MANUALLY, USE THE CLI EQUIVALENTS ====== #
# =========================================================================== #


class MLConfig:
    def __init__(self, args=None):
        if args is None:
            # still I need to set these to make things work
            self.runtime_params = RUNTIME_PARAMS
            return None

        self.epochs = args.epochs
        self.input_dataset_parent_dir = args.input_dataset_parent_dir
        self.input_dataset_name = args.input_dataset
        self.overwrite = args.overwrite
        self.loss_type = args.loss_type
        self.activation_function = args.activation_function.lower()

        self.latent_dim = args.latent_dim

        self.update_batch_size = args.upd_bs
        self.update_batch_size_epoch = args.upd_bs_epoch

        self.train_batch_size = args.train_bs
        self.valid_batch_size = args.valid_bs

        self.hidden_dims = args.hidden_dims
        self.reg_weight_rec = args.reg_weight_rec
        self.reg_weight_mmd = args.reg_weight_mmd
        self.reg_weight_kld = args.reg_weight_kld
        self.rel_weight_mmd_kld = args.rel_weight_mmd_kld
        self.learning_rate = args.lr

        self.train_on_lines = args.train_on_lines
        self.sum_line_loss = not args.only_line_loss
        self.line_loss_weight = args.line_loss_weight
        self.line_loss_epoch = args.line_loss_epoch

        # CPU and no-not-upset-others-on-same-machine parameters
        self.overwrite_cpu_defaults = args.overwrite_cpu_defaults
        self.n_cpu = args.n_cpu
        self.niceness = args.niceness

        # scheduler
        self.upd_lr = args.upd_lr
        self.upd_lr_on_plateau = args.upd_lr_on_plateau
        self.do_warm_up = args.do_warm_up
        self.warm_up_epoch = args.warm_up_epoch

        # early stopping
        self.early_stopping = args.early_stopping
        self.es_patience = args.es_patience
        self.es_rel_delta = args.es_rel_delta
        self.es_upd_lr = args.es_upd_lr

        # debug and ID parameters
        self.tag = args.tag
        self.make_checkpoint_plot = args.make_checkpoint_plot
        self.checkpoint_epoch = args.checkpoint_epoch
        self.debug = args.debug
        self.dry_run = args.dry_run
        self.no_qa = args.no_qa
        self.save_src = args.save_src

        # scaling of the spectra
        self.scale_spectra = args.scale_spectra
        self.scale_method = args.scale_method
        self.scale_by = args.scale_by

        # runtime parameters that I do not save
        self.runtime_params = RUNTIME_PARAMS

        # hard coded parameters for future
        #  extensions
        self.minimum_batch_size = 64
        self.maximum_batch_size = 1024
        self.default_batch_size = 128

        # check for consistency and so on
        self._validate()
        self._update_cpu_pars()
        self._make_consistent()

    # ======================================================================= #

    def _make_consistent(self):
        if not self.update_batch_size:
            self.update_batch_size_epoch = None

        if not self.train_on_lines:
            self.sum_line_loss = None
            self.line_loss_weight = None
            self.line_loss_epoch = None

        if not self.overwrite_cpu_defaults:
            self.n_cpu = None
            self.niceness = None

        if not self.scale_spectra:
            self.scale_method = None
            self.scale_by = None

        if not self.do_warm_up:
            self.warm_up_epoch = None

        if not self.make_checkpoint_plot:
            self.checkpoint_epoch = None

        if not self.early_stopping:
            logger.warning(
                "`--early-stopping` not set, setting `es_upd_lr`, `es_patience` and "
                "`es_rel_delta` to None."
            )
            self.es_upd_lr = False
            self.es_patience = None
            self.es_rel_delta = None

        if self.debug:
            logger.warning("Debug mode requested, setting `--no-qa` and `--dry-run`.")
            self.no_qa = True
            self.dry_run = True

    # ======================================================================= #

    def _update_cpu_pars(self):
        if self.overwrite_cpu_defaults:
            torch.set_num_threads(self.n_cpu)
            logger.warning(
                "Overwriting default CPU settings, "
                f"limiting the number of concurring threads to {self.n_cpu}."
            )

            # optionally also set a niceness level
            nice_lvl = self.niceness
            if nice_lvl and nice_lvl <= 0:
                logger.warning(
                    f"Nice level must be greater than zero, received {nice_lvl}."
                )
            elif nice_lvl:
                os.nice(nice_lvl)
                logger.warning(f"Niceness level set to {nice_lvl}")

    # ======================================================================= #

    def _validate(self):
        # Sanity checks and backward compatibility
        if not hasattr(self, "activation_function"):
            self.activation_function = "leakyrelu"

        if not hasattr(self, "scale_spectra"):
            self.scale_spectra = False
            self.scale_method = None
            self.scale_by = None

        if not hasattr(self, "do_warm_up"):
            self.do_warm_up = False
            self.warm_up_epoch = None

        # make sure batch sizes are within our defined limits
        if not (
            self.minimum_batch_size <= self.train_batch_size <= self.maximum_batch_size
        ):
            raise ValueError(
                f"`train_batch_size` must be between "
                f"minimum {self.minimum_batch_size} and maximum {self.maximum_batch_size} "
            )

        if not (
            self.minimum_batch_size <= self.valid_batch_size <= self.maximum_batch_size
        ):
            raise ValueError(
                f"`valid_batch_size` must be between "
                f"minimum {self.minimum_batch_size} and maximum {self.maximum_batch_size} "
            )

        if self.activation_function not in ["leakyrelu", "speculator"]:
            raise ValueError(
                f"`activation_function` must be either `ReLu` or `Speculator`, got {self.activation_function}."
            )

        if self.loss_type not in ["chisq", "mse"]:
            raise ValueError(
                f"`loss_type` must be either `chisq` or `mse`, got {self.loss_type}."
            )

        if self.sum_line_loss and self.line_loss_weight < 0.0 and self.line_loss_epoch:
            raise ValueError(
                "`line_loss_weight` must be greater than or equal to 0.0 if `sum_line_loss` is True"
            )

        if self.do_warm_up and (
            self.warm_up_epoch is None or self.warm_up_epoch > self.epochs
        ):
            raise ValueError(
                "Please provide a warm-up epoch through `--warm-up-epoch`, that is less than the total number of epochs."
            )

        # Further sanity checks, tbd
        # TODO: Check that some reconstruction loss is happening
        assert True, "Placeholder"

    # ======================================================================= #

    def _unset_full_path(self):
        self.json_full_path = None

    # ======================================================================= #

    def print_params(self):
        print(self)

    # ======================================================================= #

    def from_dict(self, data):
        for key, value in data.items():
            if not hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(
                    f"Parameter {key} already set to {getattr(self, key)}, found {data[key]} in the json file."
                )

        _default_runtime_params = DEFAULT_RUNTIME_PARAMS

        # prevent mismatch if I forget to update this once again
        if len(self.runtime_params) != len(_default_runtime_params):
            raise ValueError(
                "Mismatch between runtime parameters and default values. "
                "Maintainer, please update the default values!"
            )

        logger.info("Setting runtime parameters to default values:")
        for param, pval in zip(
            self.runtime_params,
            _default_runtime_params,
        ):
            if not hasattr(self, param):
                setattr(self, param, pval)
                logger.info(f"Runtime parameter {param} set to {pval}.")
            else:
                logger.warning(
                    f"Parameter {key} already set to {getattr(self, key)}, found "
                    f"{data[key]} in the json file. Not updating."
                )

        return self

    # ======================================================================= #

    def to_dict(self, keys_to_exclude=None):
        if keys_to_exclude is None:
            keys_to_exclude = []

        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in keys_to_exclude
        }

    # ======================================================================= #

    def to_json(self, path):
        # exclude attributes that are not parameters
        #  or that I do not want to save
        data = self.to_dict(self.runtime_params)
        if self.tag is not None:
            data["tag"] = self.tag

        # dump to json
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        return data

    # ======================================================================= #

    def from_json(self, path=None, data=None):
        assert not (path is None and data is None), (
            "Path must be provided if not a parameter dict is not given."
        )

        # load from json
        if data is None:
            with open(path, "r") as f:
                data = json.load(f)

        # update the parameters
        self.from_dict(data)
        self._validate()
        self._update_cpu_pars()
        self._make_consistent()

        # add path to the json file
        self.json_full_path = path
        return self

    # ======================================================================= #

    # TODO: make it so the ... and the string cutting are optional
    def __str__(self):
        adjustable_params = [
            "epochs",
            "input_dataset_parent_dir",
            "input_dataset_name",
            "latent_dim",
            "loss_type",
            "activation_function",
            "update_batch_size",
            "update_batch_size_epoch",
            "train_batch_size",
            "valid_batch_size",
            "hidden_dims",
            "reg_weight_rec",
            "reg_weight_mmd",
            "reg_weight_kld",
            "rel_weight_mmd_kld",
            "learning_rate",
            "train_on_lines",
            "sum_line_loss",
            "line_loss_weight",
            "line_loss_epoch",
        ]

        cpu_params = [
            "overwrite_cpu_defaults",
            "n_cpu",
            "niceness",
        ]

        fixed_params = [
            "minimum_batch_size",
            "maximum_batch_size",
            "default_batch_size",
        ]

        scaling_params = [
            "scale_spectra",
            "scale_method",
            "scale_by",
        ]

        scheduler_params = [
            "early_stopping",
            "es_patience",
            "es_rel_delta",
            "es_upd_lr",
            "upd_lr",
            "upd_lr_on_plateau",
            "do_warm_up",
            "warm_up_epoch",
        ]

        debug_params = [
            "make_checkpoint_plot",
            "checkpoint_epoch",
            "dry_run",
            "no_qa",
            "tag",
            "debug",
            "save_src",
        ]

        config_str = "\nCurrent set of parameters:\n"
        config_str += "See src/quest_qso/mlconfig.py or call the script with the --help flag for additional details.\n\n"
        config_str += "  Parameter               | Value\n"
        config_str += "  ------------------------+-------------------------------\n"
        for param_type in [
            adjustable_params,
            scaling_params,
            cpu_params,
            scheduler_params,
            fixed_params,
            debug_params,
        ]:
            for key in param_type:
                try:
                    value = getattr(self, key)
                except AttributeError:
                    value = "---"

                key_str = key
                value_str = (
                    (str(value)[:27] + "...") if len(str(value)) > 30 else str(value)
                )
                config_str += f"  {key_str:<23.23} | {value_str:<30.30}\n"
            config_str += "  ------------------------+-------------------------------\n"

        try:
            config_str += f"  Overwrite cached model: | {self.overwrite}\n"
        except AttributeError:
            config_str += "  Overwrite cached model: | ---\n"

        config_str += "  ------------------------+-------------------------------\n"
        return config_str

    # ======================================================================= #

    def __getitem__(self, key):
        return getattr(self, key)

    # ======================================================================= #

    def __setitem__(self, key, value):
        setattr(self, key, value)


# =========================================================================== #
# ========================= End input CLI arguments ========================= #
# ================== Read args and assign default values to ================= #
# ================= more explicit variables through MLConfig ================ #
# =========================================================================== #


# =========================================================================== #
# ============= Functions to deal with parameter loading/finding ============ #
# =========================================================================== #


def update_master_json(json_fname, timestamp, params):
    try:
        with open(RESOURCE_PATH / "data" / json_fname, "r") as f:
            data = json.load(f)

        data[timestamp] = params

        with open(RESOURCE_PATH / "data" / json_fname, "w") as f:
            json.dump(data, f, indent=4)

    except FileNotFoundError:
        logger.warning(f"File {json_fname} not found in the resources, creating it.")
        with open(RESOURCE_PATH / "data" / json_fname, "w") as f:
            json.dump({timestamp: params}, f, indent=4)
    except (KeyboardInterrupt, SystemExit) as cerr:
        logger.info("Ctrl+C detected, exiting.")
        raise cerr
    # this will catch all, exeption I want to catch should be handled above
    except BaseException as e:
        logger.error(f"Error while updating {json_fname}: {e}")
        pass  # possibly do something else here


# =========================================================================== #


def find_matching_params(params, db_fname):
    # TODO: sort all arrays at sametime otherwise we have issues - not the latent dims though!
    # TODO: possibly add a check and warning if the model I would get is
    #  too old (like a month or so, idk)
    out = []
    json_db = resources.load_json_resource(db_fname)
    # means we don't have a parameter database yet
    if json_db is None:
        return out, json_db

    # json_db is a dictionary, so I am iterating over the keys
    #  which correspond to a timestamp
    for timestamp in json_db:
        if params == json_db[timestamp]:
            out.append(timestamp)

    # as long as I write the dates in a sensible way, this works
    out.sort()
    return out, json_db


# =========================================================================== #


def find_most_recent_params(db_fname):
    # TODO: sort all arrays at sametime otherwise we have issues
    json_db = resources.load_json_resource(db_fname)
    timestamps = list(json_db.keys())
    timestamps.sort()

    return timestamps[-1], json_db


# =========================================================================== #


def json_db_to_df(db_fname=None, model_type_name=None):
    assert not (db_fname is None and model_type_name is None), (
        "Please provide either a database file name or a model type name."
    )
    prefix = RESOURCE_PATH / "data"

    if model_type_name is not None and db_fname is None:
        print("Building DB name from model type name.")
        db_fname = model_type_name + "_cache_db.json"
        print("Trying to load the database file: ", prefix / db_fname)

    try:
        df = pd.read_json(prefix / db_fname, orient="index")
        return df
    except FileNotFoundError:
        print("DB file not found, perhaps try with a different name?")
        print("Files available in prefix: ")
        for file in prefix.iterdir():
            if file.suffix == ".json":
                print("\t" + file.name)
        return None


# =========================================================================== #


def clean_json_db(db_fname, max_age=30, cleanup_disk=False, model_cache_dir=None):
    """
    Remove all entries older than `max_age` days from the database.
    """
    out, discard = {}, {}

    data = resources.load_json_resource(db_fname, verbose=True)
    if data is not None:
        # iterate over the timestamps, decide whether they are too old and if so throw them out
        for timestamp in data:
            if (
                datetime.now().date()
                - datetime.strptime(timestamp, "%Y%m%d_%H%M%Sp%f").date()
            ).days < max_age:
                out[timestamp] = data[timestamp]
            else:
                # still, write down what I removed to keep track of it
                discard[timestamp] = data[timestamp]
    else:
        print("Database file not found, nothing to clean up.")
        return None

    # write the trimmed database
    with open(RESOURCE_PATH / "data" / db_fname, "w") as f:
        json.dump(out, f, indent=4)

    # optionally also remove content from disk
    assert cleanup_disk and model_cache_dir is not None, (
        "Must provide a directory to cleanup."
    )
    print(f"Delete all models older than {max_age} day(s) from {model_cache_dir}? ")
    print(f"Found {len(discard.keys())} models to delete:")
    user_input = input("[y/N] ").lower()

    if len(discard) == 0:
        print(f"No model older than {max_age} days: nothing to do.")
        return None

    model_cache_dir = Path(model_cache_dir)
    if cleanup_disk and user_input == "y":
        for dir_ in model_cache_dir.iterdir():
            if dir_.is_dir() and dir_.name in discard:
                shutil.rmtree(dir_)

        # write the summary of what has been removed
        cleanup_summary = (
            model_cache_dir / f"Cleanup_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        )
        with open(cleanup_summary, "w") as f:
            json.dump(discard, f, indent=4)

        print(f"Cleaned up {len(discard)} models older than {max_age} days.")
        print(f"Summary written to {cleanup_summary}")

    return None
