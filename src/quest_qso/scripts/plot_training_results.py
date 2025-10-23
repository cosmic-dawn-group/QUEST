#!/usr/bin/env python
import argparse
import logging
from datetime import datetime
from pathlib import Path

import enlighten
import pandas as pd
import torch
from torch.utils.data import random_split

import quest_qso.plots.qa_plots as qa
from quest_qso import LOCAL_PATH as local_path
from quest_qso import mlconfig as cfg
from quest_qso.models.info_vae import InfoSpecVAE
from quest_qso.utils import load as ld
from quest_qso.utils import spec_dataset as sd
from quest_qso.utils import utilities

# =========================================================================== #

logger = logging.getLogger("quest_qso.train")
TGEN = torch.Generator().manual_seed(42)

# =========================================================================== #
# =========================================================================== #
# =========================================================================== #


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-j",
        "--json-params-path",
        help="Path to the parameter file",
        default=None,
    )
    parser.add_argument(
        "-p",
        "--top-level-dir",
        help="Path to the top-level directory containing all timestamps",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--timestamp",
        help="Minimum timestamp to consider. Only used if a directory containing timestamps is provided through --top-level-dir",
        default=None,
    )
    return parser.parse_args()


# =========================================================================== #


def timestamp_greater_or_eq_min(timestamp, min_timestamp=None):
    if min_timestamp is None:
        return True

    return (
        datetime.strptime(timestamp, "%Y%m%d_%H%M%Sp%f").timestamp()
        >= datetime.strptime(min_timestamp, "%Y%m%d_%H%M%Sp%f").timestamp()
    )


# =========================================================================== #
# =========================================================================== #
# =========================================================================== #


if __name__ == "__main__":
    dataset = None
    prev_input_dataset_name = None
    prev_scale_spectra = None
    prev_scale_method = None
    prev_scale_by = None

    # mimics the train_composite_infovae.py script
    # if the user passes a json directly, ignore any directory or anything else
    if parse_args().json_params_path is not None:
        save_dirs = [Path(parse_args().json_params_path)]
    elif (
        parse_args().json_params_path is None and parse_args().top_level_dir is not None
    ):
        top_level_dir = Path(parse_args().top_level_dir)
        minimum_timestamp = parse_args().timestamp

        save_dirs = [
            top_level_dir / _.name / "InfoSpecVAE"
            for _ in top_level_dir.glob("*")
            if _.is_dir() and timestamp_greater_or_eq_min(_.name, minimum_timestamp)
        ]

        logger.info(f"Generating QA plots for {len(save_dirs)} models:")

    manager = enlighten.get_manager()
    pbar = manager.counter(
        total=len(save_dirs), desc="Processing folders: ", unit="ticks"
    )
    for save_dir in save_dirs:
        try:
            parameters = cfg.MLConfig().from_json(path=save_dir / "params.json")
        except FileNotFoundError:
            logger.error(f"Could not find the parameter file in {save_dir}")
            continue

        device = utilities.set_device(force=None)
        training_set_dir = root_dir = local_path / parameters.input_dataset_parent_dir

        # load the dataset and split in train and test, but only do so if necessary
        #  otherwise it is just a bunch of wasted CPU cycles
        if (
            dataset is None
            or parameters.input_dataset_name != prev_input_dataset_name
            or parameters.scale_spectra != prev_scale_spectra
            or parameters.scale_method != prev_scale_method
            or parameters.scale_by != prev_scale_by
        ):
            if dataset is None:
                logger.info("Dataset was None, loading it")
            elif parameters.input_dataset_name != prev_input_dataset_name:
                logger.info("Input dataset name changed, reloading dataset")
            elif parameters.scale_spectra != prev_scale_spectra:
                logger.info("Scale spectra changed, reloading dataset")
            elif parameters.scale_method != prev_scale_method:
                logger.info("Scale method changed, reloading dataset")
            elif parameters.scale_by != prev_scale_by:
                logger.info("Scale by changed, reloading dataset")

            dataset, dispersion, scaling_factor = sd.load_dataset(
                training_set_dir,
                parameters.input_dataset_name,
                subsample=1,
                replace_nan=True,
                replace_val=0,
                device=device,
                scale_spectra=parameters.scale_spectra,
                scale_method=parameters.scale_method,
                scale_by=parameters.scale_by,
                # scale_spectra -> False
                # scale_method  -> None
                #  If we need to make this explicit.
                # Here I do to make sure things are NOT being scaled!
            )

            prev_input_dataset_name = parameters.input_dataset_name
            prev_scale_spectra = parameters.scale_spectra
            prev_scale_method = parameters.scale_method
            prev_scale_by = parameters.scale_by

        # load the model, generate all the plots
        fig_save_dir = save_dir / "figures_upd"
        fig_save_dir.mkdir(parents=True, exist_ok=True)

        trained_model, _, _, _, _ = ld.load_model(
            InfoSpecVAE,
            root_dir,
            dispersion,
            device,
            parameters=parameters,
            scaling_factor=scaling_factor,
            trained_model_dir=save_dir,
            trained_model_fname="InfoSpecVAE.pt",
        )

        loss_history = pd.read_csv(save_dir / "loss_history.csv")
        val_loss_history = pd.read_csv(save_dir / "loss_history_val.csv")
        train_set, val_set = random_split(dataset, [0.9, 0.1], generator=TGEN)

        logger.info(f"Creating QA plots in {fig_save_dir}")
        qa.all_qa_plots(
            trained_model,
            parameters,
            loss_history,
            val_loss_history,
            train_set,
            val_set,
            True,  # save plots
            fig_save_dir,
            200,  # dpi
        )
        pbar.update()

# on the server
# for dd in $(echo */)
# do
#     ~/uvp/quest_qso/src/quest_qso/scripts/plot_training_results.py -j /hshome/bbe2364/data/quest_qso/SDSS_DR16Q/$dd/InfoSpecVAE
# done
