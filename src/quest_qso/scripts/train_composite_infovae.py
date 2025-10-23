#!/usr/bin/env python
# coding: utf-8

import logging
import shutil
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import random_split

import quest_qso.plots.qa_plots as qa
from quest_qso import LOCAL_PATH as local_path
from quest_qso import mlconfig as cfg
from quest_qso.models.info_vae import InfoSpecVAE
from quest_qso.utils import load as ld
from quest_qso.utils import spec_dataset as sd
from quest_qso.utils import train_utils as tu
from quest_qso.utils import utilities

## ========================================================================= ##

# TODO: Investigate slow downs, possibly with profiling?
# Would need to do so on the server, Mac profiler is not available
#  at the time of writing (2025-01-21)

## ========================================================================= ##

logger = logging.getLogger("quest_qso.train")
parameters = cfg.MLConfig(cfg.get_parser().parse_args())

# base path for everything
logger.info(f"Using local_path: {local_path}")

# Generator and constants
# TODO: Check that I always get the same spectra
TGEN = torch.Generator().manual_seed(42)


## ========================================================================= ##
## ========================================================================= ##
## ========================================================================= ##


def save_train_results(save_dir, model, model_fname, timestamp, loss_histories, params):
    if params.dry_run:
        logger.info("Dry run, not saving anything.")
        return

    logger.info(f"Saving model and loss histories to {save_dir}.")
    torch.save(model.state_dict(), save_dir / model_fname)

    for loss, loss_type in zip(loss_histories, ["", "_val"]):
        loss.to_csv(save_dir / f"loss_history{loss_type}.csv", index=False)

    # Save model parameters
    current_params = params.to_json(save_dir / "params.json")

    cfg.update_master_json(
        f"{model.__class__.__name__}_cache_db.json", timestamp, current_params
    )

    logger.info("Saved model and parameters.")
    return


## ========================================================================= ##


def train_model_from_scratch(
    model,
    parameters,
    train_set,
    val_set,
    save_dir,
    model_fname,
    timestamp,
):
    logger.info("Ignoring existing models, training from scratch.")

    # Set up the optimizer
    logger.debug("Setting up the optimizer.")
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.learning_rate)

    # logger.info("Setting up gradient clipping.")
    # for p in model.parameters():
    #     p.register_hook(lambda grad: torch.clamp(grad, -0.05, 0.05))

    # Train the model
    # TODO: Make the mask part of parameters?
    logger.debug("Training the model.")
    loss_history, val_loss_history = tu.train_model(
        model,
        train_set,
        val_set,
        optimizer,
        parameters,
        flux_autofit=False,
        coverage_mask_flag=False,
        random_mask_flag=True,
        checkpoint_parent_dir=save_dir,
    )

    # Save the model
    save_train_results(
        save_dir,
        model,
        model_fname,
        timestamp,
        [loss_history, val_loss_history],
        parameters,
    )

    return model, loss_history, val_loss_history


## ========================================================================= ##
## ========================================================================= ##
## ========================================================================= ##


if __name__ == "__main__":
    # device to train on - use `force` to manually specify a device
    device = utilities.set_device(force=None)

    # setup folders and filenames
    # save_dir is where we save all the outputs, and it is contained in training_set_dir
    training_set_dir = root_dir = local_path / parameters.input_dataset_parent_dir

    # load the dataset and split in train and test
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

    # Splitting the dataset into training and validation using
    #  a pre-defined generator for reproducible results
    train_set, val_set = random_split(dataset, [0.9, 0.1], generator=TGEN)

    # Actually start the training or, load the model
    model, save_dir, model_fname, train_model, timestamp = ld.load_model(
        InfoSpecVAE,
        root_dir,
        dispersion,
        device,
        parameters=parameters,
        scaling_factor=scaling_factor,
        train_model=parameters.overwrite,
    )

    # Save the source code if requested, so that if I make changes I know
    #  which code I used for what
    if parameters.save_src:
        # Copy the source code to the save directory
        # This is useful for debugging and reproducibility
        logger.info("Copying source code to save directory.")
        src_dir = Path(__file__).parent.parent
        dst_dir = save_dir / "src"
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            src_dir,
            dst_dir,
            ignore=shutil.ignore_patterns(
                "data",
                "__pycache__",
                "deprecated",
                "personal",
                "examples",
            ),
            dirs_exist_ok=True,
        )
        logger.info(f"Copied source code to {dst_dir}.")

    if train_model:
        trained_model, loss_history, val_loss_history = train_model_from_scratch(
            model,
            parameters,
            train_set,
            val_set,
            save_dir,
            model_fname,
            timestamp,
        )
    else:
        trained_model = model
        loss_history = pd.read_csv(save_dir / "loss_history.csv")
        val_loss_history = pd.read_csv(save_dir / "loss_history_val.csv")

    ## ========================================================================= ##

    # QA plots
    if parameters.no_qa:
        logger.info("`--no-qa` requested, skipping QA plots.")
    else:
        fig_save_dir = save_dir / "figures"
        fig_save_dir.mkdir(parents=True, exist_ok=True)

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

    if parameters.debug:
        from IPython import embed

        embed()

# Example command to remind myself how to iterate
# Use -h to get help
# for reg_w in 0.01 0.005 0.001 0.0005 0.0001
# do
#     for lr in 0.01 0.001 0.0001 0.00001
#     do
#         /Users/francesco/uvp/quest_qso/src/quest_qso/scripts/train_composite_infovae.py ...args... -lr $lr -reg_w $reg_w
#     done
# done
