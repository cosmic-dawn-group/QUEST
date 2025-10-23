import logging

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import MultiplicativeLR, ReduceLROnPlateau, StepLR

from quest_qso.plots import qa_plots as qa
from quest_qso.utils import early_stopping as es
from quest_qso.utils import utilities

## ========================================================================= ##
## ========================================================================= ##
## ========================================================================= ##


logger = logging.getLogger(__name__)
DIV_STR = ( "+" + "-" * 12 + "+" + "-" * 23 + "+" + "-" * 21 + "+" + "-" * 21 + "+" + "-" * 17 + "+" + "-" * 17 + "+")  # fmt: skip

## ========================================================================= ##
## ========================================================================= ##
## ========================================================================= ##


def fancy_print_loss(epoch, params, train_loss_history, valid_loss_history):
    if epoch == 1:
        print("\n" + DIV_STR)
        print(
            f"| {'Epochs':^10} | {'Training Losses':^65} | {'Validation':^15} | {'Training':^15} |"
        )
        print(f"| {'':^10} | {'':^65} | {'':^15} | {'over':^15} |")
        print(
            f"| {'':^10} | {'Reconstruction (χ²)':^21} | {'KL':^19} | {'MMD':^19} | {'χ²':^15} | {'validation':^15} |"
        )
        print(DIV_STR)

    def format_number(num):
        if abs(num) <= 1e2 and abs(num) >= 1e-2 and num != 0:
            return f"{num:>.5f}"
        return f"{num:>.3e}"

    train_over_valid = train_loss_history[epoch - 1, 1] / valid_loss_history[epoch - 1]

    print(
        f"|{epoch:^11} "
        f"| {format_number(train_loss_history[epoch - 1, 1]):>21} "
        f"| {format_number(train_loss_history[epoch - 1, 2]):>19} "
        f"| {format_number(train_loss_history[epoch - 1, 3]):>19} "
        f"| {format_number(valid_loss_history[epoch - 1]):>15} "
        f"| {format_number(train_over_valid):>15} |"
    )

    if epoch == params.epochs:
        print(DIV_STR + "\n")


## ========================================================================= ##


def make_checkpoint_plot(
    model, loader_train, loader_valid, save_dir, epoch, n_components=10
):
    # set evaluation mode
    model.eval()

    with torch.no_grad():
        for loader in [loader_train, loader_valid]:
            gmm, _ = utilities.create_latent_space_gmm(
                model,
                loader,
                n_components=n_components,
            )

            suffix = "train" if loader == loader_train else "valid"
            qa.compare_sampled_input(
                model.dispersion,
                # what a beast of a line
                loader.dataset.dataset.input.data[:, 0].clone().detach().cpu().numpy(),
                model,
                gmm,
                ivar=None,  # loader.dataset.dataset.input.data[:, 4].clone().detach().cpu().numpy(),
                show=False,
                save=True,
                save_dir=save_dir,
                save_name=f"sampled_vs_input_{suffix}_e{epoch}.png",
            )

            if model.latent_dim <= 32:
                qa.plot_latent_space_corner(
                    model,
                    dataloader=loader,
                    save=True,
                    save_dir=save_dir,
                    save_name=f"latent_space_dims_corner_{suffix}_e{epoch}.png",
                    random_mask_flag=False,
                    coverage_mask_flag=False,
                )

    # back to training mode
    model.train()


## ========================================================================= ##
## ========================================================================= ##
## ========================================================================= ##


def _make_loader(dataset, batch_size, shuffle=True, num_workers=0, pin_memory=False):
    """Create a DataLoader for the dataset

    :param dataset:
    :param batch_size:
    :param shuffle:
    :param num_workers:
    :param pin_memory:
    :return:
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


## ========================================================================= ##


def _iterate_minibatches(
    model,
    optimizer,
    params,
    epoch,
    train_loader,
    train_loss_history,
    flux_autofit,
    coverage_mask_flag,
    random_mask_flag,
):
    # Set the model to training mode
    model.train()

    for x in train_loader:
        # Reset the gradients
        # This is the default anyway, just making it explicit
        optimizer.zero_grad(set_to_none=True)

        # Prepare input - No Scaling factor as we don't want weights
        #  to explode
        (
            stacked_input,
            _,
            _,
            _,
            flux,
            ivar,
            coverage_mask,
            _,
        ) = utilities.prepare_input(
            x,
            model.device,
            flux_autofit=flux_autofit,
            coverage_mask_flag=coverage_mask_flag,
            random_mask_flag=random_mask_flag,
        )

        # Forward pass
        predicted, _, z, mean, log_var = model(stacked_input)

        # Calculate the loss
        # In principle the inverse variance is enough, seems like that is not the case
        #  and I still need to multiply by the coverage mask. If I don't, the
        #  reconstruction looks much much worse. I think this is due to the fact that
        #  the ivar extents a bit further compared to the coverage mask
        # To solve this, I multiply the coverage mask into the ivar and call it a day
        #  note that the multiplication is done inside the function.
        loss = model.loss_function(
            flux,
            predicted,
            ivar,
            coverage_mask,
            mean,
            log_var,
            z,
            params,
            epoch,
        )

        # Add the loss to the loss history of this epoch
        for ldx, key in enumerate(loss.keys()):
            train_loss_history[epoch, ldx] += loss[key]

        # Backpropagation
        loss["loss"].backward()

        # Update the weights
        optimizer.step()

    # these are useful for normalising the losses
    return len(train_loader), train_loader.batch_size


## ========================================================================= ##


def _eval_loss_epoch(
    model,
    params,
    epoch,
    valid_loader,
    valid_loss_history,
    flux_autofit,
    random_mask_flag,
    coverage_mask_flag,
):
    model.eval()

    with torch.no_grad():
        for x in valid_loader:
            # Prepare input, no scaling in this case
            #  we do not want NN weights to explode
            (
                stacked_input,
                _,
                _,
                _,
                flux,
                ivar,
                coverage_mask,
                _,
            ) = utilities.prepare_input(
                x,
                model.device,
                flux_autofit=flux_autofit,
                random_mask_flag=random_mask_flag,
                coverage_mask_flag=coverage_mask_flag,
            )

            # Forward pass
            predicted, _, _, _, _ = model(stacked_input)

            # Validation loss, coverage mask is multiplied into the inverse variance
            # Compute this in the same way as the training loss so that the two values are
            #  comparable
            val_loss = model.loss_function(
                flux,
                predicted,
                ivar,
                coverage_mask,
                None,
                None,
                None,
                params,
                params.epochs + 1,
                validation=True,
            )

            # Save the validation loss history
            valid_loss_history[epoch] += val_loss["valid_loss"]

    return len(valid_loader), valid_loader.batch_size


## ========================================================================= ##


def train_model(
    model,
    train_set,
    test_set,
    optimizer,
    params,
    flux_autofit=False,
    coverage_mask_flag=False,
    random_mask_flag=True,
    checkpoint_parent_dir=None,
):
    """Train the model

    :param model:
    :param train_loader:
    :param test_loader:
    :param optimizer:
    :param epochs:
    :param batch_size:
    :return:
    """
    if params.early_stopping:
        logger.info("Requested early stopping, instantiating the early stopping class.")
        stopper = es.EarlyStoppingMinimize(
            patience=params.es_patience,
            rel_delta=params.es_rel_delta,
        )
        if params.es_upd_lr:
            es_factor = 0.5
            es_scheduler = MultiplicativeLR(optimizer, lambda x: es_factor)

    if params.make_checkpoint_plot and checkpoint_parent_dir is not None:
        checkpoint_save_dir = checkpoint_parent_dir / "checkpoints"
        checkpoint_save_dir.mkdir(parents=True, exist_ok=True)

    epochs = params.epochs

    # Initialize the loss history
    # This needs to be adjusted if we change the loss, but it is surely cleaner written this way
    #  the 4 elements are the total loss, the reconstruction loss, the KL divergence and the MMD loss
    #  Might be worth adding a fit loss, for the line loss
    train_loss_history = np.zeros((epochs, 4))
    valid_loss_history = np.zeros(epochs)

    # Make the data loaders.
    # Following this paper, we use a small but constant sized batch size for the test
    #  dataset: https://arxiv.org/abs/1804.07612
    # got lost in the references, leaving that one out there but I mean the Yann LeCun
    #  tweet
    valid_batch_size = params.valid_batch_size
    valid_loader = _make_loader(
        test_set,
        batch_size=valid_batch_size,
    )

    # Set the batch size, and to keep things clean, set it once
    train_batch_size = params.train_batch_size
    train_loader = _make_loader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
    )

    if params.update_batch_size and params.upd_lr:
        logger.info(
            "Instantiating scheduler to update the learning rate in addition to the batch size."
        )
        # only set the scheduler if we are going to update the batch size and
        #  we have requested to also update the batch size
        scheduler = StepLR(
            optimizer,
            step_size=params.update_batch_size_epoch,
            gamma=0.5,
        )

    if params.upd_lr_on_plateau:
        logger.info(
            "Instantiating scheduler to update the learning rate if the validation loss plateaus."
        )
        # only if requested
        plateau_factor = 0.5
        plateau_patience = 10  # default: 10
        scheduler_on_plateau = ReduceLROnPlateau(
            optimizer,
            "min",
            plateau_factor,
            patience=plateau_patience,
        )

    # Loop over the epochs, gracefully handling ctrl+c so that I can stop the training
    #  at will and still save the model / avoid crashes
    try:
        for epoch in range(epochs):
            # Variable batch size, simpler and quicker approach
            #  Increase following this paper: https://arxiv.org/abs/1711.00489
            if (
                params.update_batch_size
                and epoch > 0
                and epoch % params.update_batch_size_epoch == 0
                and train_batch_size < params.maximum_batch_size
            ):
                # TODO: Also update the learning rate?
                # TODO: Also check the annealing strategy? There should be torch
                #  implementations for this (scheduler something something)
                train_batch_size *= 2
                train_loader = _make_loader(
                    train_set, batch_size=train_batch_size, shuffle=True
                )
                logger.info(f"\nIncreasing `batch_size` to {train_batch_size}.")
                logger.info(f"Updated learning rate to: {scheduler.get_last_lr()}\n")

            # Iterate over the minibatches
            train_n_minibatches, epoch_bs_train = _iterate_minibatches(
                model,
                optimizer,
                params,
                epoch,
                train_loader,
                train_loss_history,
                flux_autofit,
                coverage_mask_flag,
                random_mask_flag,
            )

            # model is in evaluation mode, compute losses
            # pass the mask flags explicitly to avoid any confusion
            valid_n_minibatches, epoch_bs_valid = _eval_loss_epoch(
                model,
                params,
                epoch,
                valid_loader,
                valid_loss_history,
                flux_autofit,
                coverage_mask_flag,
                random_mask_flag,
            )

            # normalise the losses wrt batch size and latent space dimension
            # also note the discussion here: https://stackoverflow.com/a/62200123
            #  (and answer above this one) which is a bit different than
            #  the approach used here
            train_loss_history[epoch] /= train_n_minibatches

            # multiply the MMD by the batch size as well to recover correct normalisation
            # train_loss_history[epoch][3] *= epoch_bs_train
            valid_loss_history[epoch] /= valid_n_minibatches

            # Schedulers steps for learning rate update. Both need to be here
            #  according to the docs.
            # Note that there is a bit of a conflict here, as both
            #  schedules can be active at the same time.
            # The way it is set up for now is that until we have not reached the
            #  maximum batch size, then the scheduler for plateau is not active
            if params.early_stopping and stopper(valid_loss_history[epoch]):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                if params.es_upd_lr and es_scheduler.get_last_lr()[0] > 1e-6:
                    es_scheduler.step()
                    logger.info(
                        f"Reducing learning rate by a factor of {int(1 / es_factor)}. "
                        f"Current learning rate: {es_scheduler.get_last_lr()[0]:.2e}"
                    )
                    stopper.reset()
                else:
                    break

            if (
                params.update_batch_size
                and params.upd_lr
                and train_batch_size < params.maximum_batch_size
            ):
                scheduler.step()

            elif (
                params.upd_lr_on_plateau
                and (
                    (train_batch_size >= params.maximum_batch_size)
                    or (not params.update_batch_size)
                )
                and not params.early_stopping
                # if we are using early stopping, we apply a different scheduler
                #  to get around the patience of the ReduceLROnPlateau scheduler
            ):
                scheduler_on_plateau.step(valid_loss_history[epoch])

            # Print training progress
            fancy_print_loss(epoch + 1, params, train_loss_history, valid_loss_history)

            # make the sample vs input plot if requested
            if (
                params.make_checkpoint_plot
                and epoch % params.checkpoint_epoch == 0
                and epoch > 0
            ):
                make_checkpoint_plot(
                    model, train_loader, valid_loader, checkpoint_save_dir, epoch
                )

    except KeyboardInterrupt:
        print(DIV_STR[2:] + "\n")
        logger.info("Training stopped by user.")

    # final checkpoint plot if requested - no need to check that the folder exists
    #  we already make sure it does at the beginning
    if params.make_checkpoint_plot:
        make_checkpoint_plot(
            model, train_loader, valid_loader, checkpoint_save_dir, epoch
        )

    # Convert to pandas DataFrame
    train_loss_history = pd.DataFrame(
        train_loss_history,
        columns=["total_loss", "reconstruction_loss", "kld_loss", "mmd_loss"],
    )
    valid_loss_history = pd.DataFrame(valid_loss_history, columns=["val_loss"])

    return train_loss_history, valid_loss_history
