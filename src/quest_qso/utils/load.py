# Moved to a new module as things were getting way too complicated
import datetime
import json
import logging
from pathlib import Path

# misc utilities
import torch

from quest_qso import mlconfig as cfg
from quest_qso.utils import resources, utilities

## ========================================================================= ##

# Logging

logger = logging.getLogger(__name__)


## ========================================================================= ##


def _valid(timestamp):
    try:
        datetime.datetime.strptime(timestamp, "%Y%m%d_%H%M%Sp%f")
        return True
    except ValueError:
        return False


## ========================================================================= ##


def _find_model_in_training_dir(trained_model_dir):
    logger.info(f"Received a trained model directory: {trained_model_dir}.")

    # no name provided - try to get model from a glob
    trained_model_dir = Path(trained_model_dir)
    available_models = list(trained_model_dir.glob("*.pt"))
    if len(available_models) != 1:
        raise ValueError(
            "Multiple or no models found in the provided directory! "
            "Please provide the model name using the `trained_model_fname` keyword parameter."
        )
    elif len(available_models) == 1:
        trained_model_fname = available_models[0]
        logger.info(f"Found model: {trained_model_fname}.")
        return trained_model_fname


## ========================================================================= ##


def _get_model_from_param_path(parameters):
    parameters_full_path = Path(parameters.json_full_path)
    trained_model_dir = parameters_full_path.parent
    trained_model_fname = list(trained_model_dir.glob("*.pt"))
    if len(trained_model_fname) != 1:
        raise ValueError(
            "Multiple or no models found in the parameter file directory! "
            "Please provide the model name using the `trained_model_fname` keyword parameter."
        )
    logger.info(f"Found model: {trained_model_fname[0]}.")
    return trained_model_dir, trained_model_fname[0], parameters


## ========================================================================= ##


def _validate_user_provided_dir_fname(
    trained_model_dir,
    trained_model_fname,
    parameters,
):
    # convert to path if needed
    try:
        trained_model_dir = Path(trained_model_dir)
    except TypeError:
        if (
            hasattr(parameters, "json_full_path")
            and parameters.json_full_path is not None
            and isinstance(parameters.json_full_path, (str, Path))
        ):
            logger.info(
                "No trained model directory provided but loading parameters from a json file. "
                "Trying to find a model in the same folder."
            )
            return _get_model_from_param_path(parameters)
        else:
            # I assume this is because trained_model_dir is None
            return None, None, parameters

    # if we only get the name of the model, but not the directory, we cannot
    #  load the model as we don't know where to look for it, so we
    #  pretend we did not receive any model
    if trained_model_dir is None and trained_model_fname is not None:
        trained_model_fname = None
        logger.warning(
            "Received a trained model file, but no directory. "
            "Ignoring the trained model filename."
        )

    # If the user provides the folder to load the model and not the model
    #  we first try to look for a pt model, and if we don't find it we raise an error
    if trained_model_dir is not None and trained_model_fname is None:
        trained_model_dir = _find_model_in_training_dir(trained_model_dir)

    # if the user initialised the parameter by loading it from file and we
    #  don't have a folder yet, we try to get the folder from the parameter file
    #  and the model name assuming it is a .pt in the same folder as param
    # This errors out if multiple models are found, don't want to deal with that

    # if we got till here means we have a valid model in the user-provided folder
    #  so we also try to load the parameter file from there an ignore anything else
    try:
        logger.info(
            f"Trying to load parameters from {trained_model_dir / 'params.json'}."
        )
        parameters = cfg.MLConfig().from_json(trained_model_dir / "params.json")
        parameters._unset_full_path()
    except FileNotFoundError:
        logger.warning("No parameter file found in the trained model directory.")
        # just return whatever was originally passed as parameters

    return trained_model_dir, trained_model_fname, parameters


## ========================================================================= ##


def _parameter_is_none(model_type_name):
    logging.warning("No parameters provided, trying to load a cached model.")
    # we are using the model and we have no parameters set
    #  we try to load, in order:
    #  - the most ecent run
    #  - the run with default parameters
    #  - if both fail, we raise an error

    # train_model defaults to false
    train_model = False

    try:
        logger.info("Trying to load the most recently trained model.")
        timestamp, parameter_db = cfg.find_most_recent_params(
            f"{model_type_name}_cache_db.json"
        )
        # find the parameter file and read it
        parameters = cfg.MLConfig().from_dict(parameter_db[timestamp])

    except json.decoder.JSONDecodeError:
        # the previous implementation made no sense, if I can't find any recent model
        #  than there is no point in trying with the default parameters
        # Either we train a new one, or we just throw an error!
        logger.warning("Could not load any model!")
        parameters = resources.load_json_resource("default_params.json")
        parameters = cfg.MLConfig().from_dict(parameters)

        user_input = input("Train a new model? [Y/n] ")
        if user_input.lower() in ["y", "\n", ""]:
            timestamp = utilities.get_timestamp()
            train_model = True
        else:
            raise ValueError("No model found in the cache, and no training requested.")

    return timestamp, train_model, parameters


## ========================================================================= ##


def _parameter_is_not_none_model_fname_is_none(model_type_name, parameters):
    # train_model defaults to false
    train_model = False

    logger.info("Trying to find a model from cache matching this set of parameters.")
    # try to find a valid model with the current set of parameters
    #  within the local_data folder
    matching_timestamps, _ = cfg.find_matching_params(
        parameters.to_dict(parameters.runtime_params),
        f"{model_type_name}_cache_db.json",
    )

    if len(matching_timestamps) > 1:
        logger.warning("Multiple match found, using the most recent one.")
        timestamp = matching_timestamps[-1]
    elif len(matching_timestamps) == 1:
        timestamp = matching_timestamps[0]
    else:
        print("Model not found for current combination of parameters.")
        user_input = input("Train a new one? [Y/n] ")
        if user_input.lower() in ["y", "\n", ""]:
            timestamp = utilities.get_timestamp()
            train_model = True
        else:
            raise ValueError("No matching model found for the given parameters.")

    # no need to return the parameters as we already have them from the start
    return timestamp, train_model


## ========================================================================= ##


def _parameter_is_not_none_model_fname_is_not_none(
    trained_model_dir,
    trained_model_fname,
):
    logger.info(f"Loading model {trained_model_fname} from {trained_model_dir}.")
    # try to get timestamp from folder structure, assuming standard folder structure
    timestamp = trained_model_dir.parent.name
    logger.info(f"Inferred timestamp from folder structure: {timestamp}")
    if not _valid(timestamp):
        timestamp = utilities.get_timestamp()
        logger.info(f"Invalid timestamp, using timestamp: {timestamp}")

    return timestamp, False  # train_model


## ========================================================================= ##
## ========================================================================= ##
## ========================================================================= ##


def load_model(
    model_type,
    root_dir,
    dispersion,
    device,
    parameters=None,
    scaling_factor=None,
    train_model=False,
    user_timestamp=None,
    trained_model_dir=None,
    trained_model_fname=None,
):
    """
    Load a (possibly pre-trained) VAE model.

    :param root_dir: Directory where the trained model is stored.
    :type dur: pathlib.Path

    :param dispersion: Dispersion axis for all spectra.
    :type dataset: Dataset

    :return: Torch model and batch size.
    :rtype: tuple
    """
    # current (20250122) solution for this is:
    # - models are identified by the name of the model class and a timestamp
    # - i have a file with the association full model name -> parameters
    # - in this function:
    #   - if I pass the parameters, means I am training the model or I want
    #     to explicitely set the parameters. In this case I use the parameters
    #     and simply ignore everything that was already saved in the text file
    #     but I still try to check whether there is already a model with that
    #     particular combination of parameters. If so, I load it
    #   - if I don't pass the parameters, I first try to get the parameters from
    #     a default parameter file and load the corresponding model. If everything
    #     fails then I try to load the most recently trained model.
    trained_model_dir, trained_model_fname, parameters = (
        _validate_user_provided_dir_fname(
            trained_model_dir,
            trained_model_fname,
            parameters,
        )
    )

    # model class name we are using so that we can possibly use this to load
    #  different models
    model_type_name = model_type.__name__

    # get parameters if nothing is provided
    if parameters is None:
        timestamp, train_model, parameters = _parameter_is_none(model_type_name)

    # if the user provides a timestamp, use that to try loading the model
    elif user_timestamp is not None:
        raise NotImplementedError(
            "User timestamp is yet to be implemented. "
            "Please provide the model fname as a temporary solution."
        )
        # TODO: implement user_timestamp at some point

    # we pass the parameters to load_model(), but want to overwrite any saved model
    #  and just train a new one
    elif parameters is not None and train_model:
        timestamp = utilities.get_timestamp()

    # we pass the parameters, but we do not want to overwrite any saved model
    # Instead, we'd like to check whether there is a model already trained with
    #  this set of parameters we can load
    elif parameters is not None and not train_model:
        # we have no info about folder or model, we try to load a cached model
        if trained_model_fname is None:
            timestamp, train_model = _parameter_is_not_none_model_fname_is_none(
                model_type_name, parameters
            )
        # we know the folder and the model by now
        elif trained_model_fname is not None:
            timestamp, train_model = _parameter_is_not_none_model_fname_is_not_none(
                trained_model_dir, trained_model_fname
            )

    # At this point I should ALWAYS have the parameters
    #  so show them before doing anything else
    parameters.print_params()

    # here we instantiate the untrained model
    model = model_type(
        input_dim=dispersion.shape[0],
        latent_dim=parameters.latent_dim,
        activation_function=parameters.activation_function,
        device=device,
        dispersion=dispersion,
        reg_weight_rec=parameters.reg_weight_rec,
        reg_weight_mmd=parameters.reg_weight_mmd,
        reg_weight_kld=parameters.reg_weight_kld,
        rel_weight_mmd_kld=parameters.rel_weight_mmd_kld,
        random_mask=dispersion.shape[0],
        hidden_dims=parameters.hidden_dims[:],
        scaling_factor=scaling_factor,
        dataset_dir=parameters.input_dataset_parent_dir,
    )

    logger.info("Instantiated untrained model.")

    # make save dir, and model fname to fallback on
    save_dir = root_dir / timestamp / model_type_name
    save_dir.mkdir(parents=True, exist_ok=True)

    model_fname = f"{model_type_name}.pt"

    # return new model if we need to train it
    if train_model:
        # make model_fname that will be used if trainded_model_fname is None
        logger.info(f"`train_model` set to {train_model}, returning untrained model.")

        return model, save_dir, model_fname, train_model, timestamp

    else:
        # Load the trained VAE parameters
        # Note: Changed to weights_only=False to avoid the warning below. Check if it is
        #  the correct way to do it or it creates issues.
        # FutureWarning: You are using `torch.load` with `weights_only=False` (the
        #  current default value), which uses the default pickle module implicitly.
        #  [...]
        # 20250109: changed to weights_only=True as recommended, seems to work
        if trained_model_fname is None:
            trained_model_dir = save_dir
            trained_model_fname = model_fname

        logger.info(
            f"Loading trained model state from {trained_model_dir / trained_model_fname}."
        )
        # Use recommended way to load the model
        try:
            model.load_state_dict(
                torch.load(
                    trained_model_dir / trained_model_fname,
                    weights_only=True,
                    map_location=device,
                )
            )
            logger.info("Model loaded.")
            logger.info("Setting model to evaluation mode.")
            model.eval()
        except FileNotFoundError:
            logger.warning(
                f"Model {trained_model_fname} not found in {trained_model_dir}. "
                "Returning untrained model and flagging the model for training."
            )
            train_model = True

        # Note that here things will be saved in the same directory as the trained model
        #   so the timestamp is not espcially relevant and could in principle be anything
        #  If the timestamp is any timestamp, that is ok. If the timestamp is inferred
        #   from the folder structure, I assume that overwriting things is ok,
        #   given we are loading a model and not experimenting with hyperparameters
        return model, trained_model_dir, trained_model_fname, train_model, timestamp


## ============================================================================= ##