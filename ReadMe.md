# QUEST: A machine learning framework to generate quasar spectra

QUEST (Quasar Unsupervised Encoder and Synthesis Tool) is an implementation of a Variational Auto-Encoder (VAE) with the primary purpose of generating realistic quasar spectra and post-processing them to obtain synthetic quasar photometry. QUEST can also be used to reconstruct spectra with limited wavelength coverage, absorption systems, and even the continuum blueward of the Lyman-$\alpha$ emission line (with some caveats).

Check out the [paper](https://arxiv.org/abs/2510.23206) for a full breakdown of its capabilities and limitations.

## Install instructions

We recommend installing QUEST in a dedicated virtual environment. We currently leverage [Atelier](https://github.com/jtschindler/atelier) to sample from a luminosity function. Unfortunately, due to the design of PyPI, we currently cannot include [Atelier](https://github.com/jtschindler/atelier) as a requirement. It is thus necessary to install it manually:

1.  **Create and activate a virtual environment** (e.g., using `venv`):
    ```bash
    python -m venv venv_name
    source venv_name/bin/activate  # Linux/macOS
    ```

2.  **Install Atelier:**
    ```bash
    git clone https://github.com/jtschindler/atelier
    cd Atelier
    pip install emcee
    pip install -e .
    ```

3.  **Install QUEST from source:**
    ```bash
    git clone https://github.com/cosmic-dawn-group/QUEST.git
    cd QUEST
    pip install -e .
    ```

3.  or **Install from PyPI:**
    ```bash
    pip install quest_qso
    ```
    *Note: Updates on the PyPI version might lag slightly behind the main repository.*

    **A note of caution:** QUEST has been tested as much as possible, but, especially at the beginning, there will be bugs and aspects to improve. Please report any issue you find using the **GitHub Issues** tab, or consider sending us an email ([francesco.guarneri@uni-hamburg.de](mailto:francesco.guarneri@uni-hamburg.de)).

## Environment variables

QUEST uses a few environment variables to set its output folders and ensure that it does not overuse resources on shared machines.

* `QUEST_LOCALPATH` — General cache directory. This is the primary folder used to download all cached files and save generated spectra/photometry in the examples. If downloaded using the utilities included in QUEST, this will also contain the datasets used to train the model.
* `QUEST_PATH_GDOWN_CACHE` - QUEST will try to redirect the cache folder of `gdown` to a custom folder. This will be create at runtime if necessary. Note: this is likely a hack and could have unintended consequences on gdown. Please also note that the code does not check for overwrites: make sure the folder you are using is empty or does not exist yet, and there is no conflict with existing files. `gdown` will in any case create subfolders in `QUEST_PATH_GDOWN_CACHE` to hold each file.
* `QUEST_LOG_TO_FILE` - QUEST logs to the terminal by default. However, if this variable is set to `True` or `1`, an additional log file will be created in `QUEST_LOCALPATH`.
* `QUEST_AM_I_ON_SHARED_SERVER` - If set to `True` or `1`, QUEST will limit its resource usage (see details in `__init__.py` -- make sure to customize this to your needs!).
* `QUEST_TORCH_SEED` - Sets the overall seed for `PyTorch`. If this is not set, the seed defaults to `42`. If negative, no seed is set. Otherwise, the seed will be set to the value of this variable.
* `QUEST_TORCH_DEBUG` - Effectively sets `torch.autograd.set_detect_anomaly(True)`. This should only be used to debug issues with the model, as it greatly slows down any PyTorch operation.

Environment variables can be set (for example, in `bash`) using the `export` command:

```bash
export QUEST_LOCALPATH="/path/to/your/cache/folder"
```

Usage
-----
Head over to the [examples](https://github.com/cosmic-dawn-group/QUEST/tree/main/src/quest_qso/examples) folder, where we've included Jupyter notebooks showing how to load the model for inference, sample from it, or generate synthetic photometry. If you install the package via `pip`, the example folder will be located in the `site-package` of your environment. Reaching this folder is generally cumbersome, and making changes to any file or folder inside `site-package` might cascade in a broken Python environment. In this case, we thus recommend downloading the notebook(s) from GitHub, place them in a folder of choice, and run them from there.

Contributing
------------
Contributions are more than welcome! Please open an issue to report problems, open PRs to contribute to the code, or just let us know if you have any feature requests! We are a small team but are happy to receive feedback!

License
-------
See `LICENSE` in the repository root.
