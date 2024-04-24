"""This module is used to discover specified experiment scenarios from configs
and/or to enable user to cherry-pick a particular experiment scenario"""

import logging
import tkinter as tk
from pathlib import Path
from typing import Union

import inquirer  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from src.utils.project_info import get_config_root

logger = logging.getLogger()


def discover_experiments_from_configs(
    root_config_path: Union[str, Path] = None
) -> pd.DataFrame:
    """
    This function iterates over configs path and discovers all specified experiments
    :param root_config_path:
    :return: pandas dataframe with each row corresponding to a given experiment
    """
    # helping function for filtering only subdirectories
    def get_subdirectories(root_path: Path) -> list[Path]:
        return [
            dir_name
            for dir_name in root_path.iterdir()
            if dir_name.is_dir() and dir_name.suffix != ".ignore"
        ]

    if root_config_path is None:
        root_config_path = get_config_root()

    # getting all model types
    root_config_path = Path(root_config_path)
    all_model_types = get_subdirectories(root_config_path)
    # predefining lists to store columns of the future dataframe
    model_types = []
    model_versions = []
    for model_type_path in all_model_types:
        model_type_name = model_type_path.stem
        all_model_versions = get_subdirectories(model_type_path)
        for model_version_path in all_model_versions:
            version_name = model_version_path.stem
            model_types.append(model_type_name)
            model_versions.append(version_name)

    final_df = pd.DataFrame(
        {
            "model_type": model_types,
            "model_version": model_versions,
        }
    )
    return final_df


class OptionSelector(tk.Tk):
    """
    This is a basic UI option selector class
    """

    # based on https://www.pythontutorial.net/tkinter/tkinter-optionmenu/
    def __init__(self, title: str, options: np.ndarray, prompt_text: str):
        """
        UI option selector constructor
        :param title: widget title
        :param options: options
        :param prompt_text: display text for prompt
        """
        super().__init__()
        self.title(title)

        # initialize data
        self.options = options
        self.prompt_text = prompt_text

        # set up variable
        self.option_var = tk.StringVar(self)

        # create widget
        self.create_wigets()

        self.attributes("-topmost", True)

        self.selection = self.options[0]

    def create_wigets(self):
        """
        Widgets creation
        """
        # padding for widgets using the grid layout
        paddings = {"padx": 5, "pady": 5}

        # label
        label = tk.Label(self, text=self.prompt_text)
        label.grid(column=0, row=0, sticky=tk.W, **paddings)

        # option menu
        self.option_var.set(self.options[0])
        option_menu = tk.OptionMenu(
            self, self.option_var, *self.options, command=self.option_changed
        )
        self.selection = self.option_var.get()

        option_menu.grid(column=1, row=0, sticky=tk.W, **paddings)

        def _destroy_quit():
            self.destroy()
            self.quit()

        confirm_button = tk.Button(self, text="Ok", command=_destroy_quit)
        confirm_button.grid(column=1, row=1, sticky=tk.W, **paddings)

    def option_changed(self, *args):  # pylint: disable=W0613
        """Option change"""
        self.selection = self.option_var.get()


def collect_single_experiment_arguments(
    config_folder_path: Union[str, Path] = None
) -> dict:
    """
    Opens dialog for specification of particular experiment to run (out of all possible experiments)
    :param config_folder_path:
    :return: Dictionary with selected configuration
    """
    # discovering all configs
    configs_df = discover_experiments_from_configs(config_folder_path)

    # checking if it runs on a head-less node
    try:
        tk.Tk().destroy()
        headless = False
    except tk.TclError:
        headless = True

    def _select_out_of_options(available_options: np.ndarray, prompt: str) -> str:
        if headless:
            question = [
                inquirer.List(
                    "next_option",
                    message=prompt,
                    choices=list(available_options),
                ),
            ]
            chosen_options = inquirer.prompt(question)
            return chosen_options["next_option"]

        selector = OptionSelector(
            title=prompt,
            options=available_options,
            prompt_text=prompt,
        )
        selector.mainloop()
        return selector.selection

    args = {}

    available_models = configs_df["model_type"].unique()
    if len(available_models) == 1:
        args["model_type"] = available_models[0]
    else:
        args["model_type"] = _select_out_of_options(
            available_models,
            "Select a model type: ",
        )

    split_model_config_df = configs_df[configs_df["model_type"] == args["model_type"]]

    available_model_versions = split_model_config_df["model_version"].unique()
    if len(available_model_versions) == 1:
        args["model_version"] = available_model_versions[0]
    else:
        args["model_version"] = _select_out_of_options(
            available_model_versions,
            f'For model type \'{args["model_type"]}\'\nSelect model version: ',
        )
    return args
