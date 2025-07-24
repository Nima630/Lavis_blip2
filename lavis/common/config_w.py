# lavis/common/config_w.py (Simplified, Registry-Free)

import logging
import json
from typing import Dict
from omegaconf import OmegaConf


class Config:
    def __init__(self, args):
        print("[TRACE] Config initialized")
        self.config = {}
        self.args = args

        # user_config = self._build_opt_list(self.args.options)
        user_config = self._build_opt_list(self.args.options) if self.args.options else {}

        config = OmegaConf.load(self.args.cfg_path)

        runner_config = self.build_runner_config(config)
        model_config = self.build_model_config(config, **user_config)
        dataset_config = self.build_dataset_config(config)

        self.config = OmegaConf.merge(
            runner_config, model_config, dataset_config, user_config
        )
        print("[DEBUG] Loaded raw config:\n", OmegaConf.to_container(config, resolve=True))

    # def _build_opt_list(self, opts):
    #     if opts is None:
    #         return []
    #     if len(opts) == 0:
    #         return opts
    #     has_equal = opts[0].find("=") != -1
    #     return opts if has_equal else [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]

    def _build_opt_list(self, opts):  # will always return dict now
        if opts is None or len(opts) == 0:
            return {}
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.from_dotlist(opts_dot_list)


    @staticmethod
    def build_model_config(config, **kwargs):
        model = config.get("model", None)
        assert model is not None, "Missing model configuration section."
        # Directly return whatever is under 'model' without registry
        return {"model": model}

    @staticmethod
    def build_runner_config(config):
        return {"run": config.run}

    @staticmethod
    def build_dataset_config(config):
        datasets = config.get("datasets", None)
        assert datasets is not None, "Missing 'datasets' key in config."
        return {"datasets": datasets}

    @property
    def run_cfg(self):
        return self.config["run"]

    @property
    def datasets_cfg(self):
        return self.config["datasets"]

    @property
    def model_cfg(self):
        return self.config["model"]

    def pretty_print(self):
        logging.info("\n=====  Running Parameters    =====")
        logging.info(self._convert_node_to_json(self.config["run"]))

        logging.info("\n======  Dataset Attributes  ======")
        for name, dataset in self.config["datasets"].items():
            logging.info(f"\n======== {name} =======")
            logging.info(self._convert_node_to_json(dataset))

        logging.info("\n======  Model Attributes  ======")
        logging.info(self._convert_node_to_json(self.config["model"]))

    # def _convert_node_to_json(self, node):
    #     container = OmegaConf.to_container(node, resolve=True)
    #     return json.dumps(container, indent=4, sort_keys=True)

    def to_dict(self):
        return OmegaConf.to_container(self.config)

    def get_config(self):
        return self.config
