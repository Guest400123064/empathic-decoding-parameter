# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Date: 04-30-2023
# =============================================================================
"""This script simply collect dialogues between a 'patient' bot and a 
    'therapist' bot given the configuration from an experiment config file. 
    The config file includes parameters like decoding hyperparameters and persona."""

from typing import Callable, Dict, List, Any, Tuple, Sequence

import os
import pathlib

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from easydict import EasyDict

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")

from src.dialogue import DialogueCollector


def load_config() -> EasyDict[str, Any]:

    import argparse
    import toml

    parser = argparse.ArgumentParser(description="Collect dialogues using self-conversation.")
    parser.add_argument("--config-path", 
                        type=str, 
                        help="Path to the experiment config file.")
    
    args = parser.parse_args()
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file {args.config_path} does not exist.")
    
    with open(args.config_path, "r") as f:
        config = EasyDict(toml.load(f))
    return EasyDict(config)


def main():

    import json
    
    config = load_config()
    print(json.dumps(config, indent=True))


if __name__ == "__main__":
    main()
