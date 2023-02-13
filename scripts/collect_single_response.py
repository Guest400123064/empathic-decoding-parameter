# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Date: 01-26-2023
# =============================================================================
"""This script simply collect the single response from the model with 
    different hyperparameter configurations. The 'therapist' will ask 
    a single question and the 'patient' will give a single response."""

from typing import Callable, Dict, List, Any

import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

import openai
openai.organization = os.getenv("OPENAI_ORGANIZATION")
openai.api_key = os.getenv("OPENAI_API_KEY")

import tqdm
from itertools import product

import json


STOP_SEQ = ["Therapist:", "Patient:"]
TEMPLATE = """
Below is a conversation between a patient and a psychotherapist.
Therapist: {question} Please give me as much detail as possible.
Patient:
""".strip()


def get_questions(path: str = "") -> List[str]:
    """Return a set of questions for the simulated patient."""
    
    return ["How would you feel if someone called you a jerk?",
            "How do birthday cakes make you feel?",
            "What would you most like to talk to a therapist about?"]


def get_static_params(path: str = "") -> Dict[str, Any]:
    """Return a dictionary with keys being the static hyperparameters
        and values being the values for the hyperparameters."""

    return {"model": "text-davinci-003",
            "max_tokens": 128,
            "stop": STOP_SEQ}


def get_tuned_params(path: str = "") -> List[Dict[str, Any]]:
    """Return a list of dictionaries with keys being the sampling hyperparameters
        and values being the values for the hyperparameters. It is the cross product
        of the possible values for the hyperparameters."""

    params = {"temperature": [0.7],
              "frequency_penalty": [0.01, 1.0, 1.99],}

    return [dict(zip(params, v)) for v in product(*params.values())]


if __name__ == "__main__":

    responses = []
    num_generation_per_question = 3

    config = get_static_params()
    for q in tqdm.tqdm(get_questions(), desc="All questions", position=0):
        config.update({"prompt": TEMPLATE.format(question=q)})
        for c in tqdm.tqdm(get_tuned_params(), desc="All configurations", position=1, leave=False):
            config.update(c)

            # Sample multiple times for each question per configuration
            for i in range(num_generation_per_question):
                response = openai.Completion.create(**config)
                data = {"question": q, 
                        "response": response["choices"][0]["text"].strip()}
                data.update(config)
                responses.append(data)

    with open(os.path.join(DATA_DIR, f"{config['model']}-single-response.json"), "w") as f:
        json.dump(responses, f, indent=4)
