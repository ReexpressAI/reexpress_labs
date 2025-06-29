from typing import Any, Callable, List, Tuple
import json
import os
import numpy as np

import argparse
import time
from pathlib import Path
import codecs

import torch
import asyncio

import data_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

model_path = "ibm-granite/granite-3.3-8b-instruct"
# device = "mps"
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
tokenizer = AutoTokenizer.from_pretrained(
        model_path
)
set_seed(42)
kSLEEP_CONSTANT = 40
VERIFICATION_CLASSIFICATION_KEY = "verification_classification"
CONFIDENCE_IN_CLASSIFICATION_KEY = "confidence_in_classification"
SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY = "short_explanation_for_classification_confidence"

FIELD_PROBABILITIES_KEY = "field_probabilities"
LOG_PROB_MODEL_RESPONSE_KEY = "LOG_PROB_MODEL"
REASONING_MODEL_RESPONSE_KEY = "REASONING_MODEL"

REEXPRESS_ID_KEY = "id"
REEXPRESS_LABEL_KEY = "label"
REEXPRESS_DOCUMENT_KEY = "document"
REEXPRESS_ATTRIBUTES_KEY = "attributes"
REEXPRESS_EMBEDDING_KEY = "embedding"

EXPECTED_EMBEDDING_SIZE = 8194


def get_embedding(document_text: str) -> list[float]:
    conv = [{"role": "user",
             "content": document_text}]
    input_ids = tokenizer.apply_chat_template(conv, return_tensors="pt", thinking=False,
                                              return_dict=True, add_generation_prompt=True).to(device)
    outputs = model.generate(
        **input_ids,
        max_new_tokens=1,
        output_hidden_states=True,
        return_dict_in_generate=True,
        output_scores=True,
    )
    hidden_states = outputs.hidden_states
    scores = outputs.scores
    no_id = tokenizer.vocab["No"]
    yes_id = tokenizer.vocab["Yes"]
    probs = torch.softmax(scores[0], dim=-1)
    # average of all hidden :: current hidden :: no_prob :: yes_prob
    embedding = torch.cat([
        torch.mean(hidden_states[0][-1][0], dim=0).unsqueeze(0),
        hidden_states[0][-1][0][-1, :].unsqueeze(0),
        probs[0:1, no_id].unsqueeze(0),
        probs[0:1, yes_id].unsqueeze(0)
    ], dim=-1)
    embedding = [float(x) for x in embedding[0].cpu().numpy().tolist()]
    assert len(embedding) == EXPECTED_EMBEDDING_SIZE
    return embedding


def get_model_explanations_formatted_as_binary_agreement_prompt(log_prob_model_explanation,
                                                                reasoning_model_explanation) -> str:
    formatted_output_string = f"Do the following model explanations agree that the response is correct? <model1_explanation> {log_prob_model_explanation} </model1_explanation> <model2_explanation> {reasoning_model_explanation} </model2_explanation> Yes or No?"
    return formatted_output_string


def llm_api_controller(log_prob_model_explanation: str, reasoning_model_explanation: str):
    try:
        prompt = get_model_explanations_formatted_as_binary_agreement_prompt(log_prob_model_explanation,
                                                                             reasoning_model_explanation)
        agreement_embedding = get_embedding(document_text=prompt)
        return agreement_embedding
    except:
        return None


def generation_is_valid_for_verification(json_list, instance_index) -> bool:
    instance = json_list[instance_index]
    neg_response_object = instance["neg_response_object"]
    pos_response_object = instance["pos_response_object"]

    return neg_response_object[LOG_PROB_MODEL_RESPONSE_KEY][SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY].strip() != "" and \
        neg_response_object[REASONING_MODEL_RESPONSE_KEY][SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY].strip() != "" and \
        pos_response_object[LOG_PROB_MODEL_RESPONSE_KEY][SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY].strip() != "" and \
        pos_response_object[REASONING_MODEL_RESPONSE_KEY][SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY].strip() != ""


def get_model_explanations(response_object):
    return response_object[LOG_PROB_MODEL_RESPONSE_KEY][SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY].strip(), response_object[REASONING_MODEL_RESPONSE_KEY][SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY].strip()


def get_existing_ids(filepath_with_name):
    existing_ids = set()
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            existing_ids.add(json_obj["original_line_id"])
    return existing_ids


def construct_embedding_verification_and_save_shard(indexes_as_list, json_list, output_file, dataset_id):
    count_incomplete_responses = 0
    # time.sleep(torch.abs(torch.randn(1)).item() / kSLEEP_CONSTANT)
    if Path(output_file).exists():
        existing_ids = get_existing_ids(output_file)
    else:
        existing_ids = set()

    for row_index in indexes_as_list:
        if row_index > len(json_list) - 1:
            break
        if row_index % 1000 == 0:
            print(f"row_index: {row_index} in {output_file}")
        if json_list[row_index]["original_line_id"] in existing_ids:
            continue
        if dataset_id == "openthoughts" or dataset_id.startswith("multiplechoice") or options.dataset_id.startswith("fever"):
            if not generation_is_valid_for_verification(json_list, row_index):
                continue
            neg_log_prob_model_explanation, neg_reasoning_model_explanation = \
                get_model_explanations(json_list[row_index]["neg_response_object"])
            pos_log_prob_model_explanation, pos_reasoning_model_explanation = \
                get_model_explanations(json_list[row_index]["pos_response_object"])
        else:
            assert False
        neg_embedding = \
            llm_api_controller(log_prob_model_explanation=neg_log_prob_model_explanation,
                               reasoning_model_explanation=neg_reasoning_model_explanation)
        # time.sleep(torch.abs(torch.randn(1)).item() / 40)
        pos_embedding = \
            llm_api_controller(log_prob_model_explanation=pos_log_prob_model_explanation,
                               reasoning_model_explanation=pos_reasoning_model_explanation)
        if neg_embedding is not None and \
                pos_embedding is not None:
            json_obj = json_list[row_index]
            json_obj["neg_response_object"][REEXPRESS_EMBEDDING_KEY] = \
                neg_embedding
            json_obj["pos_response_object"][REEXPRESS_EMBEDDING_KEY] = \
                pos_embedding
            data_utils.save_by_appending_json_lines(output_file, [json_obj])
            existing_ids.add(json_list[row_index]["original_line_id"])
        else:
            count_incomplete_responses += 1
    print(f"Lines {indexes_as_list[0]}-{indexes_as_list[-1]} complete with {count_incomplete_responses} incomplete instances skipped.")


async def run_tasks_dynamically(task_configs: List[Tuple[Callable, Tuple[Any, ...]]]):
    """
    Dynamically run a list of tasks concurrently.

    Args:
        task_configs: A list of tuples, each containing:
            - A function to run
            - A tuple of arguments to pass to the function

    Returns:
        List of results from all tasks
    """
    tasks = []

    # Create tasks dynamically
    for func, args in task_configs:
        task = asyncio.to_thread(func, *args)
        tasks.append(task)

    # Run all tasks concurrently
    await asyncio.gather(*tasks)


async def main(options, json_list):

    # Create a list of task configurations dynamically
    task_configs = []
    if options.start_index < len(json_list):  # invalid start/end are safely ignored to simplify bash scripts
        total_size = len(json_list[options.start_index: options.start_index+options.total_lines])
        if total_size > 0:
            row_indexes = np.arange(options.start_index, options.start_index+total_size)
            for np_index_list_for_shard in np.array_split(row_indexes, options.shards):
                indexes_as_list = [int(x) for x in np_index_list_for_shard.tolist()]
                if len(indexes_as_list) > 0:
                    output_file = os.path.join(options.output_dir, f"genai_verification_{options.dataset_id.strip()}_synthetic_neg_with_pos_{indexes_as_list[0]}_{indexes_as_list[-1]}_with_embedding.jsonl")
                    task_configs.append((construct_embedding_verification_and_save_shard, (indexes_as_list, json_list, output_file, options.dataset_id)))
            if len(task_configs) > 0:
                print(f"Running {len(task_configs)} tasks concurrently...")
                await run_tasks_dynamically(task_configs)

    print(f"All tasks completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="-----[Add embedding data to JSON objects]-----")
    parser.add_argument("--dataset_id", default="", help="'openthoughts'; or 'multiplechoice' as a prefix; or 'fever' as a prefix")
    parser.add_argument("--input_file", default="", help="")
    parser.add_argument("--start_index", default=0, type=int, help="")
    parser.add_argument("--total_lines", default=1000, type=int, help="")
    parser.add_argument("--shards", default=10, type=int, help="")
    parser.add_argument("--output_dir", default="", help="")
    options = parser.parse_args()

    assert options.dataset_id == "openthoughts" or options.dataset_id.startswith("multiplechoice") or options.dataset_id.startswith("fever")
    start_time = time.time()
    time.sleep(torch.abs(torch.randn(1)).item())
    json_list = data_utils.read_jsons_lines_file(options.input_file)
    print("Read input")
    asyncio.run(main(options, json_list))
    cumulative_time = time.time() - start_time
    print(f"Cumulative running time: {cumulative_time}")
