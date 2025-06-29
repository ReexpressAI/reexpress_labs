import numpy as np

import argparse
import uuid

import data_utils


REEXPRESS_ID_KEY = "id"
REEXPRESS_LABEL_KEY = "label"
REEXPRESS_DOCUMENT_KEY = "document"
REEXPRESS_ATTRIBUTES_KEY = "attributes"
REEXPRESS_EMBEDDING_KEY = "embedding"

EXPECTED_EMBEDDING_SIZE = 8194

LOG_PROB_MODEL_RESPONSE_KEY = "LOG_PROB_MODEL"
REASONING_MODEL_RESPONSE_KEY = "REASONING_MODEL"
VERIFICATION_CLASSIFICATION_KEY = "verification_classification"
CONFIDENCE_IN_CLASSIFICATION_KEY = "confidence_in_classification"

EXPECTED_UNPROCESSED_ATTRIBUTES_LENGTH = 10
# log prob + soft one hot + soft one hot; see construct_document_attributes(response_obj)
EXPECTED_ATTRIBUTES_LENGTH = EXPECTED_UNPROCESSED_ATTRIBUTES_LENGTH * 2 + 2 + 2


def get_confidence_soft_one_hot_list(is_verified, verbalized_confidence):
    assert 0.0 <= verbalized_confidence <= 1.0
    if is_verified:
        return [0.0, float(verbalized_confidence)]
    else:
        return [float(verbalized_confidence), 0.0]


def construct_document_attributes(response_obj):
    # log probability model processed log probabilities | log probability model soft one hot by verbalized uncertainty | reasoning model  soft one hot by verbalized uncertainty
    # (negative/unverified | positive/verified)
    is_verified_log_prob = response_obj[LOG_PROB_MODEL_RESPONSE_KEY][VERIFICATION_CLASSIFICATION_KEY]
    confidence_log_prob = response_obj[LOG_PROB_MODEL_RESPONSE_KEY][CONFIDENCE_IN_CLASSIFICATION_KEY]
    confidence_soft_one_hot_list_log_prob = get_confidence_soft_one_hot_list(is_verified_log_prob, confidence_log_prob)
    unprocessed_attributes = response_obj[REEXPRESS_ATTRIBUTES_KEY]
    assert len(unprocessed_attributes) == EXPECTED_UNPROCESSED_ATTRIBUTES_LENGTH
    attributes = np.zeros(EXPECTED_UNPROCESSED_ATTRIBUTES_LENGTH * 2)
    if is_verified_log_prob:
        attributes[EXPECTED_UNPROCESSED_ATTRIBUTES_LENGTH:] = unprocessed_attributes
    else:
        attributes[0:EXPECTED_UNPROCESSED_ATTRIBUTES_LENGTH] = unprocessed_attributes
    reexpression_attributes = [float(x) for x in attributes.tolist()]
    reexpression_attributes.extend(confidence_soft_one_hot_list_log_prob)
    is_verified_reasoning = response_obj[REASONING_MODEL_RESPONSE_KEY][VERIFICATION_CLASSIFICATION_KEY]
    confidence_reasoning = response_obj[REASONING_MODEL_RESPONSE_KEY][CONFIDENCE_IN_CLASSIFICATION_KEY]
    confidence_soft_one_hot_list_reasoning = get_confidence_soft_one_hot_list(is_verified_reasoning, confidence_reasoning)
    reexpression_attributes.extend(confidence_soft_one_hot_list_reasoning)

    assert len(reexpression_attributes) == EXPECTED_ATTRIBUTES_LENGTH

    assert len(response_obj[REEXPRESS_EMBEDDING_KEY]) == EXPECTED_EMBEDDING_SIZE
    embedding = response_obj[REEXPRESS_EMBEDDING_KEY] #[float(x) for x in response_obj[REEXPRESS_EMBEDDING_KEY]]
    return reexpression_attributes, embedding


def construct_reexpress_format(options, json_list):
    formatted_json_output = []
    existing_ids = set()
    for json_obj in json_list:
        # negative example
        new_dict = {}
        new_dict[REEXPRESS_LABEL_KEY] = 0
        document_id = json_obj["original_line_id"]
        assert document_id not in existing_ids
        existing_ids.add(document_id)
        if options.anonymize_and_streamline:
            if options.construct_for_support_addition:
                document_id = "add_" + str(uuid.uuid4())
            else:
                document_id = str(uuid.uuid4())
        new_dict[REEXPRESS_ID_KEY] = f"neg_{document_id}"
        question_from_conversation = json_obj["instance"]["question_from_conversation"].strip()
        ai_response = json_obj["instance"]["incorrect_solution"].strip()
        document = f"<question> {question_from_conversation} </question> <ai_response> {ai_response} </ai_response>"
        if options.anonymize_and_streamline:
            new_dict[REEXPRESS_DOCUMENT_KEY] = ""
        else:
            new_dict[REEXPRESS_DOCUMENT_KEY] = document
        response_obj = json_obj["neg_response_object"]
        reexpression_attributes, embedding = construct_document_attributes(response_obj)
        new_dict[REEXPRESS_ATTRIBUTES_KEY] = reexpression_attributes
        new_dict[REEXPRESS_EMBEDDING_KEY] = embedding
        formatted_json_output.append(new_dict)
        # positive example
        new_dict = {}
        new_dict[REEXPRESS_LABEL_KEY] = 1
        new_dict[REEXPRESS_ID_KEY] = f"pos_{document_id}"
        ai_response = json_obj["instance"]["provided_solution"].strip()
        document = f"<question> {question_from_conversation} </question> <ai_response> {ai_response} </ai_response>"
        if options.anonymize_and_streamline:
            new_dict[REEXPRESS_DOCUMENT_KEY] = ""
        else:
            new_dict[REEXPRESS_DOCUMENT_KEY] = document
        response_obj = json_obj["pos_response_object"]
        reexpression_attributes, embedding = construct_document_attributes(response_obj)
        new_dict[REEXPRESS_ATTRIBUTES_KEY] = reexpression_attributes
        new_dict[REEXPRESS_EMBEDDING_KEY] = embedding
        formatted_json_output.append(new_dict)

    return formatted_json_output


def main():
    parser = argparse.ArgumentParser(description="-----[Archive data]-----")
    parser.add_argument("--input_file", default="", help="")
    parser.add_argument("--output_file", default="", help="")
    parser.add_argument("--anonymize_and_streamline", default=False,
                        action='store_true', help="")
    parser.add_argument("--construct_for_support_addition", default=False,
                        action='store_true', help="")

    options = parser.parse_args()

    json_list = data_utils.read_jsons_lines_file(options.input_file)
    formatted_json_list = construct_reexpress_format(options, json_list)
    data_utils.save_json_lines(options.output_file, formatted_json_list)


if __name__ == "__main__":
    main()
