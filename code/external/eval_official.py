# Copy from: https://github.com/ArneBinder/dialam-2024-shared-task/blob/main/src/evaluation/eval_official.py

import pyrootutils
pyrootutils.setup_root(search_from=__file__, indicator=[".project-root"], pythonpath=True)

import argparse
import itertools
import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

from sklearn.metrics import precision_recall_fscore_support


from external.nodeset_utils import Nodeset, process_all_nodesets, read_nodeset

logger = logging.getLogger(__name__)


def eval_arguments(
    nodeset_id: str,
    predictions_dir: str,
    gold_dir: str,
    nodeset: Optional[Nodeset] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:

    preds = read_nodeset(predictions_dir, nodeset_id) if nodeset is None else nodeset
    truth = read_nodeset(gold_dir, nodeset_id)

    proposition_dict = {}
    proposition_list = []
    true_inference_list = []
    true_conflict_list = []
    true_rephrase_list = []
    pred_inference_list = []
    pred_conflict_list = []
    pred_rephrase_list = []

    # Get the list of proposition nodes
    for node in truth["nodes"]:
        if node["type"] == "I":
            proposition_list.append(node["nodeID"])
            proposition_dict[node["nodeID"]] = node["text"]

    # Check truth relations
    for node in truth["nodes"]:
        if node["type"] == "RA":
            inference_id = node["nodeID"]

            for edge in truth["edges"]:
                if edge["fromID"] == inference_id:
                    conclusion_id = edge["toID"]
                    for edge in truth["edges"]:
                        if edge["toID"] == inference_id:
                            premise_id = edge["fromID"]
                            if premise_id in proposition_list:
                                true_inference_list.append([premise_id, conclusion_id, 0])
                    break

        elif node["type"] == "CA":
            conflict_id = node["nodeID"]

            for edge in truth["edges"]:
                if edge["fromID"] == conflict_id:
                    conf_to = edge["toID"]
                    for edge in truth["edges"]:
                        if edge["toID"] == conflict_id:
                            conf_from = edge["fromID"]
                            if conf_from in proposition_list:
                                true_conflict_list.append([conf_from, conf_to, 1])
                    break

        elif node["type"] == "MA":
            rephrase_id = node["nodeID"]

            for edge in truth["edges"]:
                if edge["fromID"] == rephrase_id:
                    reph_to = edge["toID"]
                    for edge in truth["edges"]:
                        if edge["toID"] == rephrase_id:
                            reph_from = edge["fromID"]
                            if reph_from in proposition_list:
                                true_rephrase_list.append([reph_from, reph_to, 2])
                    break

    # Check predicted relation
    for node in preds["nodes"]:
        if node["type"] == "RA":
            inference_id = node["nodeID"]

            for edge in preds["edges"]:
                if edge["fromID"] == inference_id:
                    conclusion_id = edge["toID"]
                    for edge in preds["edges"]:
                        if edge["toID"] == inference_id:
                            premise_id = edge["fromID"]
                            if premise_id in proposition_list:
                                pred_inference_list.append([premise_id, conclusion_id, 0])
                    break

        elif node["type"] == "CA":
            conflict_id = node["nodeID"]

            for edge in preds["edges"]:
                if edge["fromID"] == conflict_id:
                    conf_to = edge["toID"]
                    for edge in preds["edges"]:
                        if edge["toID"] == conflict_id:
                            conf_from = edge["fromID"]
                            if conf_from in proposition_list:
                                pred_conflict_list.append([conf_from, conf_to, 1])
                    break

        elif node["type"] == "MA":
            rephrase_id = node["nodeID"]

            for edge in preds["edges"]:
                if edge["fromID"] == rephrase_id:
                    reph_to = edge["toID"]
                    for edge in preds["edges"]:
                        if edge["toID"] == rephrase_id:
                            reph_from = edge["fromID"]
                            if reph_from in proposition_list:
                                pred_rephrase_list.append([reph_from, reph_to, 2])
                    break

    # ID to text
    """
    text_inference_list = []
    for inference in inference_list:
        text_inference_list.append([proposition_dict[inference[0]], proposition_dict[inference[1]], 1])
    text_conflict_list = []
    for conflict in conflict_list:
        text_conflict_list.append([proposition_dict[conflict[0]], proposition_dict[conflict[1]], 2])
    text_rephrase_list = []
    for rephrase in rephrase_list:
        text_rephrase_list.append([proposition_dict[rephrase[0]], proposition_dict[rephrase[1]], 3])
    """

    p_c = itertools.permutations(proposition_list, 2)
    proposition_combinations = []
    for p in p_c:
        proposition_combinations.append([p[0], p[1]])

    y_true = []
    y_pred = []
    for comb in proposition_combinations:
        added_true = False
        added_pred = False

        # Prepare Y true
        for inference in true_inference_list:
            if inference[0] == comb[0] and inference[1] == comb[1]:
                y_true.append(0)
                added_true = True
                break
        if not added_true:
            for conflict in true_conflict_list:
                if conflict[0] == comb[0] and conflict[1] == comb[1]:
                    y_true.append(1)
                    added_true = True
                    break
        if not added_true:
            for rephrase in true_rephrase_list:
                if rephrase[0] == comb[0] and rephrase[1] == comb[1]:
                    y_true.append(2)
                    added_true = True
                    break
        if not added_true:
            y_true.append(3)

        # Prepare Y pred
        for inference in pred_inference_list:
            if inference[0] == comb[0] and inference[1] == comb[1]:
                y_pred.append(0)
                added_pred = True
                break
        if not added_pred:
            for conflict in pred_conflict_list:
                if conflict[0] == comb[0] and conflict[1] == comb[1]:
                    y_pred.append(1)
                    added_pred = True
                    break
        if not added_pred:
            for rephrase in pred_rephrase_list:
                if rephrase[0] == comb[0] and rephrase[1] == comb[1]:
                    y_pred.append(2)
                    added_pred = True
                    break
        if not added_pred:
            y_pred.append(3)

    return handle_true_pred(
        y_true=y_true, y_pred=y_pred, focused_value=3, nodeset_id=nodeset_id, verbose=verbose
    )


def eval_illocutions(
    nodeset_id: str,
    predictions_dir: str,
    gold_dir: str,
    nodeset: Optional[Nodeset] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:

    preds = read_nodeset(predictions_dir, nodeset_id) if nodeset is None else nodeset
    truth = read_nodeset(gold_dir, nodeset_id)

    proposition_dict = {}
    proposition_list = []
    locution_dict = {}
    locution_list = []
    true_illocution_list = []
    pred_illocution_list = []

    # Get the list of proposition and locution nodes
    for node in truth["nodes"]:
        if node["type"] == "I":
            proposition_list.append(node["nodeID"])
            proposition_dict[node["nodeID"]] = node["text"]
        elif node["type"] == "L":
            locution_list.append(node["nodeID"])
            locution_dict[node["nodeID"]] = node["text"]

    proploc_list = proposition_list + locution_list

    # Check truth illocutions
    for node in truth["nodes"]:
        if node["type"] == "YA":
            illocution_id = node["nodeID"]
            illocution_type = node["text"]

            for edge in truth["edges"]:
                if edge["fromID"] == illocution_id:
                    target_id = edge["toID"]
                    for edge in truth["edges"]:
                        if edge["toID"] == illocution_id:
                            source_id = edge["fromID"]
                            if source_id in proploc_list and target_id in proploc_list:
                                true_illocution_list.append(
                                    [source_id, target_id, illocution_type]
                                )
                    break

    # Check predicted illocutions
    for node in preds["nodes"]:
        if node["type"] == "YA":
            illocution_id = node["nodeID"]
            illocution_type = node["text"]

            for edge in preds["edges"]:
                if edge["fromID"] == illocution_id:
                    target_id = edge["toID"]
                    for edge in preds["edges"]:
                        if edge["toID"] == illocution_id:
                            source_id = edge["fromID"]
                            if source_id in proploc_list and target_id in proploc_list:
                                pred_illocution_list.append(
                                    [source_id, target_id, illocution_type]
                                )
                    break

    # print(true_illocution_list)
    # print(pred_illocution_list)

    p_c = itertools.product(locution_list, proposition_list)
    proploc_combinations = []
    for p in p_c:
        proploc_combinations.append([p[0], p[1]])

    y_true = []
    y_pred = []
    for comb in proploc_combinations:
        added_true = False
        added_pred = False

        # Prepare Y true
        for illocution in true_illocution_list:
            if illocution[0] == comb[0] and illocution[1] == comb[1]:
                y_true.append(illocution[2])
                added_true = True
                break

        if not added_true:
            y_true.append("None")

        # Prepare Y pred
        for illocution in pred_illocution_list:
            if illocution[0] == comb[0] and illocution[1] == comb[1]:
                y_pred.append(illocution[2])
                added_pred = True
                break

        if not added_pred:
            y_pred.append("None")

    return handle_true_pred(
        y_true=y_true, y_pred=y_pred, focused_value="None", nodeset_id=nodeset_id, verbose=verbose
    )


def handle_true_pred(
    y_true: List,
    y_pred: List,
    focused_value: Union[str, int],
    nodeset_id: str,
    verbose: bool = True,
):
    if verbose:
        print(y_true)
        print(y_pred)

    focused_true = []
    focused_pred = []
    for i in range(len(y_true)):
        if y_true[i] != focused_value:
            focused_true.append(y_true[i])
            focused_pred.append(y_pred[i])

    zero_division: Union[str, float]
    if verbose:
        print(focused_true)
        print(focused_pred)
        zero_division = "warn"
    else:
        zero_division = 0.0

    result = {}
    if len(y_true) > 0:
        result_general = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=zero_division
        )
        if verbose:
            print("General", result_general)
        result["general"] = {
            "p": result_general[0],
            "r": result_general[1],
            "f1": result_general[2],
        }
    else:
        logger.warning(f"nodeset_id={nodeset_id}: No true relations found")

    if len(focused_true) > 0:
        result_focused = precision_recall_fscore_support(
            focused_true, focused_pred, average="macro", zero_division=zero_division
        )
        if verbose:
            print("Focused", result_focused)
        result["focused"] = {
            "p": result_focused[0],
            "r": result_focused[1],
            "f1": result_focused[2],
        }
    else:
        logger.warning(f"nodeset_id={nodeset_id}: No focused true relations found")

    return result


def eval_single_nodeset(mode: str, **kwargs):
    if mode == "arguments":
        return eval_arguments(**kwargs)
    elif mode == "illocutions":
        return eval_illocutions(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def flatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    # flatten arbitrary nested dict
    def items():
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in flatten_dict(value).items():
                    yield key + sep + subkey, subvalue
            else:
                yield key, value

    return dict(items())


def main(
    predictions_dir: str, nodeset_id: Optional[str] = None, show_progress: bool = True, **kwargs
):
    if nodeset_id is not None:
        result = eval_single_nodeset(
            nodeset_id=nodeset_id, predictions_dir=predictions_dir, **kwargs
        )
        print(result)
    else:
        result = defaultdict(list)
        for nodeset_id, result_or_error in process_all_nodesets(
            func=eval_single_nodeset,
            nodeset_dir=predictions_dir,
            show_progress=show_progress,
            predictions_dir=predictions_dir,
            **kwargs,
        ):
            if isinstance(result_or_error, Exception):
                logger.error(f"nodeset={nodeset_id}: Failed to process: {result_or_error}")
            else:
                result_flat = flatten_dict(result_or_error, sep=".")
                if any(math.isnan(stat_value) for stat_value in result_flat.values()):
                    logger.error(
                        f"nodeset={nodeset_id}: NaN value found in result, skipping this nodeset: {result_flat}"
                    )
                    continue
                for stat_name, stat_value in flatten_dict(result_or_error, sep=".").items():
                    result[stat_name].append(stat_value)
        for stat_name, stat_values in result.items():
            print(f"{stat_name}: {sum(stat_values) / len(stat_values)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate arguments")
    parser.add_argument(
        "--mode", type=str, required=True, help="Mode of evaluation (arguments/illocutions)"
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        required=True,
        help="Path to the directory containing the gold nodesets",
    )
    parser.add_argument(
        "--gold_dir",
        type=str,
        required=True,
        help="Path to the directory containing the predicted nodesets",
    )
    parser.add_argument(
        "--nodeset_id",
        type=str,
        help="ID of the nodeset to evaluate. If not provided, evaluate all nodesets in predictions_dir",
    )
    parser.add_argument(
        "--silent", dest="verbose", action="store_false", help="Whether to show verbose output"
    )

    args = vars(parser.parse_args())
    logging.basicConfig(level=logging.INFO)
    main(**args)
