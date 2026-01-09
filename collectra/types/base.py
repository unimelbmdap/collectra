import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from rich import print

from ..commons import BaseEntity, Node, NodeStatus
from ..utils import (
    change_dir,
    error_msg,
    load_class_from_string,
    traceback_error,
)

__all__ = ["Data", "DataNode"]

import logging

logger = logging.getLogger(__name__)


@dataclass
class Data(BaseEntity):

    name: str
    id: str = field(default="")
    parents: list[str] = field(default_factory=list)
    validation: bool = field(default=False)

    def set_parents(self, parents: list["Data"]) -> None:
        self.parents = [parent.id for parent in parents]

    def serialize(self) -> dict:
        serialized = super().serialize()
        if "parents" in serialized:
            if len(serialized["parents"]) == 0:
                serialized.pop("parents")
            elif len(serialized["parents"]) == 1:
                serialized["parents"] = serialized["parents"][0]
        return serialized

    def evaluate(self, gold) -> float:
        raise NotImplementedError("Eval method not implemented for base Data class.")

    def _generate_id(self) -> str:
        # Generate a unique ID based on the name and other attributes
        return f"{self.name}-{uuid.uuid4()}"

    def __post_init__(self) -> None:
        super().__init__(self.name)  # Initialize BaseEntity
        if not self.id:
            self.id = self._generate_id()

    def attributes_to_ignore(self) -> set:
        attributes = super().attributes_to_ignore()
        attributes.add("name")
        attributes.add("validation")
        return attributes

    @classmethod
    def all_attributes(cls):
        return set(
            [
                attribute
                for attribute in dir(cls)
                if not attribute.startswith("_") and not callable(attribute)
            ]
        )


@dataclass
class DataNode(Node):

    items: dict[str, Data] = field(default_factory=dict)
    types: set[type] = field(default_factory=set)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.status = NodeStatus.READY if self.items else NodeStatus.NOT_READY

    def add_item(self, item: Data) -> None:
        self.items[item.id] = item
        self.status = NodeStatus.READY

    def add_type(self, type_: type) -> None:
        self.types.add(type_)

    def check_type(self, type_: type) -> bool:
        for t in self.types:
            if issubclass(type_, t):
                return True
        return False

    def __str__(self) -> str:
        types_str = "\n".join([t.get_class_path() for t in self.types])
        return f"{self.name}\n{types_str}"

    def evaluate(
        self, gold_items: "DataNode", threshold: float = 0.5
    ) -> dict[str, int | float]:
        """
        Evaluate predicted items against gold standard items.

        This implements the Hungarian algorithm (optimal bipartite matching) that:
        1. Validates type compatibility between predicted and gold items
        2. Computes pairwise scores between all predicted and gold items
        3. Finds optimal assignment that maximizes total matching score
        4. Filters matches below threshold
        5. Computes aggregate metrics (precision, recall, F1, mean score)

        Args:
            gold_items: DataNode containing ground truth items
            threshold: Minimum score for a valid match (default 0.5)
                    - For ImageCrop: IoU threshold
                    - For Text: similarity threshold

        Returns:
            dict: {
                "precision": float,      # TP / (TP + FP)
                "recall": float,         # TP / (TP + FN)
                "f1": float,             # 2 * (P * R) / (P + R)
                "mean_score": float,     # Average score of matched pairs
                "num_predicted": int,    # Total predicted items
                "num_gold": int,         # Total gold items
                "num_matched": int,      # Number of successful matches (TP)
                "num_false_positives": int,  # Predicted items not matched
                "num_false_negatives": int,  # Gold items not matched
                "matches": [            # List of matched pairs
                    {
                        "predicted_id": str,
                        "gold_id": str,
                        "score": float
                    },
                    ...
                ],
                "unmatched_predicted": [str],  # IDs of unmatched predictions
                "unmatched_gold": [str]        # IDs of unmatched gold items
            }

        Raises:
            TypeError: If predicted and gold items are of incompatible types
        """

        if threshold > 1.0 or threshold < 0.0:
            raise ValueError(f"Invalid threshold provided: {threshold}")

        metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_score": 0.0,
            "num_predicted": 0,
            "num_gold": 0,
            "num_matched": 0,
            "num_false_positives": 0,
            "num_false_negatives": 0,
            "matches": [],
            "unmatched_predicted": [],
            "unmatched_gold": [],
        }

        # Edge case: Empty inputs with explicit semantic definitions
        if len(self.items) == 0 and len(gold_items.items) == 0:
            # Vacuous truth: nothing to predict, nothing to find
            metrics.update({"precision": 1.0, "recall": 1.0, "f1": 1.0})
            return metrics

        if len(self.items) == 0:
            # No predictions: precision undefined (0/0), defaults to 0.0 for consistency
            metrics.update(
                {
                    "precision": 0.0,
                    "num_gold": len(gold_items.items),
                    "num_false_negatives": len(gold_items.items),
                    "unmatched_gold": list(gold_items.items.keys()),
                }
            )
            return metrics

        if len(gold_items.items) == 0:
            # No gold items: zero precision (all FPs), perfect recall (nothing to miss)
            metrics.update(
                {
                    "recall": 1.0,
                    "num_predicted": len(self.items),
                    "num_false_positives": len(self.items),
                    "unmatched_predicted": list(self.items.keys()),
                }
            )
            return metrics

        # Step 0: Validate type compatibility upfront
        predicted_types = set([type(value).__name__ for value in self.items.values()])
        gold_types = set([type(value).__name__ for value in gold_items.items.values()])
        compatible = predicted_types == gold_types
        if not compatible:
            raise TypeError(
                f"Cannot evaluate {predicted_types} predictions against "
                f"{gold_types} gold items. Types must match."
            )

        # Step 1: Build score matrix
        # Shape: (num_predicted, num_gold)
        predicted_ids = list(self.items.keys())
        gold_ids = list(gold_items.items.keys())
        n_pred = len(self.items.keys())
        n_true = len(gold_items.items.keys())
        score_matrix = np.zeros((n_pred, n_true))

        if n_true > n_pred:
            padding = np.full([n_true - n_pred, n_true], 1e-9)
            score_matrix = np.vstack([score_matrix, padding])
        elif n_true < n_pred:
            padding = np.full([n_pred, n_pred - n_true], 1e-9)
            score_matrix = np.hstack([score_matrix, padding])

        for pred_idx, pred_id in enumerate(predicted_ids):
            pred_item = self.items[pred_id]
            for gold_idx, gold_id in enumerate(gold_ids):
                try:
                    gold_item = gold_items.items[gold_id]
                    score = pred_item.evaluate(gold_item)
                    score_matrix[pred_idx, gold_idx] = score
                except NotImplementedError as e:
                    logger.info(f"Skipping evaluation for {pred_id} vs {gold_id}: {e}")

        # Step 2: Hungarian algorithm for optimal bipartite matching
        # Convert to cost matrix (minimize cost = maximize score)
        # scipy.optimize.linear_sum_assignment minimizes total cost
        # If cost_matrix has a shape of 1, skip this step
        if score_matrix.size != 1:
            from scipy.optimize import linear_sum_assignment

            cost_matrix = 1.0 - score_matrix

            # Note: When multiple predicted items have identical scores against the same
            # gold item, the assignment is deterministic but arbitrary (based on internal
            # algorithm traversal order). All such assignments are equally optimal.
            try:
                pred_indices, gold_indices = linear_sum_assignment(cost_matrix)
            except Exception as e:
                breakpoint()
        else:
            pred_indices = np.array([0])
            gold_indices = np.array([0])

        # Step 3: Filter matches by threshold and record results
        matches = []
        matched_predicted = set()
        matched_gold = set()

        for pred_idx, gold_idx in zip(pred_indices, gold_indices):
            if pred_idx >= n_pred or gold_idx >= n_true:
                continue
            score = score_matrix[pred_idx, gold_idx]
            # Only accept matches above the threshold
            if score >= threshold:
                predicted_id = predicted_ids[pred_idx]
                gold_id = gold_ids[gold_idx]
                matches.append(
                    {
                        "predicted_id": predicted_id,
                        "gold_id": gold_id,
                        "score": float(score),
                    }
                )
                matched_predicted.add(predicted_id)
                matched_gold.add(gold_id)

        # Step 4: Identify unmatched items
        unmatched_predicted = [
            predicted_id
            for predicted_id in predicted_ids
            if predicted_id not in matched_predicted
        ]

        unmatched_gold = [
            gold_id for gold_id in gold_ids if gold_id not in matched_gold
        ]

        # Step 5: Compute metrics
        num_true_positives = len(matches)
        num_false_positives = len(unmatched_predicted)
        num_false_negatives = len(unmatched_gold)

        precision = (
            num_true_positives / (num_true_positives + num_false_positives)
            if (num_true_positives + num_false_positives) > 0
            else 0.0
        )

        recall = (
            num_true_positives / (num_true_positives + num_false_negatives)
            if (num_true_positives + num_false_negatives) > 0
            else 0.0
        )

        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        mean_score = (
            sum(match["score"] for match in matches) / len(matches) if matches else 0.0
        )

        # Step 6: Return comprehensive evaluation results
        metrics.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_score": mean_score,
                "num_predicted": len(self.items),
                "num_gold": len(gold_items.items),
                "num_matched": num_true_positives,
                "num_false_positives": num_false_positives,
                "num_false_negatives": num_false_negatives,
                "matches": matches,
                "unmatched_predicted": unmatched_predicted,
                "unmatched_gold": unmatched_gold,
            }
        )
        return metrics

    def _create_instance(self, cls_: type, **item) -> None:
        try:
            instance = cls_(**item)
            if not instance:
                raise ValueError(f"Failed to load {item} with {cls_}")
            self.add_item(instance)
        except Exception as e:
            self.catcher.set_err(str(e))

    def _create_instances(self, **item) -> None:
        for cls_ in self.types:
            self._create_instance(cls_, **item)

    def process(
        self,
        key: str,
        value: str | Path | None = None,
        skip_type_check: bool = False,
        **kwargs,
    ) -> None:
        value = value if value else kwargs.get("file", None)
        if value and Path(value).exists() and Path(value).is_dir():
            value = Path(value)
            with change_dir(value):
                try:
                    result_file = Path("results.yaml")
                    with open(result_file, "r") as f:
                        results: dict = yaml.safe_load(f)
                        validation = results.pop(
                            "collectra_results_metadata", dict()
                        ).get("validation", None)
                        data = results.get(key, None)
                        if not data:
                            raise ValueError(
                                f"[red]{key}[/red] could not be found in {value}"
                            )
                        data = data if isinstance(data, list) else [data]
                        for item in data:
                            primitive_type = False
                            try:
                                primitive_type = not isinstance(item, dict) or not (
                                    "type" in item
                                    and ("path" in item or "data" in item)
                                )
                                if primitive_type:
                                    raise ValueError(
                                        f"Item must be a complex type with 'type' and ('path' or 'data')"
                                    )
                                cls_ = load_class_from_string(item.pop("type"))
                                if not skip_type_check and not self.check_type(cls_):
                                    raise TypeError(
                                        f"{cls_} is not a subclass or not defined in {self.types}"
                                    )
                                item["name"] = key
                                if "data" not in item:
                                    item["data"] = item.pop("path")
                                if "parents" in item and not isinstance(
                                    item["parents"], list
                                ):
                                    item["parents"] = [item["parents"]]
                                if (
                                    validation is not None
                                    and "validation" in cls_.all_attributes()
                                ):
                                    item["validation"] = validation
                                self._create_instance(cls_, **item)
                            except Exception as e:
                                if not primitive_type:
                                    self.catcher.set_err(str(e))
                                else:
                                    self._create_instances(name=key, data=str(item))

                except Exception as e:
                    self.catcher.set_err(str(e))
        elif key:
            self._create_instances(name=key, data=value)

    @staticmethod
    def batch_process(item_file: Path, data_nodes: list["DataNode"]) -> list[Data]:
        with change_dir(item_file):
            try:
                data: list[Data] = list()
                result_file = Path("results.yaml")
                if not result_file.exists():
                    print(error_msg(f"Invalid file: {Path.cwd()}. Ignoring..."))
                    return data
                with open(result_file, "r") as f:
                    file_data: dict = yaml.safe_load(f)
                    validation = file_data.get(
                        "collectra_results_metadata", dict()
                    ).get("validation", None)
                names = [data_node.name for data_node in data_nodes]
                all_names_not_found = all(name not in file_data for name in names)
                if all_names_not_found:
                    print(
                        f"[yellow]No matching data found in [blue]{item_file}[/blue] for names: {', '.join(names)}. Ignoring..."
                    )
                    found_base = False
                    for name, values in file_data.items():
                        values = values if isinstance(values, list) else [values]
                        for item in values:
                            if (
                                not isinstance(item, dict)
                                or "type" not in item
                                and ("data" not in item or "path" not in item)
                            ):
                                continue
                            cls_str = item.pop("type", "")
                            if cls_str == "collectra.Image":
                                cls_ = load_class_from_string(cls_str)
                                try:
                                    item["name"] = name
                                    item["data"] = (
                                        item.pop("path")
                                        if "path" in item
                                        else item["data"]
                                    )
                                    if validation is not None:
                                        item["validation"] = validation
                                    instance = cls_(**item)
                                    if not instance:
                                        raise Warning(
                                            f"Failed to load {item} with {cls_}"
                                        )
                                    found_base = True
                                    data.append(instance)
                                    break
                                except Exception as e:
                                    traceback_error(
                                        e,
                                        f"Failed to load data item {name} from {item.get('data', '')}",
                                        verbose=True,
                                    )
                        if found_base:
                            break
                for i, name in enumerate(names):
                    if name not in file_data:
                        continue
                    value = file_data[name]
                    if not value:
                        raise Warning(
                            f"Data seems to be empty for {name} in {item_file}. Provided: {value}"
                        )
                    value = value if isinstance(value, list) else [value]
                    for item in value:
                        if not isinstance(item, dict) or not (
                            "type" in item and ("data" in item or "path" in item)
                        ):
                            continue
                        cls_ = load_class_from_string(item.pop("type"))
                        match = False
                        for type_ in data_nodes[i].types:
                            if issubclass(cls_, type_) or cls_ == type_:
                                match = True
                                break
                        if not match:
                            continue
                        item["name"] = name
                        item["data"] = (
                            item.pop("path") if "path" in item else item["data"]
                        )
                        if validation is not None:
                            item["validation"] = validation
                        try:
                            instance = cls_(**item)
                            if instance:
                                data.append(instance)
                        except Exception as e:
                            traceback_error(
                                e,
                                f"Failed to load data item {name} from {item.get('data', '')}: {e}",
                            )
                return data
            except Exception as e:
                traceback_error(e, verbose=True)
                return list()
