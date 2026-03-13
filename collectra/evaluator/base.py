__all__ = ["Evaluator"]

from dataclasses import dataclass
from pathlib import Path

import yaml
from rich.table import Table

from ..logger import get_logger
from ..types import DataNode
from ..utils import change_dir

logger = get_logger(__name__)


@dataclass
class FileEvaluationResult:
    """Results for a single file pair evaluation."""

    filename: str
    label_metrics: dict[str, dict]


@dataclass
class EvaluationReport:
    """Complete evaluation report across all files."""

    file_results: list[FileEvaluationResult]
    aggregate: dict
    threshold: float

    def __post_init__(self):
        self.aggregate_tables = []
        self.tables = []
        for data_type, metrics in self.aggregate.items():
            """Turn the aggregate dict into a rich Table for display."""
            table = Table(title=f"Aggregate Evaluation Metrics for {data_type}")
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", justify="right", style="green")

            # Add overall metrics
            overall = metrics.get("overall", {})
            table.add_row("Precision (Overall)", f"{overall.get('precision', 0):.3f}")
            table.add_row("Recall (Overall)", f"{overall.get('recall', 0):.3f}")
            table.add_row("F1 Micro", f"{overall.get('f1_micro', 0):.3f}")
            table.add_row(
                "F1 File-Averaged", f"{overall.get('f1_file_averaged', 0):.3f}"
            )
            table.add_row("Total Files", str(overall.get("total_files", 0)))
            self.aggregate_tables.append(table)
            # Create per-label table
            per_label = metrics.get("per_label", {})
            if not per_label:
                continue
            table = Table(title=f"Per-Label Metrics for Data Type: {data_type}")
            table.add_column("Label", style="magenta")
            table.add_column("Precision", justify="right", style="green")
            table.add_column("Recall", justify="right", style="green")
            table.add_column("F1", justify="right", style="green")
            table.add_column("Mean Score", justify="right", style="green")
            table.add_column("Support", justify="right", style="blue")
            for label_name, metrics in per_label.items():
                table.add_row(
                    label_name,
                    f"{metrics.get('precision', 0):.3f}",
                    f"{metrics.get('recall', 0):.3f}",
                    f"{metrics.get('f1', 0):.3f}",
                    f"{metrics.get('mean_score', 0):.3f}",
                    str(metrics.get("support", 0)),
                )
            self.tables.append(table)


class EvaluationError(Exception):
    """Raised when evaluation cannot proceed (e.g., missing gold files)."""

    pass


class Evaluator:
    """
    End-to-end evaluation orchestrator for Collectra workflows.

    Responsibilities:
    1. Match input files to gold standard files by filename
    2. Load and parse .grapto (or other ext) folder structures
    3. Delegate per-label evaluation to DataNode.evaluate()
    4. Aggregate metrics across files and labels
    5. Export results in tabular format
    """

    def __init__(self, predicted_folder: Path, gold_folder: Path, ext: str) -> None:
        self.ext = ext.lower().replace(".", "")
        self.predicted_files: dict[str, Path] = self._discover_collectra_folders(
            predicted_folder
        )
        self.gold_files: dict[str, Path] = self._discover_collectra_folders(gold_folder)
        self.results: list[FileEvaluationResult] = []
        self.aggregate_metrics: dict = {}

    def __call__(self) -> None:
        self.evaluate()

    def _discover_collectra_folders(self, folder: Path) -> dict[str, Path]:
        """
        Find all .{ext} directories (e.g., .grapto folders).

        Returns:
            dict mapping base filename -> full path
            e.g., {"MMRIRN1505070_P350015": Path(".../MMRIRN1505070_P350015.grapto")}
        """
        result = {}
        for item in folder.iterdir():
            if item.is_dir() and item.suffix == f".{self.ext}":
                # Extract base name without extension for matching
                base_name = item.stem
                result[base_name] = item
        return result

    def _validate_file_pairs(
        self,
    ) -> tuple[list[tuple[Path, Path, str]], list[str], list[str]]:
        """
        Match input files to gold files and identify mismatches.

        Returns:
            - matched_pairs: list of (input_path, gold_path, base_name)
            - missing_gold: input files without corresponding gold
            - extra_gold: gold files without corresponding input
        """
        input_names = set(self.predicted_files.keys())
        gold_names = set(self.gold_files.keys())

        matched_names = input_names & gold_names
        missing_gold = input_names - gold_names
        extra_gold = gold_names - input_names

        matched_pairs = [
            (self.predicted_files[name], self.gold_files[name], name)
            for name in matched_names
        ]

        return matched_pairs, list(missing_gold), list(extra_gold)

    def _load_collectra_file(self, path: Path) -> dict[str, DataNode]:
        """
        Parse a collectra file data into DataNodes organized by label.

        Each label (e.g., "registration_number", "local_text") becomes
        a separate DataNode containing its items (ImageCrops, Texts).

        Returns:
            dict mapping label_name -> DataNode
        """
        labels = dict()
        with change_dir(path):
            with open("results.yaml", "r") as f:
                data = yaml.safe_load(f)
                data.pop("collectra_results_metadata", "")
                node_names = list(data.keys())
        for name in node_names:
            node = DataNode(name=name)
            node.process(key=node.name, value=path, skip_type_check=True)
            labels[name] = node
        return labels

    def _create_missing_prediction_metrics(self, gold_node: DataNode) -> dict:
        """
        Generate metrics when predictions are entirely missing for a gold label.

        Scenario: Gold has items for this label, but predictions have none.
        Result:
            - Precision = 0.0 (undefined 0/0, defaults to 0.0 for consistency)
            - Recall = 0.0 (missed everything)
            - F1 = 0.0
            - All gold items are false negatives

        This mirrors the edge case handling in DataNode.evaluate()
        when self.items is empty but gold_items is not.
        """
        gold_ids = list(gold_node.items.keys())

        return {
            "types": list(
                set([type(item).__name__ for item in gold_node.items.values()])
            ),
            "precision": 0.0,  # No predictions = undefined (0/0), defaults to 0.0
            "recall": 0.0,  # Missed all gold items
            "f1": 0.0,
            "mean_score": 0.0,
            "num_predicted": 0,
            "num_gold": len(gold_ids),
            "num_matched": 0,
            "num_false_positives": 0,
            "num_false_negatives": len(gold_ids),
            "matches": [],
            "unmatched_predicted": [],
            "unmatched_gold": gold_ids,
        }

    def _create_extra_prediction_metrics(self, input_node: DataNode) -> dict:
        """
        Generate metrics when predictions exist for a label not in gold.

        Scenario: Predictions have items for this label, but gold has none.
        Result:
            - Precision = 0.0 (all predictions are false positives)
            - Recall = 1.0 (nothing to miss, vacuously true)
            - F1 = 0.0
            - All predicted items are false positives

        This mirrors the edge case handling in DataNode.evaluate()
        when gold_items.items is empty but self.items is not.

        Note: This typically indicates either:
            1. Model hallucinated a label class
            2. Mismatch in label naming conventions
            3. Gold annotation is incomplete
        Consider logging a warning when this occurs.
        """
        predicted_ids = list(input_node.items.keys())

        return {
            "types": list(
                set([type(item).__name__ for item in input_node.items.values()])
            ),
            "precision": 0.0,  # All predictions are wrong
            "recall": 1.0,  # Nothing to find = found everything
            "f1": 0.0,
            "mean_score": 0.0,
            "num_predicted": len(predicted_ids),
            "num_gold": 0,
            "num_matched": 0,
            "num_false_positives": len(predicted_ids),
            "num_false_negatives": 0,
            "matches": [],
            "unmatched_predicted": predicted_ids,
            "unmatched_gold": [],
        }

    def _evaluate_file_pair(
        self, input_path: Path, gold_path: Path, threshold: float = 0.5
    ) -> FileEvaluationResult:
        """
        Evaluate a single input file against its gold standard.

        Process:
        1. Load both files into label -> DataNode mappings
        2. For each label present in gold, evaluate predictions
        3. Handle labels present in only one file (missing predictions or extra labels)

        Returns:
            FileEvaluationResult containing per-label metrics
        """
        input_labels = self._load_collectra_file(input_path)
        gold_labels = self._load_collectra_file(gold_path)

        file_result = FileEvaluationResult(filename=input_path.stem, label_metrics={})

        all_labels = set(input_labels.keys()) | set(gold_labels.keys())

        for label_name in all_labels:
            input_node = input_labels.get(label_name)
            gold_node = gold_labels.get(label_name)

            if input_node is None and gold_node is None:
                # This should not happen since label is in all_labels
                continue
            elif input_node is None:
                # Missing predictions for this label
                metrics = self._create_missing_prediction_metrics(gold_node)
            elif gold_node is None:
                # Extra predictions for non-existent gold label
                metrics = self._create_extra_prediction_metrics(input_node)
            else:
                # Both exist - use DataNode.evaluate()
                metrics = input_node.evaluate(gold_node, threshold=threshold)

            file_result.label_metrics[label_name] = metrics

        return file_result

    def _aggregate_results(self) -> dict:
        """
        Compute aggregate metrics across all files.

        Aggregation strategy:
        - Macro average: Average of per-file metrics (treats each file equally)
        - Micro average: Pool all TP/FP/FN then compute (treats each item equally)
        - Per-label aggregates: Metrics grouped by label class
        """
        from collections import defaultdict

        metrics = ["precision", "recall", "f1", "mean_score"]
        data_types_set = set()
        for result in self.results:
            for label_name, label_metrics in result.label_metrics.items():
                data_types = label_metrics.get("types", [])
                for data_type in data_types:
                    data_types_set.add(data_type)

        data_types = list(data_types_set)

        aggregated_results = dict()

        for data_type in data_types:
            # Initialize accumulators
            label_pools = defaultdict(
                lambda: {
                    "total_tp": 0,
                    "total_fp": 0,
                    "total_fn": 0,
                    "total_score": 0.0,
                    "total_matches": 0,
                }
            )

            file_f1_scores = []

            for file_result in self.results:
                file_tp = file_fp = file_fn = 0
                for label_name, metrics in file_result.label_metrics.items():
                    if data_type not in metrics.get("types", []):
                        continue
                    pool = label_pools[label_name]
                    pool["total_tp"] += metrics["num_matched"]
                    pool["total_fp"] += metrics["num_false_positives"]
                    pool["total_fn"] += metrics["num_false_negatives"]
                    pool["total_score"] += (
                        metrics["mean_score"] * metrics["num_matched"]
                    )
                    pool["total_matches"] += metrics["num_matched"]
                    pool["types"] = metrics.get("types", [])

                    file_tp += metrics["num_matched"]
                    file_fp += metrics["num_false_positives"]
                    file_fn += metrics["num_false_negatives"]

                # Per-file F1
                file_precision = (
                    file_tp / (file_tp + file_fp) if (file_tp + file_fp) > 0 else 0
                )
                file_recall = (
                    file_tp / (file_tp + file_fn) if (file_tp + file_fn) > 0 else 0
                )
                file_f1 = (
                    2 * file_precision * file_recall / (file_precision + file_recall)
                    if (file_precision + file_recall) > 0
                    else 0
                )
                file_f1_scores.append(file_f1)

            # Compute per-label micro metrics
            per_label_metrics = {}
            for label_name, pool in label_pools.items():
                tp, fp, fn = pool["total_tp"], pool["total_fp"], pool["total_fn"]
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )
                mean_score = (
                    pool["total_score"] / pool["total_matches"]
                    if pool["total_matches"] > 0
                    else 0
                )

                per_label_metrics[label_name] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mean_score": mean_score,
                    "support": pool["total_tp"]
                    + pool["total_fn"],  # Number of gold items
                }

            # Overall metrics
            total_tp = sum(p["total_tp"] for p in label_pools.values())
            total_fp = sum(p["total_fp"] for p in label_pools.values())
            total_fn = sum(p["total_fn"] for p in label_pools.values())

            overall_precision = (
                total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            )
            overall_recall = (
                total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            )
            overall_f1_micro = (
                2
                * overall_precision
                * overall_recall
                / (overall_precision + overall_recall)
                if (overall_precision + overall_recall) > 0
                else 0
            )
            overall_f1_file_averaged = (
                sum(file_f1_scores) / len(file_f1_scores) if file_f1_scores else 0
            )
            aggregated_results[data_type] = {
                "overall": {
                    "precision": overall_precision,
                    "recall": overall_recall,
                    "f1_micro": overall_f1_micro,
                    "f1_file_averaged": overall_f1_file_averaged,
                    "total_files": len(self.results),
                },
                "per_label": per_label_metrics,
            }

        return aggregated_results

    def evaluate(self, threshold: float = 0.5) -> EvaluationReport:
        """
        Execute full evaluation across all matched file pairs.

        Raises:
            EvaluationError: If any input file lacks a corresponding gold file
        """
        matched_pairs, missing_gold, extra_gold = self._validate_file_pairs()

        # Fail fast if gold files are missing
        if missing_gold:
            raise EvaluationError(f"Missing gold standard files for: {missing_gold}")

        # Warn about extra gold files (not an error)
        if extra_gold:
            logger.warning(f"Gold files without predictions: {extra_gold}")

        # Evaluate each pair
        for input_path, gold_path, base_name in matched_pairs:
            file_result = self._evaluate_file_pair(input_path, gold_path, threshold)
            self.results.append(file_result)

        # Aggregate across all files
        self.aggregate_metrics = self._aggregate_results()

        return EvaluationReport(
            file_results=self.results,
            aggregate=self.aggregate_metrics,
            threshold=threshold,
        )
