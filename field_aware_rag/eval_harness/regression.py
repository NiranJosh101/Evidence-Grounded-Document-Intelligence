import json
import os
from field_aware_rag.eval_harness.eval_harness import EvaluationMetrics


class RegressionDetector:
    """Compares current evaluation run metrics against historical baselines."""

    def __init__(self, baseline_file: str = "field_aware_rag/data/eval_baseline.json"):
        self.baseline_file = baseline_file

    def check_for_regression(self, current_metrics: EvaluationMetrics, tolerance: float = 0.02) -> dict:
        if not os.path.exists(self.baseline_file):
            # Save first run as baseline
            self.save_baseline(current_metrics)
            return {"status": "BASELINE_CREATED", "message": "First eval run saved as baseline."}

        with open(self.baseline_file, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)

        baseline_overall = baseline_data.get("overall_score", 0.0)
        diff = current_metrics.overall_score - baseline_overall

        if diff < -tolerance:
            return {
                "status": "REGRESSION_DETECTED",
                "diff": round(diff, 3),
                "current_score": current_metrics.overall_score,
                "baseline_score": baseline_overall,
                "passed": False
            }

        # Update baseline if overall metrics improved
        if diff > 0:
            self.save_baseline(current_metrics)

        return {
            "status": "PASS",
            "diff": round(diff, 3),
            "current_score": current_metrics.overall_score,
            "baseline_score": baseline_overall,
            "passed": True
        }

    def save_baseline(self, metrics: EvaluationMetrics) -> None:
        # Create directory recursively if it doesn't exist
        os.makedirs(os.path.dirname(self.baseline_file), exist_ok=True)
        
        with open(self.baseline_file, "w", encoding="utf-8") as f:
            json.dump(metrics.model_dump(), f, indent=2)