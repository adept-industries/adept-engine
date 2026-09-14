# Temporary file to test realistic risk scoring evaluation
"""Module containing sample test structures for risk evaluation."""

class SampleFeatureAnalyzer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.metrics: dict[str, float] = {}

    def record(self, key: str, value: float) -> None:
        self.metrics[key] = value

    def compute_aggregate(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(self.metrics.values()) / len(self.metrics)

    def is_healthy(self) -> bool:
        return self.compute_aggregate() > 0.5
