def register_metric_sections(self):
    @self.section_registry.register("primary_metric_section")
    def primary_metric():
        metric_map = {
            "log_loss": "  - Primary metric: log loss, for calibrated probability estimates.",
            "recall": "  - Primary metric: recall, to catch students in need.",
            "precision": "  - Primary metric: precision, for fewer false positives.",
            "roc_auc": "  - Primary metric: ROC AUC, to measure classification strength.",
            "f1": "  - Primary metric: F1-score, balancing precision and recall.",
        }
        metric = self.cfg.modeling.training.primary_metric
        return metric_map.get(metric, "  - Default metric explanation.")

    @self.section_registry.register("sample_weight_section")
    def sample_weight():
        has_weights = any(col.startswith("sample_weight") for col in self.training_data.columns)
        note = None if has_weights else "  - Sample weights were used to stabilize training."
        mlops = (
            f"  - Used Databricks AutoML with {self.context['num_runs_in_experiment']} experiments."
        )
        return "\n".join(filter(None, [mlops, note]))