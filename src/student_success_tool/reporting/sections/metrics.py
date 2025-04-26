def register_metric_sections(card, registry):
    @registry.register("primary_metric_section")
    def primary_metric():
        metric_map = {
            "log_loss": "  - Our primary metric for training was log loss to ensure that the model produces well-calibrated probability estimates.\n  - Lower log loss is better, as it indicates more accurate and confident probability predictions.",
            "recall": "  - Our primary metric for training was recall in order to ensure that we correctly identify as many students in need of support as possible.\n  - Higher recall is better, as it indicates fewer students in need are missed.",
            "precision": "  - Our primary metric for training was precision to ensure that when the model identifies a student as needing support, it is likely to be correct.\n  - Higher precision is better, as it indicates fewer students are incorrectly flagged.",
            "roc_auc": "  - Our primary metric for training was ROC AUC, which measures the model's ability to distinguish between students who need support and those who do not.\n  - Higher ROC AUC is better, as it indicates stronger overall classification performance across all thresholds.",
            "f1": "  - Our primary metric for training was F1-score to balance the trade-off between precision and recall.\n  - A higher F1-score indicates that the model is effectively identifying students in need while minimizing both false positives and false negatives.",
        }
        metric = card.cfg.modeling.training.primary_metric
        return metric_map.get(metric, "  - Default metric explanation.")

    @registry.register("sample_weight_section")
    def sample_weight():
        used_weights = any(col.startswith("sample_weight") for col in card.training_data.columns)
        sw_note = None if used_weights else "  - Sample weights were used to stabilize training."
        mlops_note = (
            f"  - Used Databricks AutoML with {card.context['num_runs_in_experiment']} experiments."
        )
        return "\n".join(filter(None, [mlops_note, sw_note]))