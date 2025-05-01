import logging 

LOGGER = logging.getLogger(__name__)


def register_data_sections(card, registry):
    """
    Registers sections related to data split and feature tables.
    It does not contain sections relevant to performance or bias. These tables
    are gathered from a combination of config.toml and mlflow artifacts.
    """
    @registry.register("data_split_table")
    def data_split():
        """
        Produces a markdown table of the data split.
        """
        total_students = len(card.modeling_data)
        splits = card.cfg.preprocessing.splits
        if splits:
            labels = {"train": "Training", "validate": "Validation", "test": "Test"}

            rows = [
                f"| {labels[k]:<10} | {round(total_students * splits[k]):<8} | {int(splits[k] * 100)}%       |"
                for k in ["train", "validate", "test"]
            ]

            return "\n".join(
                [
                    "| Split      | Students | Percentage |",
                    "|------------|----------|------------|",
                    *rows
                ]
            )
        else:
            LOGGER.warning(
                "Unable to produce data split table. No splits found in config."
            )
            return f"{card.format.bold('Could not parse data split')}"
