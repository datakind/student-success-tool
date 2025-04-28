def register_table_sections(card, registry):
    @registry.register("data_split_table")
    def data_split():
        total_students = len(card.modeling_data)
        splits = card.cfg.preprocessing.splits
        labels = {"train": "Training", "validate": "Validation", "test": "Test"}

        rows = [
            f"| {labels[k]:<10} | {round(total_students * splits[k]):<8} | {int(splits[k] * 100)}%       |"
            for k in ["train", "validate", "test"]
        ]

        return "\n".join([
            "| Split      | Students | Percentage |",
            "|------------|----------|------------|",
            *rows
        ])
    