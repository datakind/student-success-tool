import logging
import typing as t

LOGGER = logging.getLogger(__name__)


def register_attribute_sections(card, registry):
    """
    Registers all attributes or characteristics of a model such as its outcome,
    checkpoint, and target population. All of this information is gathered from the model's
    config.toml file.
    """

    @registry.register("outcome_section")
    def outcome():
        """
        Produce section for outcome variable based on config target definition. If target from
        config does not explicitly state "graduation", "grad", "retention", or "ret", this method
        will raise an error. The assumption is that only a graduation or retention model is in scope
        with the SST.
        """
        name = card.cfg.preprocessing.target.name
        limits = card.cfg.preprocessing.selection.intensity_time_limits

        if not name or not limits:
            LOGGER.warning(
                "Unable to determine target or time limit for outcome information. Please specify in model card or in config.toml."
            )
            return f"{card.format.bold('Target or Time Limit Information Not Found')}"

        if "graduation" in name.lower() or "grad" in name.lower():
            outcome = "graduation"
        elif "retention" in name.lower() or "ret" in name.lower():
            outcome = "retention"
        else:
            LOGGER.warning(
                "Unable to interpret target variable. Please specify in model card or in config.toml."
            )
            return f"{card.format.bold('Target Variable Not Found')}"

        # Normalize intensity labels to support flexible formats
        normalized_limits = {
            k.strip().upper().replace(" ", "-"): v for k, v in limits.items()
        }

        def format_time(duration: t.Tuple[str, str]) -> str:
            """
            We want to format a intensity_limit within config.toml by unpacking
            the value (3.0) and unit ("year"), for example.

            Args:
                duration: intensity limit in config (3.0, "year"), for example.
            """
            num, unit = duration

            # Format number cleanly
            if isinstance(num, float):
                if num.is_integer():
                    num = int(num)
                else:
                    num = round(
                        num, 2
                    )  # Keep at most 1 decimals with no trailing zeros

            unit = unit if num == 1 else unit + "s"
            return f"{num} {unit}"

        full_time = normalized_limits.get("FULL-TIME")
        part_time = normalized_limits.get("PART-TIME")

        if not full_time:
            LOGGER.warning(
                "Unable to determine timeframe of outcome for students. Please specify in model card or in config.toml."
            )
            return f"{card.format.bold('Timeframe for Outcome Variable Not Found')}"

        full_str = format_time(full_time)
        description = f"The model predicts the risk of {outcome} within {full_str} for full-time students"

        if part_time:
            part_str = format_time(part_time)
            description += f", and within {part_str} for part-time students"

        description += ", based on student, course, and academic data."
        return description

    @registry.register("target_population_section")
    def target_population():
        """
        Produce a section for the target population. This method does a rough cleaning of
        turning underscores into spaces and capitalizing the first letter of each word. This
        will need to later be refined to support both PDP and custom institutions well.

        TODO: Create an alias for column names. Custom schools will need to create their
        own alias and feed it into the ModelCard as an attribute.
        """
        criteria = card.cfg.preprocessing.selection.student_criteria

        if not criteria:
            return "No specific student criteria were applied."

        def clean_key(key):
            return key.replace("_", " ").title()

        lines = []
        for k, v in criteria.items():
            field = clean_key(k)
            lines.append(f"{card.format.indent_level(3)}- {field}")

            # Handle if value is a list or a single string
            if isinstance(v, list):
                for item in v:
                    lines.append(f"{card.format.indent_level(4)}- {item}")
            else:
                lines.append(f"{card.format.indent_level(4)}- {v}")

        description = (
            f"{card.format.indent_level(2)}- We focused our final dataset on the following target population:\n"
            + "\n".join(lines)
        )
        return description


    # TODO: Right now there are no standards in the config for the checkpoint section.
    # HACK: This section assumes certain standards in the config.
    @registry.register("checkpoint_section")
    def checkpoint():
        """
        Produce a section for the checkpoint. This method does a rough cleaning of
        turning underscores into spaces for semester or term checkpoints. We assume the
        checkpoint name has semester, term, or credit information.
        """
        checkpoint_name = card.cfg.preprocessing.checkpoint.name.lower()
        params = card.cfg.preprocessing.checkpoint.params

        if "credit" in checkpoint_name:
            num_credits = params.get("min_num_credits", "X")
            return f"The model makes this prediction when the student has completed {card.format.bold(f'{num_credits} credits')}**."

        elif "semester" in checkpoint_name or "term" in checkpoint_name:
            # Try to extract a label from the name (e.g., "first_semester" → "First semester")
            friendly_label = checkpoint_name.replace("_", " ")
            return f"The model makes this prediction when the student has completed {card.format.bold(f'{friendly_label}')}."
        else:
            LOGGER.warning(
                "Unable to determine checkpoint information. Please specify in model card or in config.toml."
            )
            return f"{card.format.bold('Checkpoint Information Not Found')}"
