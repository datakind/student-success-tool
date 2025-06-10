import logging
from datetime import datetime

LOGGER = logging.getLogger(__name__)


def register_attribute_sections(card, registry):
    """
    Registers all attributes or characteristics of a model such as its outcome,
    checkpoint, and target population. All of this information is gathered from the model's
    config.toml file.
    """

    @registry.register("development_note_section")
    def development_note():
        """
        Produce a note describing when the model was developed and listing the
        model version (if available).
        """
        version_number = card.context.get("version_number", None)
        current_year = str(datetime.now().year)
        if version_number:
            return f"Developed by DataKind in {current_year}, Model Version {version_number}"
        else:
            return f"Developed by DataKind in {current_year}"

    # HACK: This section assumes certain standards in the config
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

        full_time = normalized_limits.get("FULL-TIME")
        part_time = normalized_limits.get("PART-TIME")

        if not full_time:
            LOGGER.warning(
                "Unable to determine timeframe of outcome for students. Please specify in model card or in config.toml."
            )
            return f"{card.format.bold('Timeframe for Outcome Variable Not Found')}"

        full_str = card.format.format_intensity_time_limit(full_time)
        description = f"The model predicts the likelihood of {outcome} within {full_str} for full-time students"

        if part_time:
            part_str = card.format.format_intensity_time_limit(part_time)
            description += f", and within {part_str} for part-time students"

        description += ", based on student, course, and academic data."
        return description

    @registry.register("target_population_section")
    def target_population():
        """
        Produce a section for the target population.  This
        will need to later be refined to support both PDP and custom institutions well.

        TODO: Create an alias for column names. Custom schools will need to create their
        own alias and feed it into the ModelCard as an attribute.
        """
        criteria = card.cfg.preprocessing.selection.student_criteria

        if not criteria:
            return "No specific student criteria were applied."

        lines = []
        for k, v in criteria.items():
            lines.append(
                f"{card.format.indent_level(2)}- {card.format.friendly_case(k)}"
            )

            # Handle if value is a list or a single string
            if isinstance(v, list):
                for item in v:
                    lines.append(
                        f"{card.format.indent_level(3)}- {card.format.friendly_case(item)}"
                    )
            else:
                lines.append(
                    f"{card.format.indent_level(3)}- {card.format.friendly_case(v)}"
                )

        description = (
            f"{card.format.indent_level(1)}- We focused our final dataset on the following target population.\n"
            + "\n".join(lines)
        )
        return description

    # HACK: This section assumes certain standards in the config.
    @registry.register("checkpoint_section")
    def checkpoint():
        """
        Produce a section for the checkpoint. We assume the
        checkpoint name has semester, term, or credit information.
        """
        checkpoint_name = card.cfg.preprocessing.checkpoint.name.lower()
        params = card.cfg.preprocessing.checkpoint.params

        if "credit" in checkpoint_name:
            num_credits = params.get("min_num_credits", "X")
            return f"The model makes this prediction when the student has completed {num_credits} credits."

        elif "semester" in checkpoint_name or "term" in checkpoint_name:
            return f"The model makes this prediction when the student has completed {card.format.friendly_case(checkpoint_name, capitalize=False)}."
        else:
            LOGGER.warning(
                "Unable to determine checkpoint information. Please specify in model card or in config.toml."
            )
            return f"{card.format.bold('Checkpoint Information Not Found')}"
