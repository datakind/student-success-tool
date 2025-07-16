import logging
from datetime import datetime

LOGGER = logging.getLogger(__name__)


def register_attribute_sections(card, registry):
    """
    Registers all attributes or characteristics of a custom model such as its outcome,
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

    @registry.register("outcome_section")
    def outcome():
        """
        Produce section for outcome variable based on config target definition. Custom schools have
        validation around the config file, which enables us to assume
        that fields will be present.
        """
        try:
            outcome_type = card.cfg.preprocessing.target.category

            if outcome_type == "retention":
                outcome = "non-retention into the student's second academic year"
                description = f"The model predicts the likelihood of {outcome} based on student, course, and academic data."
            else:
                outcome = "not graduating on time"
                unit = card.cfg.preprocessing.target.unit
                value = card.cfg.preprocessing.target.value

                if unit in {"credit", "year", "term", "semester"}:
                    unit_label = unit + ("s" if value != 1 else "")
                    unit_str = f"{value} {unit_label}"
                elif unit == "pct_completion":
                    unit_str = f"{value}% completion"
                else:
                    unit_str = f"{value} {unit}"

                # Customize phrasing based on unit
                if unit == "credit":
                    timeframe_phrase = (
                        f"in achieving {unit_str} required for graduation"
                    )
                elif unit == "year":
                    timeframe_phrase = f"within {unit_str}"
                elif unit in {"term", "semester"}:
                    timeframe_phrase = f"within {unit_str}"
                elif unit == "pct_completion":
                    timeframe_phrase = f"at {unit_str}"
                else:
                    timeframe_phrase = f"within {unit_str}"

                description = (
                    f"The model predicts the likelihood of {outcome} {timeframe_phrase}, "
                    "based on student, course, and academic data."
                )

            return description

        except (AttributeError, TypeError, KeyError) as e:
            LOGGER.warning(
                f"[outcome_section] Failed to generate outcome description: {e}"
            )
            return "Unable to retrieve model outcome information"

    @registry.register("target_population_section")
    def target_population():
        """
        Produce a section for the target population.
        """
        try:
            criteria = card.cfg.preprocessing.selection.student_criteria
            aliases = getattr(
                card.cfg.preprocessing.selection, "student_criteria_aliases", {}
            )

            if not criteria:
                LOGGER.warning("No student criteria provided in config.")
                return "No specific student criteria were applied."

            if not isinstance(criteria, dict):
                LOGGER.warning(
                    f"Expected 'student_criteria' to be a dict but got {type(criteria).__name__}."
                )
                return "Student criteria should be provided as a dictionary."

            lines = []
            for k, v in criteria.items():
                label = aliases.get(k, card.format.friendly_case(k))
                lines.append(f"{card.format.indent_level(2)}- {label}")

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

        except Exception:
            LOGGER.error(
                "Unable to retrieve student criteria configuration in config",
                exc_info=True,
            )
            return f"{card.format.indent_level(1)}- Student criteria configuration was unavailable."

    @registry.register("checkpoint_section")
    def checkpoint():
        """
        Produce a section for the custom checkpoint. Also defines an
        ordinal function.
        """
        try:
            unit = card.cfg.preprocessing.checkpoint.unit
            value = card.cfg.preprocessing.checkpoint.value
            base_message = "The model makes this prediction when the student has"
            if unit == "credit":
                return f"{base_message} earned {card.cfg.preprocessing.checkpoint.value} credits."
            elif unit in {"year", "term", "semester"}:
                unit_label = unit + ("s" if value != 1 else "")
                return f"{base_message} completed {value} {unit_label}"
        except (AttributeError, TypeError, KeyError) as e:
            LOGGER.warning(
                f"[checkpoint_section] Failed to generate checkpoint description: {e}"
            )
            return "Unable to retrieve model checkpoint information"
