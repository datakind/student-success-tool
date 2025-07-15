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
                timeframe_phrase = f"in achieving {unit_str} required for graduation"
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

    @registry.register("target_population_section")
    def target_population():
        """
        Produce a section for the target population.  This
        will need to later be refined to support both PDP and custom institutions well.
        """
        try:
            criteria = card.cfg.preprocessing.selection.student_criteria_aliases

            if not criteria:
                LOGGER.info("No student criteria provided in config.")
                return "No specific student criteria were applied."

            if not isinstance(criteria, dict):
                LOGGER.warning(
                    f"Expected 'student_criteria_aliases' to be a dict but got {type(criteria).__name__}."
                )
                return "Student criteria should be provided as a dictionary."

            lines = []
            for k, v in criteria.items():
                lines.append(
                    f"{card.format.indent_level(2)}- {card.format.friendly_case(k)}"
                )

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

        except Exception as e:
            LOGGER.error("Unable to retrieve student criteria configuration in config", exc_info=True)
            return f"{card.format.indent_level(1)}- Student criteria configuration was unavailable."


    @registry.register("checkpoint_section")
    def checkpoint():
        """
        Produce a section for the PDP checkpoint. Also defines an
        ordinal function
        """
        checkpoint_type = card.cfg.preprocessing.checkpoint.type_
        base_message = "The model makes this prediction when the student has"
        if checkpoint_type == "nth":
            n_ckpt = card.cfg.preprocessing.checkpoint.n
            exclude_pre_cohort_terms = (
                card.cfg.preprocessing.checkpoint.exclude_pre_cohort_terms
            )
            exclude_non_core_terms = (
                card.cfg.preprocessing.checkpoint.exclude_non_core_terms
            )
            valid_enrollment_year = (
                card.cfg.preprocessing.checkpoint.valid_enrollment_year
            )
            if n_ckpt >= 0:
                message = f"{base_message} completed their {card.format.ordinal(n_ckpt + 1)} term"
            elif n_ckpt == -1:
                message = f"{base_message} completed their last term"
            else:
                raise ValueError(
                    f"Unable to interpret value for nth checkpoint: {n_ckpt}"
                )

            included = []
            if not exclude_pre_cohort_terms:
                included.append("pre-cohort terms")
            if not exclude_non_core_terms:
                included.append("non-core terms")
            if included:
                if len(included) == 1:
                    message += f" including {included[0]}"
                else:
                    message += (
                        f" including {', '.join(included[:-1])} and {included[-1]}"
                    )
            if valid_enrollment_year:
                message += f", provided the term occurred in their {card.format.ordinal(valid_enrollment_year)} year of enrollment"
            message = message.rstrip(". ") + "."
            return message
        elif checkpoint_type == "first":
            return f"{base_message} completed their first term."
        elif checkpoint_type == "last":
            return f"{base_message} completed their last term."
        elif checkpoint_type == "first_at_num_credits_earned":
            credit_thresh = card.cfg.preprocessing.checkpoint.min_num_credits
            return f"{base_message} earned {credit_thresh} credits."
        elif checkpoint_type == "first_within_cohort":
            return f"{base_message} completed their first term within their cohort."
        elif checkpoint_type == "last_in_enrollment_year":
            enrl_year = card.cfg.preprocessing.checkpoint.enrollment_year
            return f"{base_message} completed their {card.format.ordinal(enrl_year)} year of enrollment."
        else:
            raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")
