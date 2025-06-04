import logging

LOGGER = logging.getLogger(__name__)


def register_attribute_sections(card, registry):
    """
    Registers all attributes or characteristics of a PDP model such as its outcome,
    checkpoint, and target population. All of this information is gathered from the model's
    config.toml file.
    """

    @registry.register("outcome_section")
    def outcome():
        """
        Produce section for outcome variable based on config target definition. PDP has
        validation around the data schema and the config file, which enables us to assume
        that fields will be present.
        """
        outcome_type = card.cfg.preprocessing.target.type_

        if outcome_type == "retention":
            outcome = "non-retention into the student's second academic year"
            description = f"The model predicts the risk of {outcome} based on student, course, and academic data."
        else:
            limits = card.cfg.preprocessing.selection.intensity_time_limits

            if outcome_type == "graduation":
                outcome = "not graduating on time"
            elif outcome_type == "credits_earned":
                credit_thresh = card.cfg.preprocessing.target.min_num_credits
                outcome = f"not earning {credit_thresh} credits"

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
            description = f"The model predicts the risk of {outcome} within {full_str} for full-time students"

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

    @registry.register("checkpoint_section")
    def checkpoint():
        """
        Produce a section for the PDP checkpoint. Also defines an
        ordinal function
        """
        checkpoint_type = card.cfg.preprocessing.checkpoint.type_
        base_message = "The model makes this prediction when the student has"
        n_ckpt = str(card.cfg.preprocessing.checkpoint.n) 

        if checkpoint_type == "all":
            return f"{base_message} completed their {card.format.ordinal(n_ckpt)} term."
        elif checkpoint_type == "num_credits_earned":
            credit_thresh = card.cfg.preprocessing.checkpoint.min_num_credits
            return f"{base_message} earned {credit_thresh} credits."
        elif checkpoint_type == "within_cohort":
            return f"{base_message} completed their {card.format.ordinal(n_ckpt)} term within their cohort."
        elif checkpoint_type == "enrollment_year":
            enrl_year = card.cfg.preprocessing.checkpoint.enrollment_year
            return f"{base_message} completed their {card.format.ordinal(enrl_year)} year of enrollment."
        else:
            raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")
