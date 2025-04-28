def register_attribute_sections(card, registry):    
    @registry.register("outcome_section")
    def outcome():
    
        name = card.cfg.preprocessing.target.name
        limits = card.cfg.preprocessing.selection.intensity_time_limits

        if not name or not limits:
            return "  - NOTE TO DATA SCIENTIST: Cannot detect target information. Please specify in model card."

        if "graduation" in name.lower() or "grad" in name.lower():
            outcome = "graduation"
        elif "retention" in name.lower() or "ret" in name.lower():
            outcome = "retention"
        else:
            raise NameError("Unable to determine outcome variable from config.")

        # Normalize intensity labels to support flexible formats
        normalized_limits = {
            k.strip().upper().replace(" ", "-"): v for k, v in limits.items()
        }

        def format_time(duration):
            num, unit = duration

            # Format number cleanly
            if isinstance(num, float):
                if num.is_integer():
                    num = int(num)
                else:
                    num = round(num, 2)  # Keep at most 1 decimals with no trailing zeros
  
            unit = unit if num == 1 else unit + "s"
            return f"{num} {unit}"

        full_time = normalized_limits.get("FULL-TIME")
        part_time = normalized_limits.get("PART-TIME")

        if not full_time:
            raise ValueError("Full-time duration must be specified in intensity_time_limits.")

        full_str = format_time(full_time)
        description = f"The model predicts the risk of {outcome} within {full_str} for full-time students"

        if part_time:
            part_str = format_time(part_time)
            description += f", and within {part_str} for part-time students"

        description += ", based on student, course, and academic data."
        return description


    @registry.register("target_population_section")
    def target_population():
        criteria = card.cfg.preprocessing.selection.student_criteria

        if not criteria:
            return "  - NOTE TO DATA SCIENTIST: Cannot detect information on student criteria filtering. Please specify in model card."

        def clean_key(key):
            return key.replace("_", " ").title()

        lines = [f"    - {clean_key(k)} = {v}" for k, v in criteria.items()]
        description = (
            "  - We focused our final dataset on the following target population:\n" +
            "\n".join(lines)
        )
        return description
    
    @registry.register("checkpoint_section")
    def checkpoint():
        # if "credits" in card.cfg.preprocessing.checkpoint.params: 
        #     return card.cfg.preprocessing.checkpoint.params["min_num_credits"]
        return "ok"