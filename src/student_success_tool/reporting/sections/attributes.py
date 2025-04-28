def register_attribute_sections(card, registry):    
    @registry.register("outcome_section")
    def outcome():
        name = card.cfg.preprocessing.target.name.lower()

        if "graduation" in name or "grad" in name:
            outcome = "graduation"
        elif "retention" in name or "ret" in name:
            outcome = "retention"
        else:
            raise NameError("Unable to determine outcome variable from config.")

        limits = card.cfg.preprocessing.selection.intensity_time_limits

        # Normalize intensity labels to support flexible formats
        normalized_limits = {
            k.strip().upper().replace(" ", "-"): v for k, v in limits.items()
        }

        def format_time(duration):
            num, unit = duration
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
        return "ok"
    
    @registry.register("checkpoint_section")
    def checkpoint():
        if "credits" in card.cfg.preprocessing.checkpoint.params: 
        return card.cfg.preprocessing.checkpoint.params["min_num_credits"]
