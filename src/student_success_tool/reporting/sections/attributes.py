def register_attribute_sections(card, registry):
    @registry.register("outcome_variable_section")
    def outcome_variable():
        name = card.cfg.preprocessing.target.name.lower()
        if "graduation" in name or "grad" in name:
            return "graduation"
        elif "retention" in name or "ret" in name:
            return "retention"
        raise NameError("Unable to determine outcome variable from config.")
    
    @registry.register("timeframe_section")
    def timeframe():
        return "ok"
    
    @registry.register("target_population_section")
    def target_population():
        return "ok"
    
    @registry.register("checkpoint_section")
    def checkpoint():
        return "ok"