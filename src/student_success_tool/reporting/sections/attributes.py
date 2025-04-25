def register_attribute_sections(self):
    @self.section_registry.register("outcome_variable_section")
    def outcome_variable():
        name = self.cfg.preprocessing.target.name.lower()
        if "graduation" in name or "grad" in name:
            return "graduation"
        elif "retention" in name or "ret" in name:
            return "retention"
        raise NameError("Unable to determine outcome variable from config.")
    
    @self.section_registry.register("timeframe_section")
    def timeframe():
        return "ok"
    
    @self.section_registry.register("target_population_section")
    def target_population():
        return "ok"
    
    @self.section_registry.regist("checkpoint_section")
    def checkpoint():
        return "ok"