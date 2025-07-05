## SST Custom Templates 
- We start our training notebook at `02-train-model-TEMPLATE` to assume that all preprocessing is 01, though these notebooks will likely be renumbered depending on the number of preprocessing notebooks before.
- Define the _run_type_ parameter in a DB workflow setting with either "train" or "predict". These can be used in a workflow once inference is performed or training.
- This is the skeleton for a pipeline, but can be built upon depending on the amount of customization in the school's pipeline, processing, outcome definition etc.
- 1 config is mapped to 1 model. If a school has multiple models, we need a config for each. This is so that model cards can be created accordingly and configs can have a flat & simple structure (same implemention for PDP).