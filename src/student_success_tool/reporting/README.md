# Model Cards 
- This module is used to create model cards for each model developed by an institution. 
- Inputs to this module generally-speaking are the `config.toml` file, catalog name, and model name.
- A model card is produced via `config.toml` (source of truth, model definition) and MLflow artifacts (plots, tables, metrics).

## Structure
- `model_card`: This contains the "main" functions of the model card module and where the ModelCard base class is defined, as well as different overrides such as for PDP or Custom schools.
- `template`: This contains markdown template used to generate model cards. All institutions use this template (PDP and custom).
- `sections`: This has all the separate sections that are registered for the model card. This is separated out from the model card class
for unit testing and to scale our model card for our institution and organizational needs.
- `utils`: This has a formatting class and a general utils file that makes calls to MLflow and embeds images in markdown. We also have a types file, which defines a general model card config so that we can override the config for different platforms.

## Usage

- We first instantiate a model card object using our config, catalog name, and model name.
- Then, we build the model card. 
- Finally, we have a model card markdown that is generated locally.
```
import 
# Initialize card
card = PDPModelCard(config=cfg, catalog=catalog, model_name=model_name)

# Build context and download artifacts
card.build()
```