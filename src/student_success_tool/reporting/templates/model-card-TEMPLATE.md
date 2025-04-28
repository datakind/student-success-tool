
{logo}

## Model Card: {institution_name}

### Overview
- The model predicts the risk of {outcome_variable_section} within {timeframe_section} based on student, course, and academic data.
- The model makes this prediction when the student has completed {checkpoint_section}.
- Developed by DataKind in {current_year}, Model Version {version_number}
- If there are questions or concerns about the model, you can contact education@datakind.org or your client success manager.

### Intended Use
- **Primary Purpose**: 
  - Identify students who may need support in order to either retain or graduate on time. 
  - Empower academic advisors who provide intervention strategies with information on the factors impacting their need for support.
- **Out-of-Scope Uses**:
  - Outside of the target population: _see below_
  - Without intervention strategies carefully designed by academic advisors, student success professionals, and researchers. 

### Methodology
- **Sample Development**
  - Our first step was our data audit & validation, which included handling null and duplicate values, checking for any inconsistencies between files, and ensuring all student IDs are unique.
  - After validation, we then proceeded with exploratory data analysis (EDA) to develop a deeper understanding of the raw dataset prior to our feature engineering & model development, ensuring alignment with stakeholders through an iterative process.
- **Feature Development**
  - We then proceeded with feature engineering, which involved transforming raw data into meaningful representations by applying semantic abstractions, aggregating at varying levels of term, course, or section analysis, and comparing values cumulatively over time.
  - Stakeholder collaboration was also essential to our feature engineering effort, ensuring domain and use-case knowledge shaped the development of insightful features.
  - Then, our next step was feature selection, applying the following processing:
    - Collinearity Threshold
      - Threshold Applied: Removed features with VIF greater than {collinearity_threshold} were removed to reduce multicollinearity and improve model stability.
      - Explanation: Variance Inflation Factor (VIF) measures how much a feature is linearly correlated with other features. A VIF of 1 would imply no multicollinearity, while a VIF of 10 indicates high collinearity, meaning the feature's information is largely redundant.
    - Low Variance Threshold
      - Threshold Applied: Removed features with variance less than {low_variance_threshold}.
      - Explanation: Features with very low variance do not vary much across observations, meaning they carry little predictive signal. For example, features with variance below 0.01 are often considered near-constant.
    - Missing Data Threshold
      - Threshold Applied: Removed features with more than {incomplete_threshold}% missing values.
      - Explanation: Features with a high percentage of missing values may introduce noise or require extensive imputation.
  - After our feature selection processes, {number_of_features} actionable, relevant, and non-redundant features were retained for modeling.
- **Target Population**
{target_population_section}
  - This resulted in a training dataset of {training_dataset_size} students within the target timeframe.
- **Model Development**
{sample_weight_section}
  - Outcome Variable Definition: {outcome_variable_section}
  - Model Experimentation Data Split:

{data_split_table}

- **Model Evaluation**
  - Evaluated top 10 models for performance across key metrics: accuracy, precision, AUC, recall, log loss, F-1.
  - Evaluated SHAP values indicating relative importance in the models of key features for top-performing models.
  - Evaluated initial model output for interpretability and actionability.
  - Prioritized model quality with transparent and interpretable model outputs.

{model_comparison_plot}

- **Model Interpretability** 
  - Utilized SHAP (Shapley Additive Explanations) values to quantify the contribution of individual features in top-performing models.
  - Leveraged SHAP to enhance interpretability & model transparency, while making model outputs more explainable and actionable.

### Performance
- **Model Performance Metric**
{primary_metric_section}