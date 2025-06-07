
{logo}

## Model Card: {institution_name}

### Overview
- {outcome_section}
- {checkpoint_section}
- {development_note_section}
- If there are questions or concerns about the model, you can contact **education@datakind.org** or your client success manager.

### Intended Use
- **Primary Purpose**
    - Identify students who may need support in order to either retain or graduate on time. 
    - Empower academic advisors who provide intervention strategies with information on the factors impacting their need for support.
- **Out-of-Scope Uses**
    - Outside of the target population:  _see below_
    - Without intervention strategies carefully designed by academic advisors, student success professionals, and researchers. 

### Methodology
- **Sample Development**
    - Our first step was our data audit & validation, which included handling null and duplicate values, checking for any inconsistencies between files, and ensuring all student IDs are unique.
    - After validation, we then proceeded with exploratory data analysis (EDA) to develop a deeper understanding of the raw dataset prior to our feature engineering & model development, ensuring alignment with stakeholders through an iterative process.
- **Feature Development**
    - We then proceeded with feature engineering, which involved transforming raw data into meaningful representations by applying semantic abstractions, aggregating at varying levels of term, course, or section analysis, and comparing values cumulatively over time.
    - Stakeholder collaboration was also essential to our feature engineering effort, ensuring domain and use-case knowledge shaped the development of insightful features.
    - Then, our next step was feature selection, applying the following processing below.
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
    - Model Experimentation Data Split

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

- **Model Performance Plots**
{test_confusion_matrix}
{test_calibration_curve}
{test_roc_curve}
{test_histogram}

### Quantitative Bias Analysis
- **Model Bias Metric**
    - Our bias evaluation metric for our model includes utilizing _False Negative Rate Parity_, which measures the disproportionate rate of false negatives across subgroups. 
    - FNR Parity helps assess whether the model is underperforming for any specific group in terms of incorrectly predicting that a student is not in need of support when the true outcome is that the student is in need of support.

- **Analyzing Bias Across Student Groups**
{bias_groups_section}
    - We evaluated FNR across these student groups and tested for statistically significant disparities.

{bias_summary_section}

### Important Features
- **Analyzing Feature Importance**
    - SHAP (Shapley Additive Explanations) is a method based on cooperative game theory that quantifies the contribution of each feature to a model's prediction for an individual instance. It helps us understand how much did a particular feature contribute to predicting whether a student needs more or less support.
    - SHAP provides detailed insight into how much each feature contributed for each individual, as well as Whether higher or lower feature values are associated with higher or lower need for support.

- **Feature Importance Plot**
    - This figure below helps explain how individual features contribute to the model’s prediction for each student-term record. 
    - Here are some guidelines for how to interpret the plot below.
        - Each dot represents a single student record.
        - Color of the dot corresponds to the feature value for that student (e.g., a higher or lower numeric value for that feature), while gray indicates a categorical feature.
        - The x-axis shows the SHAP value for that student-feature pair:
        - A higher SHAP value (further to the right) means the feature is pushing the prediction toward a greater need for support.
        - A lower SHAP value (further to the left) means the feature is contributing toward a lower need for support.
        - Features are ordered from top to bottom by their overall importance to the model — the most influential features appear at the top.
        - Example: _If students have a low percentage of grades above the section average, they tend to have higher SHAP values, indicating a greater need of support in order to graduate on time._

{feature_importances_by_shap_plot}

### Appendix

{performance_by_splits_section}

{selected_features_ranked_by_shap}

{evaluation_by_group_section}