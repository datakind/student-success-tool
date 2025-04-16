# Data Validation Task with TFDV

This Databricks job uses [TensorFlow Data Validation (TFDV)](https://www.tensorflow.org/tfx/data_validation/get_started) to validate data against a reference schema and detect anomalies. It's designed for use in both training and inference pipelines.

## How it Works

1. **Loads a reference schema:** The job loads a schema.pbtxt file, this should be the expected schema determined and persisted during the training / experimentation phase of model development and maintained as need (see [ref.](https://www.tensorflow.org/tfx/data_validation/get_started#inferring_a_schema_over_the_data)). It also assumes that there are two schema environments; TRAINING and SERVING, see [ref.](https://www.tensorflow.org/tfx/data_validation/get_started#schema_environments) for details.
2. **Generates statistics:** It generates statistics from a Delta table or CSV file and persists them for further analysis (e.g. [checking skew and drift](https://www.tensorflow.org/tfx/data_validation/get_started#checking_data_skew_and_drift)).
3. **Validates data:** It validates the data against the schema and generates a list of anomalies, optionally causing the task to fail upon detection.
4. **Saves anomalies:** It saves the anomalies to a pbtxt file for review.

## Job Parameters

The job requires the following parameters:

* `--input_table_path`: Path to the input Delta table.
* `--input_table_format`: Format of the input table (default: `delta`).
* `--input_schema_path`: Path to the input reference schema pbtxt file.
* `--output_artifact_path`: Path to write output artifacts (anomalies and statistics).
* `--environment`: Environment to use for validation (`TRAINING` or `SERVING`).
* `--fail_on_anomalies_true`: Fail the job if anomalies are found (default).
* `--fail_on_anomalies_false`: Do not fail the job if anomalies are found.

**NOTE:** `--fail_on_anomalies_true` and `fail_on_anomalies_false` are a [mutually exclusive group](https://docs.python.org/3/library/argparse.html#mutual-exclusion), only one per run should be passed to the task.

## How to Run

1. **Create a Databricks Job:** In your Databricks workspace, create a new job.
2. **Add a Task:** Add a Python task to the job.
3. **Configure the Task:**
    * **Python File:** Upload the Python script to DBFS and provide the path in the task configuration (git repo optional as well).
    * **Parameters:** Provide the required parameters as described above.
4. **Run the Job:** Run the job to validate your data.

**NOTE:** You can also easily integrate this task with other pipelines / jobs / tasks as well, and also import the DataValidationTask class and create new tasks!

## Example

To validate data in a Delta table against a schema and fail the job if anomalies are found:
```
[
"--input_table_path", "project.schema.table",
"--input_schema_path", "/path/to/your/schema.pbtxt",
"--output_artifact_path", "/path/to/your/output/directory",
"--environment", "SERVING",
"--fail_on_anomalies_true"
]
```

You can also use [job parameters](https://docs.databricks.com/en/jobs/job-parameters.html#configure-job-parameters) or [task values](https://docs.databricks.com/en/jobs/task-values.html#reference-task-values) to dynamically pass values in when looking to parametrize or integrate with other tasks:
```
[
"--input_table_path", "{{tasks.prepro.values.processed_dataset}}",
"--input_schema_path", "{{job.parameters.input_schema_path}}",
"--output_artifact_path", "{{job.parameters.output_artifact_path}}",
"--environment", "SERVING",
"--fail_on_anomalies_true"
]
```


## Contributing

Contributions are welcome! Please submit a pull request with your changes.