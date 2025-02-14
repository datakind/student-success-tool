# Databricks notebook source
# This is the notebook for data validation
#
# TODO This is a placeholder notebook Still to be implemented

processed_dataset_table_path = dbutils.jobs.taskValues.get(taskKey="data_processing", key="processed_dataset_path")
print(processed_dataset_table_path)

print("Data validation notebook")