# Databricks notebook source
# MAGIC %md
# MAGIC #Overview
# MAGIC - Data Loading and Type Recommendations: The notebook begins by loading raw data and recommending appropriate data types for each column. These recommendations are both displayed on-screen and saved as configuration files in the working directory. You are encouraged to review these recommendations carefully and modify the configuration files as needed.
# MAGIC
# MAGIC - Data Type Conversion: The recommended configurations are then fed into the data type conversion function, which applies these changes to the DataFrame.
# MAGIC
# MAGIC - Data Cleaning Recommendations: Subsequent functions suggest various data cleaning steps, including:
# MAGIC
# MAGIC   - Imputing missing values in boolean columns that contain only one non-null value.
# MAGIC   - Dropping columns with only a single unique value or those that are entirely null.
# MAGIC   - Removing rows with missing values in columns where the missingness is minimal.
# MAGIC You can adjust the thresholds for these cleaning operations directly in the code. Recommendations are outputted to configuration files, which you can modify before applying the changes.
# MAGIC
# MAGIC #Example
# MAGIC For instance, the notebook recommends removing the EarliestAwardEarned column due to its 80%+ null rate. However, since this column is used to determine if a student graduated by a target date, we choose to manually handle this recommendation by removing it from the configuration file and addressing it according to our own judgment.
# MAGIC

# COMMAND ----------

# MAGIC %pip install matplotlib=="3.9.0"
# MAGIC dbutils.library.restartPython()
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os
import re

def generate_file_mapping(main_path, keywords=['course', 'semester', 'student']):
    """
    Generate a dictionary mapping keywords to CSV filenames for a specific client.

    Args:
        main_path (str): The base directory where school files are stored.
        keywords (list of str): List of keywords to search for in filenames, e.g. course, student, semester.

    Returns:
        dict: A dictionary where each keyword is mapped to the filename of the corresponding CSV file.
              Only keywords with matching files are included in the dictionary.
    
    Raises:
        FileNotFoundError: If school's directory does not exist.
        FileNotFoundError: If any of the keywords do not have a corresponding CSV file.
    """
    
    # Check if the directory exists
    if not os.path.isdir(main_path):
        raise FileNotFoundError(f"The directory {main_path} does not exist.")
    
    # List all files in the client's directory
    files = [file for file in os.listdir(main_path) if file.endswith('.csv')]
    
    # Initialize a dictionary to store file mappings
    file_mapping = {}

   # Process each file in the directory
    for file in files:
        # Get the base name of the file (without extension) and convert to lower case
        file_name = os.path.splitext(file)[0].lower()
        
        # Check if any keyword is in the file name
        for keyword in keywords:
            if keyword.lower() in file_name:
                # Map the keyword to the filename of the CSV file
                file_mapping[keyword] = file
                break  # Move to the next file once a keyword match is found
    
    # Check if all keywords have corresponding files
    for keyword in keywords:
        if keyword not in file_mapping:
            raise FileNotFoundError(f"No file found for keyword '{keyword}' in directory '{main_path}'")
    
    return file_mapping

# COMMAND ----------

import unittest
import os
import tempfile


class TestGenerateFileMapping(unittest.TestCase):
    """
    Unit tests for the generate_file_mapping function.

    This class tests the function to ensure it correctly
    maps keywords to CSV filenames within a specified directory. It also
    verifies the function's behavior when files are missing, when no keywords are
    provided, and when the directory does not exist.

    Methods:
        setUpClass: Creates a temporary directory and csv files for testing purposes.
        tearDownClass: Cleans up the temporary directory and its contents after all tests are run.
        test_valid_files: Tests the function with all expected files present.
        test_missing_file: Tests the function's response when a required file is missing.
        test_no_keywords: Tests the function's behavior when no keywords are provided.
        test_directory_not_found: Tests the function's response when the directory does not exist.

    Attributes:
        test_dir (str): Path to the temporary directory used for testing.
        files (dict): Dictionary of filenames and their contents for use in tests.
    """
    @classmethod
    def setUp(self):
        """
        Set up a temporary directory with test CSV files.
        """
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        print(f"Temporary directory created at: {self.test_dir}")
        
        # Create temporary CSV files with different names
        self.files = {
            'products_2023.csv': 'dummy content',
            'sales_data.csv': 'dummy content',
            'inventory_report.csv': 'dummy content',
            'notes.txt': 'dummy content'
        }
        
        # Write files to the temporary directory
        for file_name, content in self.files.items():
            file_path = os.path.join(self.test_dir, file_name)
            try:
                with open(file_path, 'w') as f:
                    f.write(content)
                print(f"File created: {file_path}")
            except IOError as e:
                print(f"Error creating file {file_path}: {e}")
    @classmethod
    def tearDown(self):
        """
        Clean up the temporary directory after tests.
        """
        for file_name in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file_name))
        os.rmdir(self.test_dir)

    def test_valid_files(self):
        """
        Test when all expected files are present.
        """
        keywords = ['products', 'sales', 'inventory']
        expected_mapping = {
            'products': 'products_2023.csv',
            'sales': 'sales_data.csv',
            'inventory': 'inventory_report.csv'
        }
        
        result = generate_file_mapping(main_path=self.test_dir, keywords=keywords)
        self.assertEqual(result, expected_mapping)

    def test_missing_file(self):
            """
            Test when one of the files is missing.
            """
            # Remove one file to simulate the missing file scenario
            os.remove(os.path.join(self.test_dir, 'sales_data.csv'))
            
            keywords = ['products', 'sales', 'inventory']
            
            with self.assertRaises(FileNotFoundError) as cm:
                generate_file_mapping(main_path=self.test_dir, keywords=keywords)
            
            # Check if the exception message contains the correct keyword
            self.assertIn("No file found for keyword 'sales'", str(cm.exception))
            self.assertIn(f"in directory '{self.test_dir}'", str(cm.exception))
        
    def test_no_keywords(self):
            """
            Test when no keywords are provided.
            """
            keywords = []
            expected_mapping = {}
            
            result = generate_file_mapping(main_path=self.test_dir, keywords=keywords)
            self.assertEqual(result, expected_mapping)
    
    def test_directory_not_found(self):
        """
        Test when the directory does not exist.
        """
        invalid_path = os.path.join(self.test_dir, 'non_existent_directory')
        
        with self.assertRaises(FileNotFoundError) as cm:
            generate_file_mapping(main_path=invalid_path, keywords=['products'])
        
        # Check if the exception message is correct
        self.assertIn(f"The directory {invalid_path} does not exist.", str(cm.exception))

    
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)   

# COMMAND ----------

# MAGIC %md
# MAGIC ## Change Path as Needed

# COMMAND ----------

# Get the current directory and go one level up
current_directory = os.getcwd()  # Get current working directory
parent_directory = os.path.dirname(current_directory)  # Go one level up

# Specify the path to the data folder and the CSV file
data_folder_path = os.path.join(parent_directory, 'synthetic-data/zogotech')

# COMMAND ----------

files = generate_file_mapping(data_folder_path,keywords=['CourseFile', 'SemesterFile', 'StudentFile'])
print(files)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create dataframes from these files

# COMMAND ----------

import os
import pandas as pd

def create_dataframes_from_files(files, main_path):
    """
    Create Pandas DataFrames from CSV files and store them in a dictionary.

    Args:
        files (dict): A dictionary where keys are DataFrame names (without suffix) 
                      and values are CSV file names.
        main_path (str): The base path where the CSV files are located.

    Returns:
        dict: A dictionary where each key is a DataFrame name with '_df' suffix 
              and each value is a Pandas DataFrame created from the corresponding CSV file.

    Raises:
        FileNotFoundError: If any of the files in the `files` dictionary do not exist.
    """
    dataframes = {}
    
    for key, file_name in files.items():
        file_path = os.path.join(main_path, file_name)
        
        # Check if the file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(file_path)
        
        # Store the DataFrame in the dictionary with a key like 'inventory_df'
        dataframes[f"{key}_df"] = df

    return dataframes


# COMMAND ----------

import unittest
import os
import tempfile
import pandas as pd


class TestCreateDataFramesFromFiles(unittest.TestCase):
    """
    Unit tests for the `create_dataframes_from_files` function.
    
    This class tests the `create_dataframes_from_files` function to ensure it 
    correctly creates Pandas DataFrames from CSV files and handles errors 
    appropriately.
    """

    @classmethod
    def setUpClass(cls):
        """
        Create a temporary directory and test files before any tests are run.
        """
        cls.test_dir = tempfile.mkdtemp()
        
        # Create test files with dummy content
        cls.files = {
            'inventory': 'inventory.csv',
            'sales': 'sales_data.csv',
            'products': 'products_2023.csv'
        }
        
        for file_name in cls.files.values():
            file_path = os.path.join(cls.test_dir, file_name)
            with open(file_path, 'w') as f:
                f.write('id,name,quantity\n1,Item1,10\n2,Item2,20\n')

    @classmethod
    def tearDownClass(cls):
        """
        Remove the temporary directory and its contents after all tests are run.
        """
        for file_name in cls.files.values():
            os.remove(os.path.join(cls.test_dir, file_name))
        os.rmdir(cls.test_dir)

    def test_create_dataframes_success(self):
        """
        Test the function with all expected files present.
        """
        expected_keys = ['inventory_df', 'sales_df', 'products_df']
        
        dataframes = create_dataframes_from_files(self.files, self.test_dir)
        
        # Check that all expected DataFrame keys are in the dictionary
        for key in expected_keys:
            self.assertIn(key, dataframes)
        
        # Check the contents of one DataFrame
        inventory_df = dataframes['inventory_df']
        self.assertEqual(inventory_df.shape, (2, 3))  # Check if DataFrame has 2 rows and 3 columns
        self.assertEqual(list(inventory_df.columns), ['id', 'name', 'quantity'])

    def test_file_not_found(self):
        """
        Test the function's response when a required file is missing.
        """
        files_with_missing_file = {
            'inventory': 'inventory.csv',  # This file will exist
            'sales': 'non_existent_file.csv',  # This file does not exist
            'products': 'products_2023.csv'  # This file will exist
        }
        
        with self.assertRaises(FileNotFoundError) as cm:
            create_dataframes_from_files(files_with_missing_file, self.test_dir)
        
        # Check if the exception message contains the correct file path
        self.assertIn("The file", str(cm.exception))
        self.assertIn("does not exist.", str(cm.exception))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 

# COMMAND ----------

dfs = create_dataframes_from_files(files, data_folder_path)

# COMMAND ----------

print(len(dfs.keys()), "dataframes were created. They are: ", ', '.join(dfs.keys()))

# COMMAND ----------

course_df = dfs['CourseFile_df']
semester_df = dfs['SemesterFile_df']
student_df = dfs['StudentFile_df']

# COMMAND ----------

# MAGIC %md
# MAGIC ## The below function will recommend data type conversion

# COMMAND ----------

import pandas as pd
import numpy as np
import json

def generate_data_type_cleaning_report(df, df_name):
    """
    Generate a data cleaning report as a DataFrame to identify and correct data type issues,
    including handling binary columns and flag columns.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        df_name (str): The name of the DataFrame to be used in generating the filename for the configuration.

    Returns:
        pd.DataFrame: A DataFrame containing current and suggested data types for each column.
    """
    # Identify current data types
    current_dtypes = df.dtypes
    
    # Deduce suggested data types
    suggested_dtypes = {}
    for col in df.columns:
        dtype = df[col].dtype
        if col.lower() == 'student_id' or col.lower() == 'stunum' :
            suggested_dtypes[col] = 'text'
        elif dtype == 'object':
            unique_vals = set(df[col].dropna().unique())
            if unique_vals.issubset({'Y', 'N', np.nan}):
                suggested_dtypes[col] = 'boolean'
            elif df[col].nunique() < 0.05 * len(df):
                suggested_dtypes[col] = 'category'
            else:
                suggested_dtypes[col] = 'text'
        elif dtype in ['int64', 'float64']:
            # Check for binary columns and suggest boolean or numeric
            if df[col].nunique() == 2 and set(df[col].dropna().unique()).issubset({0, 1}):
                suggested_dtypes[col] = 'boolean'
            else:
                # For numerical types, check if there are missing values or need for type conversion
                if df[col].isnull().any():
                    suggested_dtypes[col] = 'numeric'
                else:
                    suggested_dtypes[col] = 'numeric'
        elif dtype == 'datetime64[ns]':
            suggested_dtypes[col] = 'datetime'
        else:
            suggested_dtypes[col] = dtype

    # Create a DataFrame report
    report_df = pd.DataFrame({
        'Column': df.columns,
        'Current Data Type': [current_dtypes[col] for col in df.columns],
        'Suggested Data Type': [suggested_dtypes[col] for col in df.columns]
    })

    # Display the DataFrame to the screen
    print("Data Cleaning Report:")
    print(report_df)

    # Output a configuration file
    config = {
        'suggested_data_types': suggested_dtypes
    }
    
    config_filename = f"{df_name}_data_cleaning_config.json"
    with open("configs/"+config_filename, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuration file saved as '{config_filename}'")

    return report_df


# COMMAND ----------

import unittest
import pandas as pd
import numpy as np
import json
import os


class TestGenerateDataCleaningReport(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame({
            'student_id': ["1", "2", "3", "4"],
            'status': ['Y', 'N', 'Y', 'N'],
            'score': [90, 85, np.nan, 88],
            'date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01']),
            'flag': [0, 1, 1, 0]
        })
        self.df_name = 'student_df'

    def test_generate_data_cleaning_report(self):
        # Run the function
        report_df = generate_data_type_cleaning_report(self.df, self.df_name)
        
        # Check if the report DataFrame has the expected columns
        self.assertTrue('Column' in report_df.columns)
        self.assertTrue('Current Data Type' in report_df.columns)
        self.assertTrue('Suggested Data Type' in report_df.columns)
        
        # Check if the configuration file is created
        config_filename = f"{self.df_name}_data_cleaning_config.json"
        self.assertTrue(os.path.isfile("configs/"+config_filename))
        
        # Load the configuration file
        with open("configs/"+config_filename, 'r') as f:
            config = json.load(f)
        
        # Check if the configuration file contains the expected data
        expected_config = {
            'suggested_data_types': {
                'student_id': 'text',
                'status': 'boolean',
                'score': 'numeric',
                'date': 'datetime',
                'flag': 'boolean'
            }
        }
        
        self.assertEqual(config, expected_config)
        
        # Clean up the configuration file
        os.remove("configs/"+config_filename)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## This function will execute data type conversions

# COMMAND ----------

import pandas as pd

def convert_data_types(df, config):
    """
    Convert data types in the DataFrame based on the cleaning configuration.

    Args:
        df (pd.DataFrame): The DataFrame to clean.
        config (dict): The configuration dictionary containing suggested data types.

    Returns:
        pd.DataFrame: The DataFrame with updated data types.
    """
    # Extract suggested data types from the config dictionary
    suggested_dtypes = config.get('suggested_data_types', {})

    for col, suggested_dtype in suggested_dtypes.items():
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame.")
            continue
        
        if suggested_dtype == 'category':
            df[col] = df[col].astype('category')
        elif suggested_dtype == 'text':
            df[col] = df[col].astype(str)
        elif suggested_dtype == 'numeric':
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, coerce errors to NaN
        elif suggested_dtype == 'datetime':
            df[col] = pd.to_datetime(df[col], errors='coerce')  # Convert to datetime, coerce errors to NaT
        elif suggested_dtype == 'boolean':
            if df[col].dropna().isin(['Y', 'N']).all():
                df[col] = df[col].map({'Y': True, 'N': False})
            else:
                df[col] = df[col].astype(bool)  # Handle binary columns or other cases
        
    return df


# COMMAND ----------

import unittest
import pandas as pd
import numpy as np


class TestConvertDataTypes(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame({
            'student_id': ["1", "2", "3", "4"],
            'status': ['Y', 'N', 'Y', 'N'],
            'score': [90, 85, np.nan, 88],
            'date': ['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01'],
            'flag': [0, 1, 1, 0]
        })

        # Define the configuration dictionary
        self.config = {
            'suggested_data_types': {
                'student_id': 'str',
                'status': 'boolean',
                'score': 'numeric',
                'date': 'datetime',
                'flag': 'boolean'
            }
        }

    def test_convert_data_types(self):
        # Run the function
        converted_df = convert_data_types(self.df, self.config)
        
        # Check the data types of the columns
        self.assertEqual(converted_df['student_id'].dtype.name, 'object')
        self.assertEqual(converted_df['status'].dtype.name, 'bool')
        self.assertEqual(converted_df['score'].dtype.name, 'float64')
        self.assertEqual(converted_df['date'].dtype.name, 'datetime64[ns]')
        self.assertEqual(converted_df['flag'].dtype.name, 'bool')

        # Additional assertions to check the actual data content
        self.assertTrue(pd.api.types.is_object_dtype(converted_df['student_id']))
        self.assertTrue(pd.api.types.is_bool_dtype(converted_df['status']))
        self.assertTrue(pd.api.types.is_numeric_dtype(converted_df['score']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(converted_df['date']))
        self.assertTrue(pd.api.types.is_bool_dtype(converted_df['flag']))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)   


# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate data type conversion reports. Add any additional dfs as needed. Examine recommendations. You can modify them in a config manually if you do not agree with these recommendations

# COMMAND ----------


if not os.path.exists("configs"):
    os.mkdir("configs")
course_df_report = generate_data_type_cleaning_report(course_df, "course")
semester_df_report = generate_data_type_cleaning_report(semester_df, "semester")
student_df_report = generate_data_type_cleaning_report(student_df, "student")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Went to config and changed a few things that did not make sense

# COMMAND ----------

# MAGIC %md
# MAGIC - Course_Number category
# MAGIC - Semester category
# MAGIC - Degree_Type_Pursued category
# MAGIC - Major category
# MAGIC - Department category
# MAGIC - FirstGenFlag boolean
# MAGIC - Age group category
# MAGIC - Ethnicipt category
# MAGIC - ZIP text            
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply data type conversion recommendations. If you have additional files please add the config load and the function call below

# COMMAND ----------

import json

# Load the JSON data from the files
with open("configs/course_data_cleaning_config.json") as f:
    course_config = json.load(f)

with open("configs/semester_data_cleaning_config.json") as f:
    semester_config = json.load(f)

with open("configs/student_data_cleaning_config.json") as f:
    student_config = json.load(f)

# Call the convert_data_types() function with the correct config variables
course_df_converted = convert_data_types(course_df, course_config)
semester_df_converted = convert_data_types(semester_df, semester_config)
student_df_converted = convert_data_types(student_df, student_config)

# COMMAND ----------

import pandas as pd
import json

def check_single_unique_value(df):
    """
    Check for columns with a single unique value or 100% missing values.
    """
    recommendations = {}
    for col in df.columns:
        if df[col].isnull().all() or df[col].nunique() == 1:
            recommendations[col] = {
                'recommendation': 'Drop column',
                'count_missing': int(df[col].isnull().sum()),  # Convert to int
                'percentage_null': 100.0
            }
    return recommendations

def check_missing_values(df, threshold=0.5):
    """
    Check for columns with more than the specified percentage of missing values.
    """
    recommendations = {}
    for col in df.columns:
        if df[col].isnull().mean() > threshold:
            recommendations[col] = {
                'recommendation': 'Drop column',
                'count_missing': int(df[col].isnull().sum()),  # Convert to int
                'percentage_null': df[col].isnull().mean() * 100
            }
    return recommendations

def impute_boolean_columns(df):
    """
    Impute missing values in boolean columns where there is only one type of non-null value.
    """
    recommendations = {}
    for col in df.columns:
        if df[col].dtype == 'bool':
            if df[col].isnull().sum() > 0:
                unique_values = set(df[col].dropna().unique())
                if len(unique_values) == 1:
                    existing_value = unique_values.pop()
                    impute_value = not existing_value
                    recommendations[col] = {
                        'recommendation': f'Impute with "{impute_value}"',
                        'count_missing': int(df[col].isnull().sum()),  # Convert to int
                        'percentage_null': df[col].isnull().mean() * 100
                    }
    return recommendations

def impute_flag_columns(df):
    """
    Impute missing values in flag columns based on opposite values.
    """
    recommendations = {}
    positive_values = {'Y', 'yes', 'True', 'T', 1, '1'}
    negative_values = {'N', 'no', 'False', 'F', 0, '0'}
    
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_values = set(df[col].dropna().unique())
            if unique_values.issubset(positive_values.union(negative_values)):
                if df[col].isnull().sum() > 0:
                    if any(value in positive_values for value in unique_values):
                        impute_value = 'False'
                        impute_type = 'negative'
                    elif any(value in negative_values for value in unique_values):
                        impute_value = 'True'
                        impute_type = 'positive'
                    else:
                        impute_value = 'Unknown'  # Fallback imputation value
                        impute_type = 'unknown'

                    recommendations[col] = {
                        'recommendation': f'Impute with "{impute_value}"',
                        'count_missing': int(df[col].isnull().sum()),  # Convert to int
                        'percentage_null': df[col].isnull().mean() * 100,
                        'impute_type': impute_type
                    }
    return recommendations

def check_columns_with_low_missing(df, threshold=0.01):
    """
    Check for columns with less than the specified percentage of missing values and recommend dropping rows.
    """
    recommendations = {}
    for col in df.columns:
        if df[col].isnull().mean() < threshold:
            if df[col].isnull().sum() > 0:
                recommendations[col] = {
                    'recommendation': 'Drop rows with missing values',
                    'count_missing': int(df[col].isnull().sum()),  # Convert to int
                    'percentage_null': df[col].isnull().mean() * 100
                }
    return recommendations

def generate_data_cleaning_recommendations(df, df_name):
    """
    Generate data cleaning recommendations based on the DataFrame's characteristics and save them as a configuration file.
    
    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        df_name (str): The name of the DataFrame used for generating the filename.

    Returns:
        None
    """
    recommendations = {
        'drop': {},
        'impute': {
            'positive': {},
            'negative': {}
        },
        'drop_rows': {}
    }

    # Perform individual checks
    imputed_boolean = impute_boolean_columns(df)
    imputed_flags = impute_flag_columns(df)

    # Track columns recommended for imputation
    imputed_columns = set(imputed_boolean.keys()).union(imputed_flags.keys())

    # Add imputation recommendations
    for col, rec in imputed_boolean.items():
        recommendations['impute']['negative'][col] = rec

    for col, rec in imputed_flags.items():
        if rec['impute_type'] == 'positive':
            recommendations['impute']['positive'][col] = rec
        elif rec['impute_type'] == 'negative':
            recommendations['impute']['negative'][col] = rec

    # Add drop recommendations for columns with 100% unique values or more than 50% missing values
    drop_recommendations = check_single_unique_value(df)
    drop_recommendations.update(check_missing_values(df, threshold=0.5))
    
    for col, rec in drop_recommendations.items():
        if col not in imputed_columns:
            recommendations['drop'][col] = rec

    # Add recommendations for dropping rows where columns with less than 1% missing data have missing values
    drop_rows_recommendations = check_columns_with_low_missing(df, threshold=0.01)
    
    for col, rec in drop_rows_recommendations.items():
        recommendations['drop_rows'][col] = rec

    # Create a DataFrame report of recommendations
    report_entries = []
    for action_type, actions in recommendations.items():
        if action_type == 'drop':
            for col, rec in actions.items():
                report_entries.append({
                    'Column': col,
                    'Action': rec['recommendation'],
                    'Count Missing': rec['count_missing'],
                    'Percentage Null': f"{rec['percentage_null']:.1f}%"
                })
        elif action_type == 'impute':
            for impute_type, actions in actions.items():
                for col, rec in actions.items():
                    report_entries.append({
                        'Column': col,
                        'Action': rec['recommendation'],
                        'Count Missing': rec['count_missing'],
                        'Percentage Null': f"{rec['percentage_null']:.1f}%",
                        'Impute Type': 'Positive' if impute_type == 'positive' else 'Negative'
                    })
        elif action_type == 'drop_rows':
            for col, rec in actions.items():
                report_entries.append({
                    'Column': col,
                    'Action': rec['recommendation'],
                    'Count Missing': rec['count_missing'],
                    'Percentage Null': f"{rec['percentage_null']:.1f}%"
                })

    report_df = pd.DataFrame(report_entries)

    # Display the DataFrame to the screen
    print("Data Cleaning Recommendations Report:")
    print(report_df)

    # Save recommendations to a JSON file
    config_filename = f"{df_name}_data_cleaning_recommendations.json"
    with open("configs/"+config_filename, 'w') as f:
        json.dump(recommendations, f, indent=4, default=int)  # Use default=int to handle numpy.int64

    print(f"Data cleaning recommendations saved as '{config_filename}'")

# Example usage
# df = pd.DataFrame(...) # your DataFrame here
# generate_data_cleaning_recommendations(df, 'student_df')


# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate recommended data cleaning reports. Add any additional dfs as needed. Examine recommendations. You can modify them in a config manually if you do not agree with these recommendations

# COMMAND ----------

generate_data_cleaning_recommendations(course_df_converted, 'course_df')

# COMMAND ----------

generate_data_cleaning_recommendations(semester_df_converted, 'semester_df')

# COMMAND ----------

generate_data_cleaning_recommendations(student_df_converted, 'student_df')

# COMMAND ----------

# MAGIC %md
# MAGIC ## CHECK THESE RECOMMENDATIONS CAREFULLY AND ADJUST CONFIGS AS NEEDED BEFORE RUNNING THE NEXT FUNCTION. YOU MAY WANT TO CONDUCT EDA ELSEWHERE TO CONFIRM AS THIS WAS WRITTEN BASED ON ONE EXAMPLE :)

# COMMAND ----------

# MAGIC %md
# MAGIC e.g do not drop EarnedAwardDate as we will use it later in the code to figure out if the student graudated on time or not

# COMMAND ----------

import pandas as pd
import json

def apply_recommendations(df, config_file):
    """
    Apply data cleaning recommendations from a JSON config file to a DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to clean.
        config_file (str): Path to the JSON config file with recommendations.
    
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Load recommendations from the JSON config file
    with open(config_file, 'r') as f:
        recommendations = json.load(f)
    
    # Apply column dropping recommendations
    if 'drop' in recommendations:
        columns_to_drop = recommendations['drop'].keys()
        df.drop(columns=columns_to_drop, inplace=True)
        print(f"Dropped columns: {columns_to_drop}")
    
    # Apply row dropping recommendations
    if 'drop_rows' in recommendations:
        for col, rec in recommendations['drop_rows'].items():
            if col in df.columns:
                df = df[df[col].notnull()]
                print(f"Dropped rows with missing values in column: {col}")
    
    # Apply imputations
    if 'impute' in recommendations:
        # Handle positive imputations
        for col, rec in recommendations['impute'].get('positive', {}).items():
            if col in df.columns:
                impute_value = rec['recommendation'].split('"')[1]
                df[col].fillna(impute_value, inplace=True)
                print(f"Imputed missing values in column {col} with '{impute_value}'")
        
        # Handle negative imputations
        for col, rec in recommendations['impute'].get('negative', {}).items():
            if col in df.columns:
                impute_value = rec['recommendation'].split('"')[1]
                df[col].fillna(impute_value, inplace=True)
                print(f"Imputed missing values in column {col} with '{impute_value}'")
    
    return df

# Example usage
# df = pd.DataFrame(...) # your DataFrame here
# cleaned_df = apply_recommendations(df, 'student_df_data_cleaning_recommendations.json')


# COMMAND ----------

cleaned_student_df = apply_recommendations(student_df_converted, 'configs/student_df_data_cleaning_recommendations.json')

# COMMAND ----------

cleaned_course_df = apply_recommendations(course_df_converted, 'configs/course_df_data_cleaning_recommendations.json')

# COMMAND ----------

cleaned_semester_df = apply_recommendations(semester_df_converted, 'configs/semester_df_data_cleaning_recommendations.json')

# COMMAND ----------

# MAGIC %md
# MAGIC > ## Write clean data to databricks - will not work here based on parameters not being specified

# COMMAND ----------

catalog = 
schema = 
table_name = 
df = cleaned_semester_df

# COMMAND ----------

# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(df)

# Register the Spark DataFrame as a temporary view
spark_df.createOrReplaceTempView("temp_spark_df")

# Save the Spark DataFrame as a Delta table using SQL statements
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {catalog}.{schema}.{table_name}
    USING DELTA
    AS SELECT * FROM temp_spark_df""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## JOIN data

# COMMAND ----------

joined_df = cleaned_student_df.merge(cleaned_course_df, left_on='StuNum', right_on='Student_ID')

# COMMAND ----------

joined_df.head()

# COMMAND ----------

joined_df.rename(columns={'Number_Of_Credits_Attempted': 'Course_Number_Of_Credits_Attempted'}, inplace=True)
joined_df.rename(columns={'Number_Of_Credits_Earned': 'Course_Number_Of_Credits_Earned'}, inplace=True)

# COMMAND ----------

cleaned_semester_df.rename(columns={'Number_Of_Credits_Attempted': 'Term_Number_Of_Credits_Attempted'}, inplace=True)
cleaned_semester_df.rename(columns={'Number_Of_Credits_Earned': 'Term_Number_Of_Credits_Earned'}, inplace=True)

# COMMAND ----------

joined_df_final = joined_df.merge(cleaned_semester_df, on=["Student_ID","Semester"])
joined_df_final.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply Agreed-Upon Filters

# COMMAND ----------

# Drop rows where dualenroll is 1
joined_df_final.drop(columns=['DualEnrollFlag'], inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Now Filter to Students who started 3 years before max date of the dataset and 6 years for part time 

# COMMAND ----------

# Get the latest graduation date
joined_df_final['EarnedAwardDate'] = pd.to_datetime(joined_df_final['EarnedAwardDate'], errors='coerce')
latest_graduation_date = joined_df_final['EarnedAwardDate'].max()

# Map enrollment types to their corresponding time frames
time_frames = {
    'FT': 6,  # Full-time: 6 years
    'PT': 3   # Part-time: 3 years
}

# Convert First_Enrollment_Date from MM/DD/YYYY to datetime
joined_df_final['First_Enrollment_Date'] = pd.to_datetime(joined_df_final['First_Enrollment_Date'], format='%m/%d/%Y', errors='coerce')

# COMMAND ----------

def calculate_graduation_date(row):
    if pd.isna(row['First_Enrollment_Date']):
        return pd.NaT
    
    # Get the years to add; ensure it defaults to 0 if not found
    years_to_add = time_frames.get(row['Enrollment_Type'], 0)
    
    # Ensure years_to_add is an integer
    if not isinstance(years_to_add, int):
        years_to_add = 0  # Default to 0 if it's not an integer
    
    return row['First_Enrollment_Date'] + pd.DateOffset(years=years_to_add)

# Apply the function to calculate Graduation Eligibility Date
joined_df_final['Graduation_Eligibility_Date'] = joined_df_final.apply(calculate_graduation_date, axis=1)


# COMMAND ----------

joined_df_final.head()

# COMMAND ----------

# Extract year and month from the graduation eligibility date
joined_df_final['Graduation_Eligibility_Year'] = joined_df_final['Graduation_Eligibility_Date'].dt.year
joined_df_final['Graduation_Eligibility_Month'] = joined_df_final['Graduation_Eligibility_Date'].dt.month

# Convert Actual_Graduation_Year and Actual_Graduation_Month columns to numeric
joined_df_final['Actual_Graduation_Year'] = joined_df_final['EarnedAwardDate'].dt.year
joined_df_final['Actual_Graduation_Month'] = joined_df_final['EarnedAwardDate'].dt.month

# COMMAND ----------

# Define cutoff date for filtering (e.g., August 2024)
cutoff_date = pd.to_datetime('2024-08-31')

# Find students who could have graduated before the cutoff date
eligible_students = joined_df_final[
    (joined_df_final['Graduation_Eligibility_Date'] <= cutoff_date)
]

# COMMAND ----------

# Create a flag for graduated on time
eligible_students['Graduated_On_Time'] = (
    (eligible_students['Graduation_Eligibility_Year'] >= eligible_students['Actual_Graduation_Year']) &
    (eligible_students['Graduation_Eligibility_Month'] >= eligible_students['Actual_Graduation_Month'])
)

# COMMAND ----------

def sort_terms(term):
    try:
        year = int(term[:4])  # Extract the year as an integer
        semester = term[4:]  # Extract the semester
        # Define the order for semesters: SP < SU < FA
        semester_order = {'SP': 1, 'SU': 2, 'FA': 3, 'OE': 4}  # Adjusted to the required order
        semester_value = semester_order.get(semester, 5)  # Use a large number for unknown semesters
        return (year, semester_value)  # Return a tuple for sorting
    except Exception as e:
        print(f"Error processing term '{term}': {e}")
        return (float('inf'), float('inf'))  # Handle errors gracefully


# COMMAND ----------

# Check if the DataFrame is MultiIndex
if isinstance(eligible_students.index, pd.MultiIndex):
    eligible_students = eligible_students.reset_index()

# COMMAND ----------

# Ensure we only process non-null 'Semester' values
eligible_students['Semester'] = eligible_students['Semester'].astype(str)

# Apply the sorting function
eligible_students['Sort_Key'] = eligible_students['Semester'].apply(sort_terms)

# COMMAND ----------

# MAGIC %md
# MAGIC ## A bit of Feature Engineering

# COMMAND ----------

def classify_major(major):
    """
    Classify a major as 'Certificate', 'Degree', 'Non Degree Seeking', or 'Unclassified'.
    """
    major_lower = major.lower()
    if 'non degree seeking' in major_lower:
        return 'Non Degree Seeking'
    elif 'certificate' in major_lower or 'certifiicate' in major_lower :
        return 'Certificate'
    elif 'aa' in major_lower or 'associate of' in major_lower or 'as' in major_lower or 'aas' in major_lower:
        return 'Degree'
    else:
        return 'Unclassified'

# Apply classification
eligible_students['Classification'] = eligible_students['Major'].apply(classify_major)
# Identify majors that do not fit into the specified categories
eligible_students = eligible_students[eligible_students['Classification'] == 'Degree']

# COMMAND ----------

# Group by Student_ID and Term, summing the credits attempted and earned
aggregated = eligible_students.groupby(['Student_ID', 'Semester']).agg(
    Term_Number_Of_Credits_Attempted=('Term_Number_Of_Credits_Attempted', 'max'),
    Term_Number_Of_Credits_Earned=('Term_Number_Of_Credits_Earned', 'max')
).reset_index()

aggregated['Sort_Key'] = aggregated['Semester'].apply(sort_terms)
# Sort the aggregated DataFrame by Student_ID and Sort_Key
aggregated = aggregated.sort_values(by=['Student_ID', 'Sort_Key']).drop(columns='Sort_Key')

# Calculate cumulative credits for each student
aggregated['Cumulative_Credits_Attempted'] = aggregated.groupby('Student_ID')['Term_Number_Of_Credits_Attempted'].cumsum()
aggregated['Cumulative_Credits_Earned'] = aggregated.groupby('Student_ID')['Term_Number_Of_Credits_Earned'].cumsum()

# COMMAND ----------

cols_to_keep = ['Student_ID', 'Semester', 'Cumulative_Credits_Attempted', 'Cumulative_Credits_Earned']
aggregated = aggregated[cols_to_keep]

# COMMAND ----------

df = eligible_students.merge(aggregated, on=["Student_ID", "Semester"])

# COMMAND ----------

# Find the first term for each student where cumulative credits reach or exceed 10
df['CheckPoint_Achieved'] = df['Cumulative_Credits_Earned'] >= 10

# COMMAND ----------

# Updated function to determine course level
def determine_course_level(course):
    # Extract the course number, ignoring any suffixe
    course_number = ''.join(filter(str.isdigit, course))  # Keep only digits
    if course_number:  # Check if course_number is not empty
        course_number = int(course_number)  # Convert to int
        if course_number < 100:
            return 'Below_100'
        elif 100 <= course_number < 200:
            return '100_Level'
        elif 200 <= course_number < 300:
            return '200_Level'
        else:
            return 'Above_300'  # For completeness
    return 'Unknown'

# COMMAND ----------

# Target encoding for Course based on Term_GPA
df['Course_Number'] = df['Course_Number'].astype(str)
df['Course_Level'] = df['Course_Number'].apply(determine_course_level)

# COMMAND ----------

df['Term_Pass_Rate'] = df['Term_Number_Of_Credits_Earned']/df['Term_Number_Of_Credits_Attempted']
df['Cum_Pass_Rate'] = df['Cumulative_Credits_Earned']/df['Cumulative_Credits_Attempted']

# COMMAND ----------

graduation_status_counts = df.groupby('Student_ID')['Graduated_On_Time'].nunique()
# Step 2: Filter to find students with more than one unique flag, i.e. they are both graduated on time and not graduiated oin time. This happens when student was part time at some point and full time at another point
students_with_multiple_flags = graduation_status_counts[graduation_status_counts > 1].index

# Step 3: Update Graduated_On_Time for these students
df.loc[df['Student_ID'].isin(students_with_multiple_flags) & (df['Graduated_On_Time'] == False), 'Graduated_On_Time'] = True

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find First Check Pont

# COMMAND ----------

import pandas as pd

# Assuming df is your DataFrame

# Step 1: Sort the DataFrame by Student_ID and Sort_Key
df['Sort_Key'] = df['Sort_Key'].apply(tuple)
df_sorted = df.sort_values(by=['Student_ID', 'Sort_Key'])

# Step 2: Create a DataFrame to identify the first term where CheckPoint_Achieved is True
first_checkpoint = df_sorted[df_sorted['CheckPoint_Achieved']].groupby('Student_ID')['Sort_Key'].first().reset_index()

# COMMAND ----------

df_merged = df_sorted.merge(first_checkpoint, on='Student_ID', how='left', suffixes=('', '_first'))
df_merged.head()
# Step 2: Create a mask for filtering
mask = (df_merged['Sort_Key'] <= df_merged['Sort_Key_first'])
# Step 3: Apply the mask to get the filtered DataFrame
filtered_df = df_merged[mask]

# COMMAND ----------

filtered_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate/Colapse to oone row per student

# COMMAND ----------

import pandas as pd
import numpy as np

def aggregate_students(df, group_by_cols, drop_cols, last_cols, mode_cols, avg_cols, count_cols):
    # Drop specified columns
    df = df.drop(columns=drop_cols)
    
    # Create a dictionary to define aggregation functions
    agg_funcs = {}
    
    # Define aggregation functions
    agg_funcs.update({col: 'last' for col in last_cols})  # Last value
    agg_funcs.update({col: lambda x: x.mode()[0] if not x.mode().empty else np.nan for col in mode_cols})  # Mode
    agg_funcs.update({col: 'mean' for col in avg_cols})  # Average

    # Prepare to count occurrences for specified categorical columns
    count_dfs = []
    for count_col in count_cols:
        # Create a count DataFrame
        count_df = df.groupby(group_by_cols)[count_col].value_counts().unstack(fill_value=0)
        count_df.columns = [f'{count_col}_{val}' for val in count_df.columns]
        count_dfs.append(count_df)

    # Group by specified columns and aggregate
    grouped_df = df.groupby(group_by_cols).agg(agg_funcs).reset_index()

    # Join all count DataFrames to the grouped DataFrame
    for count_df in count_dfs:
        grouped_df = grouped_df.join(count_df, on=group_by_cols, how='left').fillna(0)

    return grouped_df

# Example usage
result = aggregate_students(
    filtered_df,
    group_by_cols=['Student_ID'],
    drop_cols=['EarnedAward', 'EarnedAwardDate', 'Semester', 'Course_Prefix', 'Course_Number', 'Online_Course_Flag', 'Weeks', 'Graduation_Eligibility_Date', 'Graduation_Eligibility_Year', 'Graduation_Eligibility_Month', 'Actual_Graduation_Year', 'Actual_Graduation_Month', 'Classification', 'Sort_Key'],
    last_cols=['Intent_To_Transfer_Flag', 'Major', 'Cumulative_GPA', 'Cumulative_Credits_Attempted', 'Cumulative_Credits_Earned', 'CheckPoint_Achieved', 'Cum_Pass_Rate', 'Graduated_On_Time'],
    mode_cols=[ 'AgeGroup', 'Gender', 'Race', 'Ethnicity', 'IncarceratedFlag', 'ABEFlag', 'Zip', 'DisabilityFlag', 'HighSchoolStatus', 'HighSchoolGradDate', 'StudentAthlete', 'Enrollment_Type', 'Degree_Type_Pursued', 'Pell_Recipient'],
    avg_cols=['Semester_GPA', 'Course_Pass_Rate', 'Online_Course_Rate', 'Term_Pass_Rate'],
    count_cols=['Pass/Fail_Flag', 'Grade', 'Prerequisite_Course_Flag', 'Modality', 'Number_Of_Courses_Enrolled', 'Course_Level']
)

result.head()


# COMMAND ----------

result.drop(columns=['Major', 'AgeGroup', 'Gender', 'Race', 'Ethnicity', 'IncarceratedFlag', 'ABEFlag', 'Zip', 'DisabilityFlag', 'HighSchoolGradDate', 'Pell_Recipient', 'CheckPoint_Achieved'], inplace=True)

# COMMAND ----------

result
