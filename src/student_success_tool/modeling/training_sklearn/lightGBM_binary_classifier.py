# Databricks notebook source
# MAGIC %pip install "optuna~=3.6.1"
# MAGIC %pip install "optuna-integration~=3.6.0"
# MAGIC %restart_python
# MAGIC

# COMMAND ----------

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import optuna
from optuna.integration.mlflow import MLflowCallback
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from datetime import datetime
import pickle

# COMMAND ----------

# log everything to the same experiment

experiment_location = "/Workspace/Users/vishakh@datakind.org/automl_kinda/automl_experiment"

existing_experiment = mlflow.get_experiment_by_name(experiment_location)

if existing_experiment is None:
    experiment_id = mlflow.create_experiment(experiment_location)
else:
    experiment_id = existing_experiment.experiment_id

mlflow.set_experiment(experiment_location)

# COMMAND ----------

# we are assuming binary classification type here

class LightGBMWrapper:
    def __init__(self, experiment_name="/", random_state=42, n_trials=100, cv_folds=5):
        self.experiment_name = experiment_name
        self.random_state = random_state
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.problem_type = None
        self.target_encoder = None
        self.preprocessing_info = {}
        self.run_id = None
        self.model_uri = None
    
        self.experiment = mlflow.set_experiment(self.experiment_name)
        
    def analyze_data(self, X, y):
        """Automatic data analysis and preprocessing"""
        mlflow.log_param("n_samples", X.shape[0])
        mlflow.log_param("n_features", X.shape[1])

        categorical_features = []
        for i, col in enumerate(X.columns): 
          if X[col].dtype == 'object' or X[col].dtype.name == 'category': 
            categorical_features.append(col)

        self.preprocessing_info['categorical_feature_names'] = categorical_features
                    
    def preprocess_data(self, X, y):
        """Preprocess data optimally for LightGBM"""
        X_processed = X.copy()
        categorical_columns = self.preprocessing_info['categorical_feature_names']

        # we need to log this label encoders, so we encode the test data in the same way
        self.label_encoders = {} 

        for col in categorical_columns: 
          le = LabelEncoder()
          X_processed[col] = le.fit_transform(X_processed[col])
          self.label_encoders[col] = le

        categorical_indices = [i for i, col in enumerate(X_processed.columns) if col in categorical_columns]
        self.preprocessing_info['categorical_indices'] = categorical_indices

        if y.dtype == 'object' or y.dtype.name == 'category': 
          # need encoding aka columns look like [m, f, m ...]
          self.target_encoder = LabelEncoder()
          y_processed = self.target_encoder.fit_transform(y)
        else: 
          # no encoding necessary aka columns looks like [0, 1, 0 ..]
          self.target_encoder = None
          y_processed = y
          

        return X_processed, y_processed
    
    def objective(self, trial, X_train, y_train, X_val, y_val):
        params = {
            'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
            'solver': 'saga',
            'max_iter': trial.suggest_int('max_iter', 100, 2000),
            'random_state': self.random_state
        }

        if params['penalty'] == 'elasticnet':
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)

        try:
            mlflow.start_run(run_name=f"trial_{trial.number}", nested=False)
            mlflow.log_params(params)
            mlflow.set_tag("optuna_trial_number", trial.number)
            mlflow.set_tag("tuning_session_id", self.tuning_session_id)

            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', lgb(**params))
            ])

            pipeline.fit(X_train, y_train)
            y_pred_proba = pipeline.predict_proba(X_val)[:, 1]

            score = roc_auc_score(y_val, y_pred_proba)
            mlflow.log_metric("val_auc", score)

            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                signature=mlflow.models.infer_signature(X_train, y_train)
            )

        except Exception as e:
            mlflow.set_tag("fit_failed", "true")
            mlflow.set_tag("failure_reason", str(e))
            raise  # Re-raise the exception so Optuna marks the trial as failed

        finally:
            mlflow.end_run()

        return -score  # Optuna minimizes


    
    def train(self, X, y, test_size=0.2):
        """Main training pipeline with MLflow tracking"""
        with mlflow.start_run() as run:
            self.run_id = run.info.run_id
            
            mlflow.log_param("timestamp", datetime.now().isoformat())
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("n_trials", self.n_trials)
            mlflow.log_param("cv_folds", self.cv_folds)
            mlflow.log_param("test_size", test_size)
            
            self.analyze_data(X, y)

            X_processed, y_processed = self.preprocess_data(X, y)
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_processed, y_processed, 
                test_size=test_size, 
                random_state=self.random_state,
                stratify=y_processed
            )
          
            mlflow.log_param("train_samples", X_train.shape[0])
            mlflow.log_param("val_samples", X_val.shape[0])
                      
            # create MLFlow Callback so all hyperparam runs are logged to experiment
                    
            study = optuna.create_study(direction='minimize')
            study.optimize(
                lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
                n_trials=self.n_trials
            )
            # Save Optuna study as CSV
            trials_df = study.trials_dataframe()
            trials_df.to_csv("optuna_trials.csv", index=False)
            mlflow.log_artifact("optuna_trials.csv")

            # Plot optimization history
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.tight_layout()
            plt.savefig("optuna_optimization_history.png", dpi=300, bbox_inches='tight')
            mlflow.log_artifact("optuna_optimization_history.png")
            plt.close()

            # Plot parameter importance
            try:
                optuna.visualization.matplotlib.plot_param_importances(study)
                plt.tight_layout()
                plt.savefig("optuna_param_importance.png", dpi=300, bbox_inches='tight')
                mlflow.log_artifact("optuna_param_importance.png")
                plt.close()
            except Exception as e:
                print("Param importance plot skipped:", e)

            
            self.best_params = study.best_params
            for param_name, param_value in self.best_params.items():
                mlflow.log_param(f"best_{param_name}", param_value)
            
            mlflow.log_metric("best_objective_score", study.best_value)
            mlflow.log_metric("n_optimization_trials", len(study.trials))
            
            final_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'random_state': self.random_state,
                'verbose': -1,
                **self.best_params
            }
            
            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=self.preprocessing_info['categorical_indices'])
            val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=self.preprocessing_info['categorical_indices'])
            
            callbacks = [
                lgb.early_stopping(100),
                lgb.log_evaluation(100)
            ]
            
            self.model = lgb.train(
                final_params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=callbacks
            )
            
            mlflow.lightgbm.log_model(
                self.model,
                "model",
                signature=mlflow.models.infer_signature(X_train, y_train))
              
            self.feature_importance = pd.DataFrame({
                'feature': X_processed.columns,
                'importance': self.model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
            
            for i, (_, row) in enumerate(self.feature_importance.head(10).iterrows()):
                mlflow.log_metric(f"top_{i+1}_feature_importance", row['importance'])
                mlflow.log_param(f"top_{i+1}_feature_name", row['feature'])
            
            preprocessing_artifacts = {
                'target_encoder': self.target_encoder,
                'preprocessing_info': self.preprocessing_info,
                'problem_type': self.problem_type
            }
            
            with open('preprocessing_artifacts.pkl', 'wb') as f:
                pickle.dump(preprocessing_artifacts, f)
            mlflow.log_artifact('preprocessing_artifacts.pkl')
            
            self._save_feature_importance_plot()
            mlflow.log_artifact('feature_importance.png')
            
            self._log_validation_metrics(X_val, y_val)
            
            print(f"MLflow run: {self.run_id}")
            
        return self
    
    def _log_validation_metrics(self, X_val, y_val):
        """Log validation metrics to MLflow"""          
        y_pred_proba = self.model.predict(X_val)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        mlflow.log_metric("val_accuracy", accuracy)
        mlflow.log_metric("val_precision", precision)
        mlflow.log_metric("val_recall", recall)
        mlflow.log_metric("val_f1", f1)
        mlflow.log_metric("val_auc", auc)
    
    def _save_feature_importance_plot(self, top_n=20):
        """Save feature importance plot"""
        if self.feature_importance is None:
            return
        
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance (Gain)')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

# COMMAND ----------

df = spark.table("samples.healthverity.claims_sample_synthetic").limit(100).toPandas()
df.head()

# COMMAND ----------

X = df.drop(columns=['patient_gender'])
y = df['patient_gender']

automl = LightGBMWrapper(experiment_name=experiment_location, n_trials=50, cv_folds=5)

automl.train(X, y, test_size=0.2)
