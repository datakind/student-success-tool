# Databricks notebook source
# MAGIC %pip install "optuna~=3.6.1"
# MAGIC %restart_python

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# log everything to the same experiment
experiment_location = "/Workspace/Users/vishakh@datakind.org/automl_kinda/automl_experiment"
try: 
  mlflow.create_experiment(experiment_location)
except: 
  pass

# COMMAND ----------

class LogisticRegressionWrapper:
    def __init__(self, experiment_name="/", random_state=42, n_trials=100, cv_folds=5, encoding_strategy='onehot'):
        """
        AutoML wrapper for Logistic Regression
        """
        self.experiment_name = experiment_name
        self.random_state = random_state
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.encoding_strategy = encoding_strategy
        self.model = None
        self.pipeline = None
        self.best_params = None
        self.feature_importance = None
        self.target_encoder = None
        self.preprocessor = None
        self.preprocessing_info = {}
        self.run_id = None
        self.model_uri = None
        self.feature_names = None
        
        self.experiment = mlflow.set_experiment(self.experiment_name)
        
    def analyze_data(self, X, y):
        """Automatic data analysis and preprocessing identification"""
        mlflow.log_param("n_samples", X.shape[0])
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("encoding_strategy", self.encoding_strategy)

        categorical_features = []
        numerical_features = []
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                categorical_features.append(col)
            else:
                numerical_features.append(col)

        self.preprocessing_info['categorical_features'] = categorical_features
        self.preprocessing_info['numerical_features'] = numerical_features
        
        mlflow.log_param("n_categorical_features", len(categorical_features))
        mlflow.log_param("n_numerical_features", len(numerical_features))
                            
    def create_preprocessor(self, X):
        """Create preprocessing pipeline based on data types"""
        categorical_features = self.preprocessing_info['categorical_features']
        numerical_features = self.preprocessing_info['numerical_features']
        
        transformers = []
        
        if numerical_features:
            transformers.append(('num', StandardScaler(), numerical_features))
        if categorical_features:
            if self.encoding_strategy == 'onehot':
                transformers.append(('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features))
            else:  
                # label encoding
                from sklearn.preprocessing import OrdinalEncoder
                transformers.append(('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features))
        
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        return self.preprocessor
    
    def preprocess_target(self, y):
        """Preprocess target variable"""
        if y.dtype == 'object' or y.dtype.name == 'category':
            self.target_encoder = LabelEncoder()
            y_processed = self.target_encoder.fit_transform(y)
            mlflow.log_param("target_classes", list(self.target_encoder.classes_))
        else:
            self.target_encoder = None
            y_processed = y
            mlflow.log_param("target_classes", list(np.unique(y)))
            
        return y_processed
    
    def get_feature_names_after_preprocessing(self, X):
        """Get feature names after preprocessing"""
        categorical_features = self.preprocessing_info['categorical_features']
        numerical_features = self.preprocessing_info['numerical_features']
        
        feature_names = []
        
        feature_names.extend(numerical_features)
        
        if categorical_features:
            if self.encoding_strategy == 'onehot':
                cat_transformer = self.preprocessor.named_transformers_['cat']
                if hasattr(cat_transformer, 'get_feature_names_out'):
                    cat_feature_names = cat_transformer.get_feature_names_out(categorical_features)
                else:
                    # for older sklearn: 
                    cat_feature_names = []
                    for i, feature in enumerate(categorical_features):
                        categories = cat_transformer.categories_[i][1:]  # drop first
                        for cat in categories:
                            cat_feature_names.append(f"{feature}_{cat}")
                feature_names.extend(cat_feature_names)
            else:  
                # ordinal encoding
                feature_names.extend(categorical_features)
        
        return feature_names
    
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
        
        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=False):

            mlflow.log_params(params)
            mlflow.set_tag("optuna_trial_number", trial.number)
            mlflow.set_tag("tuning_session_id", self.tuning_session_id)

            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', LogisticRegression(**params))
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

        return -score  # Optuna minimizes


    
    def train(self, X, y, test_size=0.2):
        """Train the Logistic Regression model with Optuna tuning and independent MLflow runs"""

        # Optional: generate a session ID to tag all runs for this tuning session
        import uuid
        self.tuning_session_id = str(uuid.uuid4())

        # Analyze and prepare data
        self.analyze_data(X, y)
        y_processed = self.preprocess_target(y)
        self.create_preprocessor(X)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y_processed, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y_processed
        )

        self.preprocessor.fit(X_train)
        self.feature_names = self.get_feature_names_after_preprocessing(X_train)

        # Run Optuna — each trial logs independently in `objective()`
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials,
        )

        # Store best params from Optuna
        self.best_params = study.best_params

        # Optionally log final trained model in its own run
        with mlflow.start_run(run_name="final_logistic_model") as run:
            self.run_id = run.info.run_id

            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("encoding_strategy", self.encoding_strategy)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("n_trials", self.n_trials)
            mlflow.log_param("cv_folds", self.cv_folds)
            mlflow.log_param("train_samples", X_train.shape[0])
            mlflow.log_param("val_samples", X_val.shape[0])
            mlflow.log_param("n_features_after_preprocessing", len(self.feature_names))
            mlflow.set_tag("tuning_session_id", self.tuning_session_id)

            for param_name, param_value in self.best_params.items():
                mlflow.log_param(f"best_{param_name}", param_value)

            mlflow.log_metric("best_objective_score", study.best_value)
            mlflow.log_metric("n_optimization_trials", len(study.trials))

            # Final model training
            final_params = {
                'solver': 'saga',
                'random_state': self.random_state,
                **self.best_params
            }
            self.pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', LogisticRegression(**final_params))
            ])
            self.pipeline.fit(X_train, y_train)
            self.model = self.pipeline.named_steps['classifier']

            mlflow.sklearn.log_model(
                self.pipeline,
                "model",
                signature=mlflow.models.infer_signature(X_train, y_train)
            )

            preprocessing_artifacts = {
                'target_encoder': self.target_encoder,
                'preprocessing_info': self.preprocessing_info,
                'feature_names': self.feature_names,
                'encoding_strategy': self.encoding_strategy
            }

            with open('preprocessing_artifacts.pkl', 'wb') as f:
                pickle.dump(preprocessing_artifacts, f)
            mlflow.log_artifact('preprocessing_artifacts.pkl')

            self._calculate_feature_importance()
            self._save_feature_importance_plot()
            mlflow.log_artifact('feature_importance.png')
            self._log_validation_metrics(X_val, y_val)

            print(f"Final model MLflow run: {self.run_id}")
            print(f"Best AUC: {-study.best_value:.4f}")

        return self
    
    def _calculate_feature_importance(self):
        """Calculate feature importance from logistic regression coefficients"""
        if self.model is None:
            return
        coef = np.abs(self.model.coef_[0])
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': coef
        }).sort_values('importance', ascending=False)
    
    def _log_validation_metrics(self, X_val, y_val):
        """Log validation metrics to MLflow"""
        y_pred_proba = self.pipeline.predict_proba(X_val)[:, 1]
        y_pred = self.pipeline.predict(X_val)
        
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
        
        print(f"Validation Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
    
    def _save_feature_importance_plot(self, top_n=20):
        """Save feature importance plot"""
        if self.feature_importance is None:
            return
        
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance (|Coefficient|)')
        plt.title(f'Top {top_n} Feature Importance - Logistic Regression')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def predict(self, X):
        """Make predictions on new data"""
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        return self.pipeline.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance dataframe"""
        return self.feature_importance


# COMMAND ----------

df = spark.table("samples.healthverity.claims_sample_synthetic").limit(100).toPandas()
df.head()

# COMMAND ----------

X = df.drop(columns=['patient_gender'])
y = df['patient_gender']

automl = LogisticRegressionWrapper(experiment_name=experiment_location, n_trials=50, cv_folds=5)

automl.train(X, y, test_size=0.2)

# COMMAND ----------


