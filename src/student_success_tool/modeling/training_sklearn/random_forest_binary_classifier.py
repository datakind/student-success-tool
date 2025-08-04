# Databricks notebook source
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import optuna
from optuna.integration.mlflow import MLflowCallback
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from datetime import datetime
import pickle

# COMMAND ----------

# log everything to the same experiment
experiment_location = "/Workspace/Users/veena.ramesh@databricks.com/automl_kinda/automl_experiment"
try: 
  mlflow.create_experiment(experiment_location)
except: 
  pass


# COMMAND ----------

class RandomForestWrapper:
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
        self.scaler = None
        self.preprocessing_info = {}
        self.run_id = None
        self.model_uri = None
    
        self.experiment = mlflow.set_experiment(self.experiment_name)
        
    def analyze_data(self, X, y):
        """Automatic data analysis and preprocessing"""
        mlflow.log_param("n_samples", X.shape[0])
        mlflow.log_param("n_features", X.shape[1])

        categorical_features = []
        numerical_features = []
        
        for col in X.columns: 
            if X[col].dtype == 'object' or X[col].dtype.name == 'category': 
                categorical_features.append(col)
            else:
                numerical_features.append(col)

        self.preprocessing_info['categorical_feature_names'] = categorical_features
        self.preprocessing_info['numerical_feature_names'] = numerical_features
        
        mlflow.log_param("n_categorical_features", len(categorical_features))
        mlflow.log_param("n_numerical_features", len(numerical_features))
                    
    def preprocess_data(self, X, y):
        """Preprocess data optimally for Random Forest"""
        X_processed = X.copy()
        categorical_columns = self.preprocessing_info['categorical_feature_names']
        numerical_columns = self.preprocessing_info['numerical_feature_names']

        self.label_encoders = {} 
        for col in categorical_columns: 
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            self.label_encoders[col] = le

        # Scale numerical features as well (optional for RF but this can help with feature importance interpretation)
        # Note: we did not do this for LGBM
        if numerical_columns:
            self.scaler = StandardScaler()
            X_processed[numerical_columns] = self.scaler.fit_transform(X_processed[numerical_columns])
        else:
            self.scaler = None

        if y.dtype == 'object' or y.dtype.name == 'category': 
            self.target_encoder = LabelEncoder()
            y_processed = self.target_encoder.fit_transform(y)
            mlflow.log_param("target_classes", list(self.target_encoder.classes_))
        else: 
            self.target_encoder = None
            y_processed = y
            mlflow.log_param("target_classes", sorted(y.unique()))

        return X_processed, y_processed
    
    def objective(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective function for hyperparameter optimization"""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        if params['bootstrap']:
            params['max_samples'] = trial.suggest_float('max_samples', 0.5, 1.0)
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # TODO: update what score to optimize
        # across all models, I used AUC to optimize
        score = -roc_auc_score(y_val, y_pred_proba)  # Negative because optuna minimizes
        
        return score
    
    def train(self, X, y, test_size=0.2):
        """Main training pipeline with MLflow tracking"""
        with mlflow.start_run() as run:
            self.run_id = run.info.run_id
            
            mlflow.log_param("timestamp", datetime.now().isoformat())
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("n_trials", self.n_trials)
            mlflow.log_param("cv_folds", self.cv_folds)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("model_type", "RandomForest")
            
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
                      
            mlflc = MLflowCallback(
                tracking_uri="databricks",
                metric_name="roc_auc_val",
                create_experiment=False,
                mlflow_kwargs={
                    "experiment_id": self.experiment.experiment_id,
                    "nested": True
                }
            )
                    
            study = optuna.create_study(direction='minimize')
            study.optimize(
                lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
                n_trials=self.n_trials, 
                callbacks=[mlflc]
            )
            
            self.best_params = study.best_params
            for param_name, param_value in self.best_params.items():
                mlflow.log_param(f"best_{param_name}", param_value)
            
            mlflow.log_metric("best_objective_score", study.best_value)
            mlflow.log_metric("n_optimization_trials", len(study.trials))
            
            final_params = {
                'random_state': self.random_state,
                'n_jobs': -1,
                **self.best_params
            }
            
            self.model = RandomForestClassifier(**final_params)
            self.model.fit(X_train, y_train)
            
            mlflow.sklearn.log_model(
                self.model,
                "model",
                signature=mlflow.models.infer_signature(X_train, y_train)
            )
              
            self.feature_importance = pd.DataFrame({
                'feature': X_processed.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for i, (_, row) in enumerate(self.feature_importance.head(10).iterrows()):
                mlflow.log_metric(f"top_{i+1}_feature_importance", row['importance'])
                mlflow.log_param(f"top_{i+1}_feature_name", row['feature'])
            
            # TODO: add preprocessing artifacts that you need. I have added a few here: 
            preprocessing_artifacts = {
                'target_encoder': self.target_encoder,
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'preprocessing_info': self.preprocessing_info,
                'problem_type': 'binary_classification'
            }
            
            with open('preprocessing_artifacts.pkl', 'wb') as f:
                pickle.dump(preprocessing_artifacts, f)
            mlflow.log_artifact('preprocessing_artifacts.pkl')
            
            self._save_feature_importance_plot()
            mlflow.log_artifact('feature_importance.png')
            self._log_validation_metrics(X_val, y_val)
            
            cv_scores = cross_val_score(self.model, X_train, y_train, 
                                      cv=self.cv_folds, scoring='roc_auc')
            mlflow.log_metric("cv_mean_auc", cv_scores.mean())
            mlflow.log_metric("cv_std_auc", cv_scores.std())
            
            print(f"MLflow run: {self.run_id}")
            
        return self
    
    def _log_validation_metrics(self, X_val, y_val):
        """Log validation metrics to MLflow"""          
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
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
        
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance (Gini Impurity)')
        plt.title(f'Top {top_n} Feature Importance - Random Forest')
        plt.gca().invert_yaxis()
        
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{importance:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def predict(self, X):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X_processed = self._preprocess_new_data(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X):
        """Make probability predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X_processed = self._preprocess_new_data(X)
        return self.model.predict_proba(X_processed)
    
    def _preprocess_new_data(self, X):
        """Preprocess new data using saved encoders and scalers"""
        X_processed = X.copy()
        
        for col in self.preprocessing_info['categorical_feature_names']:
            if col in X_processed.columns:
                X_processed[col] = X_processed[col].astype(str)
                known_categories = set(self.label_encoders[col].classes_)
                X_processed[col] = X_processed[col].apply(
                    lambda x: x if x in known_categories else 'unknown'
                )
                
                if 'unknown' not in known_categories:
                    self.label_encoders[col].classes_ = np.append(
                        self.label_encoders[col].classes_, 'unknown'
                    )
                
                X_processed[col] = self.label_encoders[col].transform(X_processed[col])
        
        if self.scaler is not None:
            numerical_cols = self.preprocessing_info['numerical_feature_names']
            X_processed[numerical_cols] = self.scaler.transform(X_processed[numerical_cols])
        
        return X_processed



# COMMAND ----------

df = spark.table("samples.healthverity.claims_sample_synthetic").limit(100).toPandas()
df.head()

# COMMAND ----------

X = df.drop(columns=['patient_gender'])
y = df['patient_gender']

# COMMAND ----------

automl = RandomForestWrapper(
    experiment_name=experiment_location, 
    n_trials=50, 
    cv_folds=5,
    random_state=42
)

automl.train(X, y, test_size=0.2)
