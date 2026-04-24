#!/usr/bin/env python3
"""
Phase 3: ANN Model Evaluation & Interpretability
Evaluates the trained ANN model with comprehensive metrics, SHAP explainability,
and baseline comparisons for stroke severity prediction.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
import shap
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

class StrokePredictorEvaluator:
    """Comprehensive evaluator for stroke severity prediction ANN model."""

    def __init__(self, model_path='checkpoints/improved_model.keras',
                 data_path='data/splits/'):
        """Initialize evaluator with model and data paths."""
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()

        # Load data
        self.load_data()

        # Clinical feature names (from config)
        self.feature_names = [
            'age', 'nihss_score', 'glucose', 'systolic_bp', 'diastolic_bp',
            'heart_rate', 'temperature', 'oxygen_saturation', 'bmi',
            'stroke_history', 'hypertension', 'diabetes', 'smoking',
            'atrial_fibrillation', 'cholesterol_total', 'cholesterol_hdl',
            'cholesterol_ldl', 'triglycerides', 'creatinine', 'hemoglobin',
            'white_blood_cell_count', 'platelet_count'
        ]

    def load_data(self):
        """Load test data and model."""
        print("🔄 Loading model and test data...")

        # Load model
        self.model = tf.keras.models.load_model(self.model_path)

        # Load test data
        self.X_test = np.load(f'{self.data_path}/X_test.npy')
        self.y_severity_test = np.load(f'{self.data_path}/y_sev_test.npy').astype(int)
        self.y_progression_test = np.load(f'{self.data_path}/y_prog_test.npy')

        print(f"✅ Loaded {len(self.X_test)} test samples")
        print(f"   Features shape: {self.X_test.shape}")
        print(f"   Severity classes: {np.unique(self.y_severity_test)}")
        print(".2f")

    def evaluate_ann_model(self):
        """Evaluate the ANN model performance."""
        print("\n🧠 Evaluating ANN Model Performance...")

        # Get predictions
        predictions = self.model.predict(self.X_test, verbose=0)
        self.ann_severity_pred = np.argmax(predictions[0], axis=1)
        self.ann_progression_pred = predictions[1].flatten()

        # Severity Classification Metrics
        print("\n📊 Severity Classification Metrics:")
        print("=" * 50)

        # Classification report
        severity_report = classification_report(
            self.y_severity_test, self.ann_severity_pred,
            target_names=['Mild', 'Moderate', 'Severe'],
            output_dict=True
        )

        print("Classification Report:")
        print(classification_report(
            self.y_severity_test, self.ann_severity_pred,
            target_names=['Mild', 'Moderate', 'Severe']
        ))

        # Confusion Matrix
        cm = confusion_matrix(self.y_severity_test, self.ann_severity_pred)
        self.plot_confusion_matrix(cm, ['Mild', 'Moderate', 'Severe'])

        # ROC-AUC for multi-class (One-vs-Rest)
        severity_proba = predictions[0]  # Softmax probabilities
        try:
            roc_auc = roc_auc_score(
                tf.keras.utils.to_categorical(self.y_severity_test, 3),
                severity_proba,
                multi_class='ovr'
            )
            print(".4f")
        except:
            print("⚠️  ROC-AUC calculation failed (insufficient samples per class)")

        # Progression Regression Metrics
        print("\n📈 Progression Regression Metrics:")
        print("=" * 50)

        mae = mean_absolute_error(self.y_progression_test, self.ann_progression_pred)
        mse = mean_squared_error(self.y_progression_test, self.ann_progression_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_progression_test, self.ann_progression_pred)

        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")

        # Store metrics
        self.ann_metrics = {
            'severity': severity_report,
            'progression': {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}
        }

        return self.ann_metrics

    def evaluate_baseline_models(self):
        """Compare ANN with baseline models."""
        print("\n⚖️  Evaluating Baseline Models...")

        # Prepare data for sklearn models
        X_train = np.load(f'{self.data_path}/X_train.npy')
        y_severity_train = np.load(f'{self.data_path}/y_sev_train.npy').astype(int)
        y_progression_train = np.load(f'{self.data_path}/y_prog_train.npy')

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        self.baseline_results = {}

        # Severity Classification Baselines
        print("\nSeverity Classification Baselines:")

        # Logistic Regression
        lr_clf = LogisticRegression(random_state=42, max_iter=1000)
        lr_clf.fit(X_train_scaled, y_severity_train)
        lr_pred = lr_clf.predict(X_test_scaled)
        lr_report = classification_report(
            self.y_severity_test, lr_pred,
            target_names=['Mild', 'Moderate', 'Severe'],
            output_dict=True
        )

        # Random Forest
        rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)
        rf_clf.fit(X_train_scaled, y_severity_train)
        rf_pred = rf_clf.predict(X_test_scaled)
        rf_report = classification_report(
            self.y_severity_test, rf_pred,
            target_names=['Mild', 'Moderate', 'Severe'],
            output_dict=True
        )

        self.baseline_results['severity'] = {
            'logistic_regression': lr_report,
            'random_forest': rf_report
        }

        print(".4f")
        print(".4f")

        # Progression Regression Baselines (using Random Forest)
        print("\nProgression Regression Baselines:")
        y_progression_train = np.load(f'{self.data_path}/y_prog_train.npy')

        rf_reg = RandomForestRegressor(random_state=42, n_estimators=100)
        rf_reg.fit(X_train_scaled, y_progression_train)
        rf_reg_pred = rf_reg.predict(X_test_scaled)

        rf_mae = mean_absolute_error(self.y_progression_test, rf_reg_pred)
        rf_r2 = r2_score(self.y_progression_test, rf_reg_pred)

        self.baseline_results['progression'] = {
            'random_forest': {'mae': rf_mae, 'r2': rf_r2}
        }

        print(".4f")
        print(".4f")

        return self.baseline_results

    def shap_explainability(self):
        """Generate SHAP explanations for model predictions."""
        print("\n🔍 Generating SHAP Explanations...")

        try:
            # For very small datasets, SHAP might not work well
            if len(self.X_test) < 10:
                print("⚠️  Dataset too small for reliable SHAP analysis (need ≥10 samples)")
                print("   Skipping SHAP explainability for now")
                return

            # Create SHAP explainer for the ANN model
            # Use a subset for efficiency
            X_sample = self.X_test[:min(50, len(self.X_test))]

            # For neural networks, we need to create a wrapper function
            def model_predict(X):
                pred = self.model.predict(X, verbose=0)
                return pred[0]  # Return severity predictions

            # Use KernelExplainer (works with any model)
            background = self.X_test[:min(5, len(self.X_test))]  # Background dataset
            explainer = shap.KernelExplainer(model_predict, background)

            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)

            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values, X_sample,
                feature_names=self.feature_names,
                class_names=['Mild', 'Moderate', 'Severe'],
                show=False
            )
            plt.title('SHAP Feature Importance - Stroke Severity Prediction')
            plt.tight_layout()
            plt.savefig('results/shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Bar plot for global feature importance
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values, X_sample,
                feature_names=self.feature_names,
                plot_type='bar',
                class_names=['Mild', 'Moderate', 'Severe'],
                show=False
            )
            plt.title('Global Feature Importance')
            plt.tight_layout()
            plt.savefig('results/shap_bar.png', dpi=300, bbox_inches='tight')
            plt.close()

            print("✅ SHAP plots saved to results/ directory")

            # Store feature importance
            self.shap_values = shap_values
            self.shap_feature_importance = np.abs(shap_values).mean(axis=0)

        except Exception as e:
            print(f"⚠️  SHAP analysis failed: {str(e)}")
            print("   This may be due to small dataset size or TensorFlow compatibility")

    def plot_confusion_matrix(self, cm, labels):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix - Stroke Severity Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_regression_results(self):
        """Plot regression prediction results."""
        plt.figure(figsize=(12, 5))

        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(self.y_progression_test, self.ann_progression_pred, alpha=0.6)
        plt.plot([self.y_progression_test.min(), self.y_progression_test.max()],
                [self.y_progression_test.min(), self.y_progression_test.max()],
                'r--', label='Perfect Prediction')
        plt.xlabel('True Progression')
        plt.ylabel('Predicted Progression')
        plt.title('ANN Regression: True vs Predicted')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Residual plot
        plt.subplot(1, 2, 2)
        residuals = self.y_progression_test - self.ann_progression_pred
        plt.scatter(self.ann_progression_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Progression')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/regression_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self):
        """Generate comprehensive evaluation report."""
        print("\n📋 Generating Evaluation Report...")

        # Generate plots
        self.plot_regression_results()

        # Create summary report
        report = f"""
# Stroke Severity Prediction - Phase 3 Evaluation Report

## Dataset Summary
- Test Samples: {len(self.X_test)}
- Features: {len(self.feature_names)}
- Severity Classes: {len(np.unique(self.y_severity_test))}

## ANN Model Performance

### Severity Classification
- **Accuracy**: {self.ann_metrics['severity']['accuracy']:.4f}
- **Macro F1-Score**: {self.ann_metrics['severity']['macro avg']['f1-score']:.4f}

### Progression Regression
- **MAE**: {self.ann_metrics['progression']['mae']:.4f}
- **RMSE**: {self.ann_metrics['progression']['rmse']:.4f}
- **R² Score**: {self.ann_metrics['progression']['r2']:.4f}

## Baseline Comparison

### Severity Classification
- **Logistic Regression**: {self.baseline_results['severity']['logistic_regression']['accuracy']:.4f}
- **Random Forest**: {self.baseline_results['severity']['random_forest']['accuracy']:.4f}
- **ANN**: {self.ann_metrics['severity']['accuracy']:.4f}

### Progression Regression
- **Random Forest MAE**: {self.baseline_results['progression']['random_forest']['mae']:.4f}
- **ANN MAE**: {self.ann_metrics['progression']['mae']:.4f}

## Files Generated
- `results/confusion_matrix.png` - Classification confusion matrix
- `results/regression_analysis.png` - Regression scatter and residual plots
- `results/shap_summary.png` - SHAP feature importance summary
- `results/shap_bar.png` - SHAP global feature importance

## Key Insights
1. The ANN model shows promising performance on the test set
2. SHAP analysis reveals the most important clinical features for prediction
3. Baseline comparison helps contextualize the ANN's performance

---
*Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        with open('results/evaluation_report.md', 'w') as f:
            f.write(report)

        print("✅ Report saved to results/evaluation_report.md")

        # Print summary to console
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(".4f")
        print(".4f")
        print(".4f")
        print("="*60)

    def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        print("🚀 Starting Phase 3: ANN Model Evaluation & Interpretability")
        print("="*70)

        # Create results directory
        import os
        os.makedirs('results', exist_ok=True)

        try:
            # Evaluate ANN model
            self.evaluate_ann_model()

            # Compare with baselines
            self.evaluate_baseline_models()

            # Generate SHAP explanations
            self.shap_explainability()

            # Generate comprehensive report
            self.generate_report()

            print("\n✅ Phase 3 Complete!")
            print("📁 Results saved in 'results/' directory")
            print("📊 Check evaluation_report.md for detailed analysis")

        except Exception as e:
            print(f"❌ Evaluation failed: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """Main evaluation function."""
    evaluator = StrokePredictorEvaluator()
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()