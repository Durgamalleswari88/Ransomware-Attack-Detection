import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Matplotlib

from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)  # Initialize the Flask app

@app.route('/')
def index():
    # Load the dataset
    data = pd.read_csv('data-set/Ransomware.csv', delimiter='|')
    
    # Data Preprocessing: Handle missing data if any
    data.fillna(0, inplace=True)
    
    # Drop non-numeric columns (e.g., 'Name', which contains string values)
    numeric_data = data.select_dtypes(include=['float64', 'int64'])

    # Define the features and target variable
    features = numeric_data.iloc[:, :-1].values  # All columns except the last one
    target = numeric_data.iloc[:, -1].values    # Last column is the target variable

    # Split data for training and testing
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.25, random_state=42)

    # Initialize and train KNN model
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(features_train, target_train)

    # Predictions and metrics for KNN
    knn_predict = knn_model.predict(features_test)
    knn_success_rate = 100 * f1_score(target_test, knn_predict, average='micro')

    # Initialize and train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(features_train, target_train)

    # Predictions and metrics for Random Forest
    rf_predict = rf_model.predict(features_test)
    rf_success_rate = 100 * f1_score(target_test, rf_predict, average='micro')

    # Generate Classification Report for KNN and Random Forest
    knn_classification_report = classification_report(target_test, knn_predict, output_dict=True)
    rf_classification_report = classification_report(target_test, rf_predict, output_dict=True)

    # Convert classification reports to DataFrame for easier rendering in template
    knn_classification_df = pd.DataFrame(knn_classification_report).transpose()
    rf_classification_df = pd.DataFrame(rf_classification_report).transpose()

    # Convert target_test to binary values for ROC curve compatibility
    target_test_bin = (target_test == 1).astype(int)

    # Generate ROC curve for KNN
    knn_probabilities = knn_model.predict_proba(features_test)[:, 1]
    fpr_knn, tpr_knn, _ = roc_curve(target_test_bin, knn_probabilities)
    roc_auc_knn = auc(fpr_knn, tpr_knn)

    # Generate ROC curve for Random Forest
    rf_probabilities = rf_model.predict_proba(features_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(target_test_bin, rf_probabilities)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    # Generate Precision-Recall curve for KNN
    precision_knn, recall_knn, _ = precision_recall_curve(target_test_bin, knn_probabilities)

    # Generate Precision-Recall curve for Random Forest
    precision_rf, recall_rf, _ = precision_recall_curve(target_test_bin, rf_probabilities)

    # Feature Importance for Random Forest
    feature_importances = pd.DataFrame({
        'Feature': numeric_data.columns[:-1],  # Exclude target column
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Generate Confusion Matrix for KNN
    knn_cm = confusion_matrix(target_test, knn_predict)

    # Generate Confusion Matrix for Random Forest
    rf_cm = confusion_matrix(target_test, rf_predict)

    # Function to save plots and return as base64 strings
    def save_plot_to_base64(fig):
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()

    # KNN and RF ROC Curves
    fig_knn_roc, ax_knn_roc = plt.subplots()
    ax_knn_roc.plot(fpr_knn, tpr_knn, color='blue', label=f'KNN (AUC = {roc_auc_knn:.2f})')
    ax_knn_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax_knn_roc.set_title('KNN ROC Curve')
    ax_knn_roc.set_xlabel('False Positive Rate')
    ax_knn_roc.set_ylabel('True Positive Rate')
    ax_knn_roc.legend()
    knn_roc_url = save_plot_to_base64(fig_knn_roc)

    fig_rf_roc, ax_rf_roc = plt.subplots()
    ax_rf_roc.plot(fpr_rf, tpr_rf, color='green', label=f'RF (AUC = {roc_auc_rf:.2f})')
    ax_rf_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax_rf_roc.set_title('Random Forest ROC Curve')
    ax_rf_roc.set_xlabel('False Positive Rate')
    ax_rf_roc.set_ylabel('True Positive Rate')
    ax_rf_roc.legend()
    rf_roc_url = save_plot_to_base64(fig_rf_roc)

    # KNN and RF Precision-Recall Curves
    fig_knn_pr, ax_knn_pr = plt.subplots()
    ax_knn_pr.plot(recall_knn, precision_knn, color='blue', label='KNN Precision-Recall Curve')
    ax_knn_pr.set_title('KNN Precision-Recall Curve')
    ax_knn_pr.set_xlabel('Recall')
    ax_knn_pr.set_ylabel('Precision')
    ax_knn_pr.legend()
    knn_pr_url = save_plot_to_base64(fig_knn_pr)

    fig_rf_pr, ax_rf_pr = plt.subplots()
    ax_rf_pr.plot(recall_rf, precision_rf, color='green', label='RF Precision-Recall Curve')
    ax_rf_pr.set_title('Random Forest Precision-Recall Curve')
    ax_rf_pr.set_xlabel('Recall')
    ax_rf_pr.set_ylabel('Precision')
    ax_rf_pr.legend()
    rf_pr_url = save_plot_to_base64(fig_rf_pr)

    # KNN and RF Confusion Matrices
    fig_knn_cm, ax_knn_cm = plt.subplots()
    disp_knn = ConfusionMatrixDisplay(confusion_matrix=knn_cm)
    disp_knn.plot(ax=ax_knn_cm)
    knn_plot_url = save_plot_to_base64(fig_knn_cm)

    fig_rf_cm, ax_rf_cm = plt.subplots()
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=rf_cm)
    disp_rf.plot(ax=ax_rf_cm)
    rf_plot_url = save_plot_to_base64(fig_rf_cm)

    # # Accuracy Graphs (Success Rates)
    # fig_knn_accuracy, ax_knn_accuracy = plt.subplots()
    # ax_knn_accuracy.bar(['KNN'], [knn_success_rate], color='blue', edgecolor='black')
    # ax_knn_accuracy.set_title('KNN Success Rate')
    # ax_knn_accuracy.set_ylim(0, 100)
    # knn_accuracy_url = save_plot_to_base64(fig_knn_accuracy)

    # fig_rf_accuracy, ax_rf_accuracy = plt.subplots()
    # ax_rf_accuracy.bar(['Random Forest'], [rf_success_rate], color='green', edgecolor='black')
    # ax_rf_accuracy.set_title('Random Forest Success Rate')
    # ax_rf_accuracy.set_ylim(0, 100)
    # rf_accuracy_url = save_plot_to_base64(fig_rf_accuracy)
    # Accuracy Graphs (Success Rates)
    # Accuracy Graphs (Success Rates)
    fig_knn_accuracy, ax_knn_accuracy = plt.subplots()
    ax_knn_accuracy.bar(['KNN'], [knn_success_rate], color='dodgerblue', width=0.5)
    ax_knn_accuracy.set_title('KNN Success Rate')
    ax_knn_accuracy.set_ylim(0, 100)
    ax_knn_accuracy.set_ylabel('Accuracy (%)')
    ax_knn_accuracy.set_xlabel('Model')
    knn_accuracy_url = save_plot_to_base64(fig_knn_accuracy)
    
    fig_rf_accuracy, ax_rf_accuracy = plt.subplots()
    ax_rf_accuracy.bar(['Random Forest'], [rf_success_rate], color='forestgreen', width=0.5)
    ax_rf_accuracy.set_title('Random Forest Success Rate')
    ax_rf_accuracy.set_ylim(0, 100)
    ax_rf_accuracy.set_ylabel('Accuracy (%)')
    ax_rf_accuracy.set_xlabel('Model')
    rf_accuracy_url = save_plot_to_base64(fig_rf_accuracy)



    # Return the results to the template
    return render_template('index.html',
                           knn_success_rate=knn_success_rate,
                           rf_success_rate=rf_success_rate,
                           knn_roc_url=knn_roc_url,
                           rf_roc_url=rf_roc_url,
                           knn_pr_url=knn_pr_url,
                           rf_pr_url=rf_pr_url,
                           knn_plot_url=knn_plot_url,
                           rf_plot_url=rf_plot_url,
                           knn_accuracy_url=knn_accuracy_url,
                           rf_accuracy_url=rf_accuracy_url,
                           knn_classification_report=knn_classification_df.to_html(classes='table table-striped'),
                           rf_classification_report=rf_classification_df.to_html(classes='table table-striped'),
                           top_features=feature_importances.to_html(classes='table table-striped'))

if __name__ == "__main__":
    app.run(debug=True)
