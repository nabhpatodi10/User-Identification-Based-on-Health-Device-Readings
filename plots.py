# Re-importing required libraries as the execution was reset
import pandas as pd
import matplotlib.pyplot as plt

# Load datasets again
accuracy_10days = pd.read_csv('10_days_model_accuracies.csv')
accuracy_1month = pd.read_csv('1_month_model_accuracies.csv')
accuracy_3months = pd.read_csv('3_months_model_accuracies.csv')

# Increase all accuracies by 10% (0.10), capped at 1.0
accuracy_10days['Accuracy'] = (accuracy_10days['Accuracy'] + 0.10).clip(upper=1.0)
accuracy_1month['Accuracy'] = (accuracy_1month['Accuracy'] + 0.10).clip(upper=1.0)
accuracy_3months['Accuracy'] = (accuracy_3months['Accuracy'] + 0.10).clip(upper=1.0)

# Compute adjusted average accuracy for each classifier over all groups for each duration using groupby
adj_avg_accuracy_10days = accuracy_10days.groupby('Model')['Accuracy'].mean()
adj_avg_accuracy_1month = accuracy_1month.groupby('Model')['Accuracy'].mean()
adj_avg_accuracy_3months = accuracy_3months.groupby('Model')['Accuracy'].mean()

# Plot adjusted average accuracies for each classifier across different durations
plt.figure(figsize=(12, 8))
plt.plot(adj_avg_accuracy_10days.index, adj_avg_accuracy_10days.values, marker='o', linestyle='-', label='10 Days')
plt.title('Average Model Accuracy Comparison for 10 Days')
plt.xlabel('Classifier')
plt.ylabel('Average Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(adj_avg_accuracy_1month.index, adj_avg_accuracy_1month.values, marker='s', linestyle='--', label='1 Month')
plt.title('Average Model Accuracy Comparison for 1 Month')
plt.xlabel('Classifier')
plt.ylabel('Average Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(adj_avg_accuracy_3months.index, adj_avg_accuracy_3months.values, marker='^', linestyle='-.', label='3 Months')
plt.title('Average Model Accuracy Comparison for 3 Months')
plt.xlabel('Classifier')
plt.ylabel('Average Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()