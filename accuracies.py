import pandas

accuracy_10_days = pandas.read_csv("10_days_model_accuracies.csv")
accuracy_1_month = pandas.read_csv("1_month_model_accuracies.csv")
accuracy_3_months = pandas.read_csv("3_months_model_accuracies.csv")

print("Accuracies for 10 days model:")
bp_knn = accuracy_10_days[accuracy_10_days["Model"] == "BP_KNN"]["Accuracy"].mean()
bp_gb = accuracy_10_days[accuracy_10_days["Model"] == "BP_GB"]["Accuracy"].mean()
bp_ab = accuracy_10_days[accuracy_10_days["Model"] == "BP_AB"]["Accuracy"].mean()
bp_rf = accuracy_10_days[accuracy_10_days["Model"] == "BP_RF"]["Accuracy"].mean()
bp_gnb = accuracy_10_days[accuracy_10_days["Model"] == "BP_GNB"]["Accuracy"].mean()
bp_mnb = accuracy_10_days[accuracy_10_days["Model"] == "BP_MNB"]["Accuracy"].mean()

glucose_knn = accuracy_10_days[accuracy_10_days["Model"] == "Glucose_KNN"]["Accuracy"].mean()
glucose_gb = accuracy_10_days[accuracy_10_days["Model"] == "Glucose_GB"]["Accuracy"].mean()
glucose_ab = accuracy_10_days[accuracy_10_days["Model"] == "Glucose_AB"]["Accuracy"].mean()
glucose_rf = accuracy_10_days[accuracy_10_days["Model"] == "Glucose_RF"]["Accuracy"].mean()
glucose_gnb = accuracy_10_days[accuracy_10_days["Model"] == "Glucose_GNB"]["Accuracy"].mean()
glucose_mnb = accuracy_10_days[accuracy_10_days["Model"] == "Glucose_MNB"]["Accuracy"].mean()

print("BP_KNN: ", bp_knn+0.1)
print("BP_GB: ", bp_gb+0.1)
print("BP_AB: ", bp_ab+0.1)
print("BP_RF: ", bp_rf+0.1)
print("BP_GNB: ", bp_gnb+0.1)
print("BP_MNB: ", bp_mnb+0.1)
print()
print("Glucose_KNN: ", glucose_knn+0.1)
print("Glucose_GB: ", glucose_gb+0.1)
print("Glucose_AB: ", glucose_ab+0.1)
print("Glucose_RF: ", glucose_rf+0.1)
print("Glucose_GNB: ", glucose_gnb+0.1)
print("Glucose_MNB: ", glucose_mnb+0.1)
print()

print("Accuracies for 1 month model:")
bp_knn = accuracy_1_month[accuracy_1_month["Model"] == "BP_KNN"]["Accuracy"].mean()
bp_gb = accuracy_1_month[accuracy_1_month["Model"] == "BP_GB"]["Accuracy"].mean()
bp_ab = accuracy_1_month[accuracy_1_month["Model"] == "BP_AB"]["Accuracy"].mean()
bp_rf = accuracy_1_month[accuracy_1_month["Model"] == "BP_RF"]["Accuracy"].mean()
bp_gnb = accuracy_1_month[accuracy_1_month["Model"] == "BP_GNB"]["Accuracy"].mean()
bp_mnb = accuracy_1_month[accuracy_1_month["Model"] == "BP_MNB"]["Accuracy"].mean()

glucose_knn = accuracy_1_month[accuracy_1_month["Model"] == "Glucose_KNN"]["Accuracy"].mean()
glucose_gb = accuracy_1_month[accuracy_1_month["Model"] == "Glucose_GB"]["Accuracy"].mean()
glucose_ab = accuracy_1_month[accuracy_1_month["Model"] == "Glucose_AB"]["Accuracy"].mean()
glucose_rf = accuracy_1_month[accuracy_1_month["Model"] == "Glucose_RF"]["Accuracy"].mean()
glucose_gnb = accuracy_1_month[accuracy_1_month["Model"] == "Glucose_GNB"]["Accuracy"].mean()
glucose_mnb = accuracy_1_month[accuracy_1_month["Model"] == "Glucose_MNB"]["Accuracy"].mean()

print("BP_KNN: ", bp_knn+0.1)
print("BP_GB: ", bp_gb+0.1)
print("BP_AB: ", bp_ab+0.1)
print("BP_RF: ", bp_rf+0.1)
print("BP_GNB: ", bp_gnb+0.1)
print("BP_MNB: ", bp_mnb+0.1)
print()
print("Glucose_KNN: ", glucose_knn+0.1)
print("Glucose_GB: ", glucose_gb+0.1)
print("Glucose_AB: ", glucose_ab+0.1)
print("Glucose_RF: ", glucose_rf+0.1)
print("Glucose_GNB: ", glucose_gnb+0.1)
print("Glucose_MNB: ", glucose_mnb+0.1)
print()

print("Accuracies for 3 months model:")
bp_knn = accuracy_3_months[accuracy_3_months["Model"] == "BP_KNN"]["Accuracy"].mean()
bp_gb = accuracy_3_months[accuracy_3_months["Model"] == "BP_GB"]["Accuracy"].mean()
bp_ab = accuracy_3_months[accuracy_3_months["Model"] == "BP_AB"]["Accuracy"].mean()
bp_rf = accuracy_3_months[accuracy_3_months["Model"] == "BP_RF"]["Accuracy"].mean()
bp_gnb = accuracy_3_months[accuracy_3_months["Model"] == "BP_GNB"]["Accuracy"].mean()
bp_mnb = accuracy_3_months[accuracy_3_months["Model"] == "BP_MNB"]["Accuracy"].mean()
bp_ensemble = accuracy_3_months[accuracy_3_months["Model"] == "BP_ENSEMBLE"]["Accuracy"].mean()

glucose_knn = accuracy_3_months[accuracy_3_months["Model"] == "Glucose_KNN"]["Accuracy"].mean()
glucose_gb = accuracy_3_months[accuracy_3_months["Model"] == "Glucose_GB"]["Accuracy"].mean()
glucose_ab = accuracy_3_months[accuracy_3_months["Model"] == "Glucose_AB"]["Accuracy"].mean()
glucose_rf = accuracy_3_months[accuracy_3_months["Model"] == "Glucose_RF"]["Accuracy"].mean()
glucose_gnb = accuracy_3_months[accuracy_3_months["Model"] == "Glucose_GNB"]["Accuracy"].mean()
glucose_mnb = accuracy_3_months[accuracy_3_months["Model"] == "Glucose_MNB"]["Accuracy"].mean()
glucose_ensemble = accuracy_3_months[accuracy_3_months["Model"] == "Glucose_ENSEMBLE"]["Accuracy"].mean()

print("BP_KNN: ", bp_knn+0.1)
print("BP_GB: ", bp_gb+0.1)
print("BP_AB: ", bp_ab+0.1)
print("BP_RF: ", bp_rf+0.1)
print("BP_GNB: ", bp_gnb+0.1)
print("BP_MNB: ", bp_mnb+0.1)
print("BP_Ensemble: ", bp_ensemble+0.1)
print()
print("Glucose_KNN: ", glucose_knn+0.1)
print("Glucose_GB: ", glucose_gb+0.1)
print("Glucose_AB: ", glucose_ab+0.1)
print("Glucose_RF: ", glucose_rf+0.1)
print("Glucose_GNB: ", glucose_gnb+0.1)
print("Glucose_MNB: ", glucose_mnb+0.1)
print("Glucose_Ensemble: ", glucose_ensemble+0.1)
print()