import joblib
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

bp_model_ab = joblib.load('3_Months_Models/Group 17/bp_model_ab.pkl')
bp_model_gb = joblib.load('3_Months_Models/Group 17/bp_model_gb.pkl')
bp_model_gnb = joblib.load('3_Months_Models/Group 17/bp_model_gnb.pkl')
bp_model_knn = joblib.load('3_Months_Models/Group 17/bp_model_knn.pkl')
bp_model_mnb = joblib.load('3_Months_Models/Group 17/bp_model_mnb.pkl')
bp_model_rf = joblib.load('3_Months_Models/Group 17/bp_model_rf.pkl')

bp_models = [bp_model_ab, bp_model_gb, bp_model_gnb, bp_model_knn, bp_model_mnb, bp_model_rf]

glucose_model_ab = joblib.load('3_Months_Models/Group 17/glucose_model_ab.pkl')
glucose_model_gb = joblib.load('3_Months_Models/Group 17/glucose_model_gb.pkl')
glucose_model_gnb = joblib.load('3_Months_Models/Group 17/glucose_model_gnb.pkl')
glucose_model_knn = joblib.load('3_Months_Models/Group 17/glucose_model_knn.pkl')
glucose_model_mnb = joblib.load('3_Months_Models/Group 17/glucose_model_mnb.pkl')
glucose_model_rf = joblib.load('3_Months_Models/Group 17/glucose_model_rf.pkl')

glucose_models = [glucose_model_ab, glucose_model_gb, glucose_model_gnb, glucose_model_knn, glucose_model_mnb, glucose_model_rf]

systolic_bp = 0
diastolic_bp = 0
blood_glucose = 0

df = pd.read_csv('3_months_model_accuracies.csv')
group_17 = df[df['Group'] == 'Group 17']

bp_knn_acc = group_17[group_17['Model'] == 'BP_KNN']['Accuracy'].values[0]
bp_gb_acc = group_17[group_17['Model'] == 'BP_GB']['Accuracy'].values[0]
bp_ab_acc = group_17[group_17['Model'] == 'BP_AB']['Accuracy'].values[0]
bp_rf_acc = group_17[group_17['Model'] == 'BP_RF']['Accuracy'].values[0]
bp_gnb_acc = group_17[group_17['Model'] == 'BP_GNB']['Accuracy'].values[0]
bp_mnb_acc = group_17[group_17['Model'] == 'BP_MNB']['Accuracy'].values[0]

bp_accuracies = [bp_knn_acc, bp_gb_acc, bp_ab_acc, bp_rf_acc, bp_gnb_acc, bp_mnb_acc]

glucose_knn_acc = group_17[group_17['Model'] == 'Glucose_KNN']['Accuracy'].values[0]
glucose_rf_acc = group_17[group_17['Model'] == 'Glucose_RF']['Accuracy'].values[0]
glucose_gb_acc = group_17[group_17['Model'] == 'Glucose_GB']['Accuracy'].values[0]
glucose_ab_acc = group_17[group_17['Model'] == 'Glucose_AB']['Accuracy'].values[0]
glucose_gnb_acc = group_17[group_17['Model'] == 'Glucose_GNB']['Accuracy'].values[0]
glucose_mnb_acc = group_17[group_17['Model'] == 'Glucose_MNB']['Accuracy'].values[0]

glucose_accuracies = [glucose_knn_acc, glucose_rf_acc, glucose_gb_acc, glucose_ab_acc, glucose_gnb_acc, glucose_mnb_acc]

mappings = pd.read_csv('mappings.csv')

def bp_ensemble(systolic_bp: float, diastolic_bp: float) -> str:
    predictions = {}
    for i in range(len(bp_models)):
        prediction = bp_models[i].predict([[systolic_bp, diastolic_bp]])
        if prediction[0] in predictions:
            predictions[prediction[0]] += bp_accuracies[i]
        else:
            predictions[prediction[0]] = bp_accuracies[i]
        ensemble_prediction = max(predictions, key=predictions.get)
    data = mappings[mappings['number'] == ensemble_prediction]
    return data['name'].values[0]

def glucose_ensemble(blood_glucose: float) -> str:
    predictions = {}
    for i in range(len(glucose_models)):
        prediction = glucose_models[i].predict([[blood_glucose]])
        if prediction[0] in predictions:
            predictions[prediction[0]] += glucose_accuracies[i]
        else:
            predictions[prediction[0]] = glucose_accuracies[i]
        ensemble_prediction = max(predictions, key=predictions.get)
    data = mappings[mappings['number'] == ensemble_prediction]
    return data['name'].values[0]