import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys


# Load Yandex/CERN's evaluation Python script from the input data
exec(open("evaluation.py").read())

# Agreement and correlation conditions
ks_cutoff  = 0.09
cvm_cutoff = 0.002

data_path = "/home/peters/test/kaggle_flavour/flavours-of-physics-start/tau_data/"
#data_path = "/home/peter/code/kaggle/flavor/"
methodname = sys.argv[1];
print ("arguments: "+str (sys.argv));


# Load the training/test data along with the check file samples
#train = pd.read_csv("../tau_data/training.csv")
#test  = pd.read_csv("../tau_data/test.csv")
#test_prediction  = pd.read_csv("test_prediction.csv")
check_agreement   = pd.read_csv(data_path+"check_agreement.csv")
check_correlation = pd.read_csv(data_path+"check_correlation.csv")
check_agreement_prediction   = pd.read_csv ("check_agreement_prediction__"+methodname+".csv")
check_correlation_prediction = pd.read_csv ("check_correlation_prediction__"+methodname+".csv")
training = pd.read_csv ("training_prediction__"+methodname+".csv")


print("\nEvaluating predictions\n")

# Agreement Test
ks = compute_ks(
    check_agreement_prediction[check_agreement['signal'].values == 0]['prediction'].values,
    check_agreement_prediction[check_agreement['signal'].values == 1]['prediction'].values,
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)

if ks<ks_cutoff:
    print("This passed the agreement test with ks=%0.6f<%0.3f" % (ks, ks_cutoff))
else:
    print("This failed the agreement test with ks=%0.6f>=%0.3f" % (ks, ks_cutoff))

labels = training["signal"]
predictions = training["prediction"]
auc = roc_auc_truncated (labels, predictions)

    
# Correlation Test
#correlation_probs = rf.predict_proba(check_correlation[good_features])[:,1]
#cvm = compute_cvm(correlation_probs, check_correlation['mass'])
cvm = compute_cvm(check_correlation_prediction['prediction'].values, check_correlation['mass'])
if cvm<cvm_cutoff:
    print("This passed the correlation test with CvM=%0.6f<%0.4f" % (cvm, cvm_cutoff))
else:
    print("This failed the correlation test with CvM=%0.6f>=%0.4f" % (cvm, cvm_cutoff))

print "AUC = ",auc    
