from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.metrics import precision_recall_curve, confusion_matrix, auc
from sklearn import metrics
import numpy as np
import torch

def IoU_from_pr(prec, rec):
    # Calculate the IoU from the Precision and recall, see: https://tomkwok.com/posts/iou-vs-f1/
    return (prec * rec) / (prec + rec - prec*rec)


y_true = np.asarray([0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1], dtype=np.int32)
y_pred = np.asarray([0.3, 0.2, 0.6, 0.9, 0.7, 0.1, 0.0, 0.9, 0.5, 0.6, 0.3, 0.2, 0.1, 0.9, 0.4], dtype=np.float32)

# PR-Curve
# precs, recs, Threshs = precision_recall_curve(y_true[::1000],y_pred[:1000])

# for every value in y_pred, assume it is the threshold and calc corresponding precision / recall value -> save in precs, recs
precs, recs, Threshs = precision_recall_curve(y_true, y_pred)


# calculate AUC score here!
# P is the number of positives
# M is the number of samples
P = y_true.sum()
M = y_true.shape[0]
TPs = recs*P
FNs = (1-recs)*P
FPs = P*recs*(1/(precs) - 1)
TNs = M - TPs - FNs - FPs

FPR = FPs/(TNs+FPs)
TPR = recs

# Cannot use epsilon division, because than the sequence might not be monotonically increasing or decreasing anymore
# Instead, spot the nan values. Also: replace the first element with coordinates (1,1)
dx = np.diff(FPR)
mondecmask1 = np.r_[dx>0,False] # False
dx = np.diff(TPR)
mondecmask2 = np.r_[dx>0,False]
valid_mask = ~( np.isnan(FPR) | np.isnan(TPR) | mondecmask1 | mondecmask2 )
FPR = FPR[valid_mask]
TPR = TPR[valid_mask]
FPR[0] = 1
TPR[0] = 1 

AUC = metrics.roc_auc_score(y_true,y_pred)

result = {}
beta = 1
epsilon = 0.001

result["binary_AUC"] = AUC

F1s_vec =  (1 + beta**2)* (precs*recs) / (beta**2 * precs + recs + epsilon) 

best_ind = np.argmax(F1s_vec) 
opt_f1 = F1s_vec[best_ind]
opt_prec, opt_rec = precs[best_ind], recs[best_ind]
opt_IoU = IoU_from_pr(opt_prec, opt_rec)
opt_thresh = Threshs[best_ind]