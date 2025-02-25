from sklearn.metrics import confusion_matrix
import numpy as np

def specificity(Y_test, Y_pred, n):

    spe = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn/ (tn + fp)
        spe.append(spe1)
    return spe