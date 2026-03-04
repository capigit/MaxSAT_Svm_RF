from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def evaluer_svm_classique(X_train, X_test, y_train, y_test):
    svm_classique = SVC(kernel='linear')
    svm_classique.fit(X_train, y_train)
    y_pred = svm_classique.predict(X_test)
    return accuracy_score(y_test, y_pred)

def evaluer_svm_lad(X_train, X_test, y_train, y_test, mss_indices):
    # Découpage des données selon le MSS
    X_train_mss = X_train[:, mss_indices]
    X_test_mss = X_test[:, mss_indices]
    
    svm_lad = SVC(kernel='linear')
    svm_lad.fit(X_train_mss, y_train)
    y_pred = svm_lad.predict(X_test_mss)
    return accuracy_score(y_test, y_pred)