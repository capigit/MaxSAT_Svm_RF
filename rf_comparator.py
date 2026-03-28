import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score

def evaluer_rf_classique(X_train, X_test, y_train, y_test, K=10):
    rf = RandomForestClassifier(n_estimators=K, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return accuracy_score(y_test, y_pred)

def evaluer_rf_lad(X_train, X_test, y_train, y_test, liste_mss, K=10):
    if len(liste_mss) < K:
        print(f"  -> Attention: Seulement {len(liste_mss)} MSS trouvés. Ajustement du nombre d'arbres (K={len(liste_mss)}).")
        K = len(liste_mss)
        
    mss_selectionnes = random.sample(liste_mss, K) 
    
    arbres = []
    
    for i in range(K):
        X_boot, y_boot = resample(X_train, y_train, random_state=i)
        mss_i = mss_selectionnes[i]
        X_boot_mss = X_boot[:, mss_i]
        arbre = DecisionTreeClassifier(random_state=i)
        arbre.fit(X_boot_mss, y_boot)
        arbres.append((arbre, mss_i))
        
    
    y_preds = np.zeros((X_test.shape[0], K))
    for i, (arbre, mss_i) in enumerate(arbres):
        y_preds[:, i] = arbre.predict(X_test[:, mss_i])
        
    y_pred_final = (np.mean(y_preds, axis=1) >= 0.5).astype(int)
    return accuracy_score(y_test, y_pred_final)