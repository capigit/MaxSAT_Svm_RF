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
    # Vérification : on doit avoir au moins autant de MSS que d'arbres K
    if len(liste_mss) < K:
        print(f"  -> Attention: Seulement {len(liste_mss)} MSS trouvés. Ajustement du nombre d'arbres (K={len(liste_mss)}).")
        K = len(liste_mss)
        
    # Ligne 8 de l'Algorithme 2 : Sélectionner aléatoirement K MSS sans remise
    mss_selectionnes = random.sample(liste_mss, K) 
    
    arbres = [] # Pour stocker notre forêt
    
    # Ligne 9 : Pour chaque arbre
    for i in range(K):
        # Ligne 10 : Échantillon Bootstrap
        X_boot, y_boot = resample(X_train, y_train, random_state=i)
        
        # Récupération du MSS assigné à cet arbre
        mss_i = mss_selectionnes[i]
        
        # Filtrage des données Bootstrap avec le MSS
        X_boot_mss = X_boot[:, mss_i]
        
        # Ligne 11 : Entraînement de l'arbre CART (DecisionTree) parfait
        arbre = DecisionTreeClassifier(random_state=i)
        arbre.fit(X_boot_mss, y_boot)
        
        # On sauvegarde l'arbre et le MSS qui lui correspond
        arbres.append((arbre, mss_i))
        
    # -- PHASE DE PRÉDICTION (VOTE MAJORITAIRE) --
    y_preds = np.zeros((X_test.shape[0], K))
    for i, (arbre, mss_i) in enumerate(arbres):
        # Chaque arbre fait sa prédiction en regardant uniquement SA partie des données
        y_preds[:, i] = arbre.predict(X_test[:, mss_i])
        
    # Vote majoritaire : on fait la moyenne et on arrondit (>= 0.5 devient 1, sinon 0)
    y_pred_final = (np.mean(y_preds, axis=1) >= 0.5).astype(int)
    
    return accuracy_score(y_test, y_pred_final)