from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

def calculer_un_mss(X, y):
    X_pos = X[y == 1]
    X_neg = X[y == 0]
    num_features = X.shape[1]
    
    wcnf = WCNF()
    
    for v_pos in X_pos:
        for v_neg in X_neg:
            clause = [k + 1 for k in range(num_features) if v_pos[k] != v_neg[k]]
            if clause:
                wcnf.append(clause)
            
    for k in range(num_features):
        wcnf.append([-(k + 1)], weight=1)
        
    mss_indices = []
    with RC2(wcnf) as solver:
        model = solver.compute()
        if model is not None:
            mss_indices = [val - 1 for val in model if val > 0]
    return mss_indices

def generer_plusieurs_mss(X, y, nb_mss=10):
    X_pos = X[y == 1]
    X_neg = X[y == 0]
    num_features = X.shape[1]
    
    wcnf = WCNF()
    
    for v_pos in X_pos:
        for v_neg in X_neg:
            clause = [k + 1 for k in range(num_features) if v_pos[k] != v_neg[k]]
            if clause:
                wcnf.append(clause)
            
    for k in range(num_features):
        wcnf.append([-(k + 1)], weight=1)
        
    liste_mss = []
    
    with RC2(wcnf) as solver:
        for _ in range(nb_mss):
            model = solver.compute()
            if model is None:
                break
            mss_actuel = [val - 1 for val in model if val > 0]
            liste_mss.append(mss_actuel)
            clause_bloquante = [-val for val in model if val > 0]
            solver.add_clause(clause_bloquante)
    return liste_mss