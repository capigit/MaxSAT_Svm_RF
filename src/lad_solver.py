from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

def calculer_un_mss(X, y):
    """Trouve un sous-ensemble minimal de caractéristiques (MSS) via MaxSAT."""
    X_pos = X[y == 1]
    X_neg = X[y == 0]
    num_features = X.shape[1]
    
    wcnf = WCNF()
    
    # Clauses strictes : Séparer chaque paire (Positif / Négatif)
    for v_pos in X_pos:
        for v_neg in X_neg:
            clause = [k + 1 for k in range(num_features) if v_pos[k] != v_neg[k]]
            if clause: # Sécurité pour éviter les clauses vides
                wcnf.append(clause)
            
    # Clauses souples : Minimisation du nombre de caractéristiques gardées
    for k in range(num_features):
        wcnf.append([-(k + 1)], weight=1)
        
    mss_indices = []
    with RC2(wcnf) as solver:
        model = solver.compute()
        if model is not None:
            mss_indices = [val - 1 for val in model if val > 0]
                    
    return mss_indices

def generer_plusieurs_mss(X, y, nb_mss=10):
    """Génère une liste de N_S ensembles de support minimaux différents."""
    X_pos = X[y == 1]
    X_neg = X[y == 0]
    num_features = X.shape[1]
    
    wcnf = WCNF()
    
    # Clauses strictes pour séparer les classes
    for v_pos in X_pos:
        for v_neg in X_neg:
            clause = [k + 1 for k in range(num_features) if v_pos[k] != v_neg[k]]
            if clause:
                wcnf.append(clause)
                
    # Clauses souples pour la minimalité
    for k in range(num_features):
        wcnf.append([-(k + 1)], weight=1)
        
    liste_mss = []
    
    with RC2(wcnf) as solver:
        for _ in range(nb_mss):
            model = solver.compute()
            if model is None:
                break
            
            # Extraction du MSS
            mss_actuel = [val - 1 for val in model if val > 0]
            liste_mss.append(mss_actuel)
            
            # Mettre au moins une de ces variables à Faux
            clause_bloquante = [-val for val in model if val > 0]
            solver.add_clause(clause_bloquante)
            
    return liste_mss