from sklearn.model_selection import train_test_split
from datetime import datetime
from src import dataset_manager
from src import lad_solver
from src import rf_comparator
from src import svm_comparator
from src import visualizer

def run_project():
    print("      DÉMARRAGE DU PROJET LAD-ML          \n")
    
    # X, y = dataset_manager.get_figure1_toy_data()
    X, y = dataset_manager.get_sklearn_breast_cancer_binarized()
    # X, y = dataset_manager.get_synthetic_data()
    
    # 2. Séparation Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    nb_features_total = X_train.shape[1]
    
    print(f"Données d'entraînement : {X_train.shape[0]} exemples.")
    print(f"Nombre de caractéristiques initiales : {nb_features_total}\n")
    
    print("1. EVALUATION DES SVM")
    
    # SVM Classique
    acc_svm_class = svm_comparator.evaluer_svm_classique(X_train, X_test, y_train, y_test)
    print(f"[SVM Classique] Accuracy : {acc_svm_class * 100:.2f}% (utilise {nb_features_total} variables)")
    
    # Recherche d'un MSS pour le SVM
    print("Recherche d'un MSS avec MaxSAT en cours...")
    mss_unique = lad_solver.calculer_un_mss(X_train, y_train)
    nb_features_lad_svm = len(mss_unique)
    
    # LAD-SVM
    if mss_unique:
        acc_svm_lad = svm_comparator.evaluer_svm_lad(X_train, X_test, y_train, y_test, mss_unique)
        print(f"[LAD-SVM] Accuracy : {acc_svm_lad * 100:.2f}% (utilise {nb_features_lad_svm} variables : {mss_unique})\n")
    else:
        print("[Erreur] Aucun MSS trouvé pour le SVM.\n")
        acc_svm_lad, nb_features_lad_svm = 0, 0

    print("2. EVALUATION DES FORÊTS ALÉATOIRES")
    K = 10
    N_S = 15
    
    acc_rf_class = rf_comparator.evaluer_rf_classique(X_train, X_test, y_train, y_test, K)
    print(f"[RF Classique] Accuracy : {acc_rf_class * 100:.2f}% (K={K} arbres)")
    
    print(f"Génération de {N_S} MSS différents avec MaxSAT (clauses bloquantes)...")
    liste_mss = lad_solver.generer_plusieurs_mss(X_train, y_train, nb_mss=N_S)
    
    if liste_mss:
        acc_rf_lad = rf_comparator.evaluer_rf_lad(
            X_train, X_test, y_train, y_test, liste_mss, K, random_state=42
        )
        
        taille_moyenne_mss = round(sum(len(mss) for mss in liste_mss) / len(liste_mss), 1)
        print(f"[RF-LAD] Accuracy : {acc_rf_lad * 100:.2f}% (Vote majoritaire sur {K} arbres)")
        print(f"(Taille moyenne d'un MSS utilisé par les arbres : {taille_moyenne_mss} variables)\n")
    else:
        print("[Erreur] Aucun MSS trouvé pour la RF.\n")
        acc_rf_lad, taille_moyenne_mss = 0, 0

    print("3. GÉNÉRATION DU RAPPORT WEB")
    visualizer.generer_rapport_resultats({
        'date_generation': datetime.now().strftime('%d/%m/%Y %H:%M'),
        'dataset': {
            'nom': 'Breast Cancer Wisconsin binarisé',
            'nb_exemples': int(X.shape[0]),
            'nb_train': int(X_train.shape[0]),
            'nb_test': int(X_test.shape[0]),
            'nb_variables': int(nb_features_total),
        },
        'parametres': {
            'random_state': 42,
            'test_size': 0.3,
            'K': K,
            'N_S': N_S,
        },
        'modeles': [
            {
                'nom': 'SVM Classique',
                'famille': 'SVM',
                'accuracy': float(acc_svm_class),
                'variables': nb_features_total,
                'lad': False,
            },
            {
                'nom': 'LAD-SVM',
                'famille': 'SVM',
                'accuracy': float(acc_svm_lad),
                'variables': nb_features_lad_svm,
                'lad': True,
            },
            {
                'nom': 'RF Classique',
                'famille': 'Forêt aléatoire',
                'accuracy': float(acc_rf_class),
                'variables': nb_features_total,
                'lad': False,
            },
            {
                'nom': 'RF-LAD',
                'famille': 'Forêt aléatoire',
                'accuracy': float(acc_rf_lad),
                'variables': f'~{taille_moyenne_mss}',
                'lad': True,
            },
        ],
        'mss_svm': mss_unique,
        'mss_rf': liste_mss,
    })

if __name__ == '__main__':
    run_project()
