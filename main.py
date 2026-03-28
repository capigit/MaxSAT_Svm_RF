from sklearn.model_selection import train_test_split
import dataset_manager
import lad_solver
import svm_comparator
import rf_comparator
import visualizer

def run_project():
    print("      DÉMARRAGE DU PROJET LAD-ML          \n")
    
    X, y = dataset_manager.get_sklearn_breast_cancer_binarized()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    nb_features_total = X_train.shape[1]
    
    print(f"[*] Données d'entraînement : {X_train.shape[0]} exemples.")
    print(f"[*] Nombre de caractéristiques initiales : {nb_features_total}\n")
    
    print("--- 1. ÉVALUATION DES SVM ---")
    
    acc_svm_class = svm_comparator.evaluer_svm_classique(X_train, X_test, y_train, y_test)
    print(f"  -> [SVM Classique] Accuracy : {acc_svm_class * 100:.2f}% (utilise {nb_features_total} variables)")
    
    print("  -> Recherche d'un MSS avec MaxSAT en cours...")
    mss_unique = lad_solver.calculer_un_mss(X_train, y_train)
    nb_features_lad_svm = len(mss_unique)
    
    if mss_unique:
        acc_svm_lad = svm_comparator.evaluer_svm_lad(X_train, X_test, y_train, y_test, mss_unique)
        print(f"  -> [LAD-SVM] Accuracy : {acc_svm_lad * 100:.2f}% (utilise {nb_features_lad_svm} variables : {mss_unique})\n")
    else:
        print("  -> [Erreur] Aucun MSS trouvé pour le SVM.\n")
        acc_svm_lad, nb_features_lad_svm = 0, 0

    print("--- 2. ÉVALUATION DES FORÊTS ALÉATOIRES ---")
    K = 10
    N_S = 15
    
    acc_rf_class = rf_comparator.evaluer_rf_classique(X_train, X_test, y_train, y_test, K)
    print(f"  -> [RF Classique] Accuracy : {acc_rf_class * 100:.2f}% (K={K} arbres)")
    
    print(f"  -> Génération de {N_S} MSS différents avec MaxSAT (clauses bloquantes)...")
    liste_mss = lad_solver.generer_plusieurs_mss(X_train, y_train, nb_mss=N_S)
    
    if liste_mss:
        acc_rf_lad = rf_comparator.evaluer_rf_lad(X_train, X_test, y_train, y_test, liste_mss, K)
        taille_moyenne_mss = round(sum(len(mss) for mss in liste_mss) / len(liste_mss), 1)
        print(f"  -> [RF-LAD] Accuracy : {acc_rf_lad * 100:.2f}% (Vote majoritaire sur {K} arbres)")
        print(f"     (Taille moyenne d'un MSS utilisé par les arbres : {taille_moyenne_mss} variables)\n")
    else:
        print("  -> [Erreur] Aucun MSS trouvé pour la RF.\n")
        acc_rf_lad, taille_moyenne_mss = 0, 0

    print("--- 3. GÉNÉRATION DU GRAPHIQUE ---")
    visualizer.afficher_graphique_resultats(
        acc_svm_class, acc_svm_lad, nb_features_total, nb_features_lad_svm,
        acc_rf_class, acc_rf_lad, nb_features_total, f"~{taille_moyenne_mss}"
    )
    print("Terminé !")

if __name__ == '__main__':
    run_project()