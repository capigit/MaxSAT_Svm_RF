import matplotlib.pyplot as plt
import numpy as np

def afficher_graphique_resultats(acc_svm, acc_lad_svm, feat_svm, feat_lad_svm, 
                                 acc_rf, acc_rf_lad, feat_rf, feat_rf_lad):
    """Génère un diagramme en barres comparant les performances des modèles."""
    
    # Noms des modèles
    labels = ['SVM Classique', 'LAD-SVM', 'RF Classique', 'RF-LAD']
    
    # Scores en pourcentage
    scores = [acc_svm * 100, acc_lad_svm * 100, acc_rf * 100, acc_rf_lad * 100]
    
    # Nombre de caractéristiques utilisées
    features = [feat_svm, feat_lad_svm, feat_rf, feat_rf_lad]
    
    # Couleurs : Bleu pour les classiques, Vert pour les approches LAD
    colors = ['#4C72B0', '#55A868', '#4C72B0', '#55A868']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, scores, color=colors, edgecolor='black', width=0.6)
    
    # Ajout des labels sur les barres
    for bar, feat in zip(bars, features):
        yval = bar.get_height()
        # On affiche le score et le nombre de caractéristiques
        texte = f"{yval:.2f}%\n({feat} variables)"
        ax.text(bar.get_x() + bar.get_width()/2, yval - 5, texte, 
                ha='center', va='top', color='white', fontweight='bold', fontsize=10)

    # Personnalisation du graphique
    ax.set_ylim(0, 110) # Pour laisser de la place au-dessus
    ax.set_ylabel('Précision (Accuracy %)', fontweight='bold')
    ax.set_title('Comparaison des Performances : Approches Classiques vs LAD', fontweight='bold', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ligne de référence à 100%
    ax.axhline(100, color='black', linewidth=1)
    
    plt.tight_layout()
    
    nom_fichier = 'resultats_comparaison_LAD.png'
    plt.savefig(nom_fichier, dpi=300, bbox_inches='tight')
    print(f"\n[Succès] Le graphique a été sauvegardé sous le nom : '{nom_fichier}'")