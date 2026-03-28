import matplotlib.pyplot as plt
import numpy as np

def afficher_graphique_resultats(acc_svm, acc_lad_svm, feat_svm, feat_lad_svm, 
                                 acc_rf, acc_rf_lad, feat_rf, feat_rf_lad):
    
    labels = ['SVM Classique', 'LAD-SVM', 'RF Classique', 'RF-LAD']
    
    scores = [acc_svm * 100, acc_lad_svm * 100, acc_rf * 100, acc_rf_lad * 100]
    
    features = [feat_svm, feat_lad_svm, feat_rf, feat_rf_lad]
    
    colors = ['#4C72B0', '#55A868', '#4C72B0', '#55A868']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, scores, color=colors, edgecolor='black', width=0.6)
    
    for bar, feat in zip(bars, features):
        yval = bar.get_height()

        texte = f"{yval:.2f}%\n({feat} variables)"
        ax.text(bar.get_x() + bar.get_width()/2, yval - 5, texte, 
                ha='center', va='top', color='white', fontweight='bold', fontsize=10)

    ax.set_ylim(0, 110)
    ax.set_ylabel('Précision (Accuracy %)', fontweight='bold')
    ax.set_title('Comparaison des Performances : Approches Classiques vs LAD', fontweight='bold', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax.axhline(100, color='black', linewidth=1)
    
    plt.tight_layout()
    
    nom_fichier = 'resultats.png'
    plt.savefig(nom_fichier, dpi=300, bbox_inches='tight')
    print(f"\n[Succès] Le graphique a été sauvegardé sous le nom : '{nom_fichier}'")