<div align="center">

# 🧠 LAD-ML
### Sélection de variables par MaxSAT pour SVM et Forêts aléatoires

[![Licence MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PySAT](https://img.shields.io/badge/solveur-PySAT%20RC2-orange.svg)](https://pysathq.github.io/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-f89b3d.svg)](https://scikit-learn.org/)
[![GitHub Pages](https://img.shields.io/badge/rapport-GitHub%20Pages-222.svg)](https://capigit.github.io/MaxSAT_Svm_RF/)

*Projet académique — Master 2 Intelligence Artificielle & Combinatoire — Encadrant : Jean Perrin*

**[Voir le rapport en ligne →](https://capigit.github.io/MaxSAT_Svm_RF/)**

</div>

---

## 📋 Table des matières

- [Présentation](#-présentation)
- [Principe — qu'est-ce que le LAD ?](#-principe--quest-ce-que-le-lad-)
- [Algorithmes implémentés](#-algorithmes-implémentés)
  - [Formulation MaxSAT](#formulation-maxsat)
  - [LAD-SVM](#lad-svm)
  - [RF-LAD (Algorithme 2)](#rf-lad-algorithme-2)
- [Résultats](#-résultats)
- [Structure du projet](#-structure-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Jeux de données disponibles](#-jeux-de-données-disponibles)
- [Rapport web](#-rapport-web)
- [Licence](#-licence)

---

## 🔍 Présentation

Ce projet implémente et évalue expérimentalement l'approche **LAD** (*Logical Analysis of Data*) appliquée à deux familles de classificateurs : les **SVM** et les **forêts aléatoires**.

L'idée centrale est d'utiliser un solveur **MaxSAT** pour identifier un **MSS** (*Minimal Support Set*) : le plus petit sous-ensemble de variables binaires capable de séparer parfaitement les exemples positifs des exemples négatifs sur les données d'entraînement. Ce sous-ensemble est ensuite utilisé comme espace de features pour entraîner des modèles ML classiques.

La question de recherche est simple : **peut-on réduire massivement le nombre de variables utilisées sans dégrader l'accuracy ?**

Sur le jeu Breast Cancer binarisé (30 variables), le LAD trouve des ensembles de **11 variables** (-63 %) avec lesquels le LAD-SVM obtient une accuracy **supérieure** au SVM classique.

---

## 💡 Principe — qu'est-ce que le LAD ?

Le LAD (*Logical Analysis of Data*, Boros et al.) est une approche de classification basée sur la logique propositionnelle et l'optimisation combinatoire. Appliqué à la sélection de variables, il cherche un **MSS** : un ensemble minimal d'indices de features tel que, pour toute paire (exemple positif, exemple négatif), au moins une des features du MSS diffère entre les deux exemples.

En d'autres termes, le MSS **sépare les classes** avec un minimum de variables, sans perte d'information discriminante sur le jeu d'entraînement.

```
Données binaires → Paires (pos, neg) → Clauses de séparation → MaxSAT → MSS minimal
```

---

## ⚙️ Algorithmes implémentés

### Formulation MaxSAT

**Fichier :** `src/lad_solver.py`

Le problème de sélection minimale est encodé comme un programme **MaxSAT pondéré** (WCNF) :

| Type de clause | Contenu | Effet |
|---|---|---|
| **Stricte** (poids ∞) | Pour chaque paire (v⁺, v⁻) : clause des features qui diffèrent | Garantit la séparabilité |
| **Souple** (poids 1) | `[¬xₖ]` pour chaque feature k | Pénalise chaque variable gardée |

Le solveur RC2 (PySAT) minimise le nombre de variables sélectionnées tout en satisfaisant toutes les clauses strictes.

Pour énumérer N MSS distincts, une **clause bloquante** est ajoutée après chaque solution : elle force le solveur à trouver un MSS différent du précédent.

```python
# Clause bloquante : au moins une variable du MSS actuel doit être retirée
clause_bloquante = [-val for val in model if val > 0]
solver.add_clause(clause_bloquante)
```

### LAD-SVM

**Fichier :** `src/svm_comparator.py`

1. Calculer un MSS unique via MaxSAT (`calculer_un_mss`)
2. Filtrer les données d'entraînement et de test sur les colonnes du MSS
3. Entraîner un SVM à noyau linéaire sur le sous-espace réduit
4. Comparer l'accuracy avec le SVM classique (toutes variables)

### RF-LAD (Algorithme 2)

**Fichier :** `src/rf_comparator.py`

Implémentation d'une forêt aléatoire guidée par le LAD :

```
1. Générer N_S MSS distincts (N_S = 15 par défaut)
2. Sélectionner aléatoirement K MSS sans remise (K = 10 arbres)
3. Pour chaque arbre i de 1 à K :
   a. Tirer un échantillon bootstrap des données d'entraînement
   b. Filtrer cet échantillon sur le MSS i
   c. Entraîner un arbre CART complet (DecisionTreeClassifier)
4. Prédiction : vote majoritaire des K arbres
   (chaque arbre prédit sur les colonnes de son propre MSS)
```

La sélection des MSS est reproductible via un `random.Random(random_state)` dédié.

---

## 📊 Résultats

Sur le jeu **Breast Cancer Wisconsin binarisé** (569 exemples, 30 variables binaires)  
Split stratifié 70 % / 30 % — `random_state=42` — K=10 arbres — N_S=15 MSS

| Modèle | Accuracy | Variables utilisées | Réduction |
|---|---|---|---|
| SVM Classique | 97.08 % | 30 | — |
| **LAD-SVM** | **97.66 %** | **11** | **−63 %** |
| RF Classique | 95.32 % | 30 | — |
| RF-LAD | 94.15 % | ~11 | −63 % |

**Observations :**

- Le LAD-SVM utilise seulement 11 variables (indices : 1, 7, 9, 11, 15, 17, 18, 21, 23, 24, 28) et **gagne +0,58 point** d'accuracy par rapport au SVM complet. La réduction de dimensionnalité améliore ici la généralisation.
- La RF-LAD perd 1,17 point par rapport à la RF classique. Cette légère régression s'explique par le fait que les arbres CART complets (sans élagage) sur les données bootstrap peuvent sur-apprendre, et que le vote repose sur seulement 10 arbres.
- Dans les deux cas, le LAD réduit l'espace de features de **63 %** sans compromettre significativement les performances.

---

## 📁 Structure du projet

```text
MaxSAT_Svm_RF/
├── main.py                 # Point d'entrée — lance le pipeline complet
├── requirements.txt        # Dépendances (numpy, python-sat, scikit-learn)
├── LICENSE                 # Licence MIT
├── src/                    # Package Python
│   ├── __init__.py
│   ├── dataset_manager.py  # Chargement et binarisation des données
│   ├── lad_solver.py       # Formulation MaxSAT et génération des MSS
│   ├── svm_comparator.py   # SVM classique vs LAD-SVM
│   ├── rf_comparator.py    # RF classique vs RF-LAD
│   └── visualizer.py       # Génération du rapport web statique
└── reports/                # Rapport généré à chaque exécution
    ├── .nojekyll           # Permet le déploiement GitHub Pages
    ├── favicon.svg
    ├── index.html          # Page principale du rapport
    └── assets/
        ├── data.js         # Résultats injectés en JSON (window.LAD_REPORT_DATA)
        ├── script.js       # Rendu dynamique : graphiques, tableau, MSS
        └── style.css       # Mise en page responsive (880 px / 600 px)
```

---

## 🛠️ Installation

**Prérequis :** Python 3.11+

```bash
# Cloner le dépôt
git clone https://github.com/capigit/MaxSAT_Svm_RF.git
cd MaxSAT_Svm_RF

# Créer et activer un environnement virtuel
python3.11 -m venv venv
source venv/bin/activate       # Linux / macOS
# venv\Scripts\activate        # Windows

# Installer les dépendances
pip install -r requirements.txt
```

`requirements.txt` contient :

```
numpy
python-sat
scikit-learn
```

---

## ▶️ Utilisation

```bash
# Depuis la racine du projet, avec le venv activé :
python main.py
```

Sortie console attendue :

```
      DÉMARRAGE DU PROJET LAD-ML

Données d'entraînement : 398 exemples.
Nombre de caractéristiques initiales : 30

1. EVALUATION DES SVM
[SVM Classique] Accuracy : 97.08% (utilise 30 variables)
Recherche d'un MSS avec MaxSAT en cours...
[LAD-SVM] Accuracy : 97.66% (utilise 11 variables : [1, 7, 9, 11, 15, 17, 18, 21, 23, 24, 28])

2. EVALUATION DES FORÊTS ALÉATOIRES
[RF Classique] Accuracy : 95.32% (K=10 arbres)
Génération de 15 MSS différents avec MaxSAT (clauses bloquantes)...
[RF-LAD] Accuracy : 94.15% (Vote majoritaire sur 10 arbres)
(Taille moyenne d'un MSS utilisé par les arbres : 11.0 variables)

3. GÉNÉRATION DU RAPPORT WEB
[Succès] Le rapport web a été sauvegardé sous le nom : 'reports/index.html'
```

Le rapport est ensuite consultable en ouvrant `reports/index.html` dans un navigateur.

### Changer de jeu de données

Dans `main.py`, décommenter la ligne souhaitée :

```python
# X, y = dataset_manager.get_figure1_toy_data()       # 5 exemples, 5 variables
X, y = dataset_manager.get_sklearn_breast_cancer_binarized()  # par défaut
# X, y = dataset_manager.get_synthetic_data()          # 150 exemples, 15 variables
```

### Ajuster les paramètres

Dans `main.py` :

```python
K   = 10   # Nombre d'arbres dans la forêt LAD
N_S = 15   # Nombre de MSS énumérés par MaxSAT
```

Dans `train_test_split` :

```python
test_size=0.3, random_state=42, stratify=y
```

---

## 🗃️ Jeux de données disponibles

| Fonction | Description | Taille | Variables |
|---|---|---|---|
| `get_figure1_toy_data()` | Données jouet issues de la Figure 1 du cours | 5 exemples | 5 binaires |
| `get_synthetic_data()` | Données synthétiques (classe = somme des 3 premières features ≥ 2) | 150 exemples | 15 binaires |
| `get_sklearn_breast_cancer_binarized()` | Breast Cancer Wisconsin, binarisé par seuillage à la moyenne colonne | 569 exemples | 30 binaires |

La binarisation du Breast Cancer se fait feature par feature : `1` si la valeur est supérieure à la moyenne de la colonne, `0` sinon.

---

## 🌐 Rapport web

Chaque exécution regénère `reports/index.html` — un rapport statique sans dépendance externe :

- **En-tête** : nom du dataset, date et heure de génération
- **Métriques clés** : nombre d'exemples, variables initiales, variables LAD-SVM, taux de réduction
- **Conclusions automatiques** : comparaison LAD-SVM vs SVM, gain de compacité, écart RF-LAD
- **Graphiques CSS** : accuracy et variables utilisées par modèle (barres CSS pures, pas de canvas)
- **Tableau des résultats** : accuracy formatée, famille, variables
- **MSS LAD-SVM** : indices affichés sous forme de chips
- **MSS RF-LAD** : liste des 15 ensembles avec leur taille

Le dossier `reports/` contient un fichier `.nojekyll` et peut être déployé directement sur **GitHub Pages** (branche `gh-pages` ou dossier `docs/`).

---

## 📄 Licence

Ce projet est distribué sous licence [MIT](LICENSE).  
Copyright (c) 2026 capigit
