# Projet LAD-ML : Sélection de variables par MaxSAT pour SVM et Forêt Aléatoire

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)
![ML](https://img.shields.io/badge/Domain-Interpretable%20ML-brightgreen)

Ce projet compare des modèles de classification **classiques** à des variantes guidées par une approche **LAD (Logical Analysis of Data)**, où la sélection de variables est formulée en **MaxSAT**.

Objectif : montrer qu’on peut conserver de bonnes performances en réduisant fortement le nombre de caractéristiques utilisées.

---

## Sommaire

- [1) Ce que fait le projet](#1-ce-que-fait-le-projet)
- [2) Structure du projet](#2-structure-du-projet)
  - [2.1 Détail des fichiers](#21-détail-des-fichiers)
- [3) Explication méthodologique (LAD + MaxSAT)](#3-explication-méthodologique-lad--maxsat)
- [4) Prérequis](#4-prérequis)
- [5) Installation](#5-installation)
- [6) Reproduire l’expérience](#6-reproduire-lexpérience)
- [7) Choisir le dataset](#7-choisir-le-dataset)
- [8) Paramètres importants](#8-paramètres-importants)
- [9) Interpréter les résultats](#9-interpréter-les-résultats)
- [10) Reproductibilité et bonnes pratiques](#10-reproductibilité-et-bonnes-pratiques)
- [11) Limites connues](#11-limites-connues)
- [12) Dépannage](#12-dépannage)
- [13) Dépendances](#13-dépendances)
- [14) Extensions possibles](#14-extensions-possibles)
- [15) Licence](#17-licence)

---

## 1) Ce que fait le projet

Le script principal :

1. Charge un dataset binaire (par défaut : `Breast Cancer` de scikit-learn, binarisé).
2. Sépare les données en train/test.
3. Évalue :
   - un **SVM classique**,
   - un **LAD-SVM** (SVM entraîné sur un sous-ensemble minimal de variables trouvé par MaxSAT),
   - une **Random Forest classique**,
   - une **RF-LAD** (chaque arbre utilise un MSS potentiellement différent).
4. Génère un graphique de comparaison (`resultats_comparaison_LAD.png`).

---

## 2) Structure du projet

- `main.py` : orchestrateur de l’expérience (chargement, entraînements, affichage résultats).
- `dataset_manager.py` : génération/chargement des jeux de données.
- `lad_solver.py` : cœur MaxSAT (calcul d’un MSS et génération de plusieurs MSS).
- `svm_comparator.py` : entraînement/évaluation SVM classique et LAD-SVM.
- `rf_comparator.py` : entraînement/évaluation RF classique et RF-LAD.
- `visualizer.py` : génération du graphique final.
- `requirements.txt` : dépendances Python.

### 2.1 Détail des fichiers

#### `main.py`

Rôle : point d’entrée du projet.

- Définit la fonction `run_project()`.
- Charge un dataset via `dataset_manager`.
- Fait le split train/test (`train_test_split`).
- Lance les 4 évaluations : SVM classique, LAD-SVM, RF classique, RF-LAD.
- Paramètre les hyperparamètres globaux (`K`, `N_S`, `random_state`).
- Envoie les résultats au module `visualizer` pour produire l’image finale.

Entrées principales : données `X, y` (issues de `dataset_manager`).
Sorties : affichage console + image `resultats_comparaison_LAD.png`.

#### `dataset_manager.py`

Rôle : fournir des datasets prêts à l’emploi.

Fonctions :

- `get_figure1_toy_data()` : mini dataset binaire pédagogique (très petit, vérification rapide).
- `get_synthetic_data(...)` : génère un dataset binaire synthétique, avec règle de classe contrôlée.
- `get_sklearn_breast_cancer_binarized()` : charge le dataset scikit-learn puis binarise chaque variable par rapport à sa moyenne.

Entrées : paramètres de génération (pour le synthétique).
Sorties : `X` (matrice), `y` (labels binaires).

#### `lad_solver.py`

Rôle : résoudre la sélection de variables sous forme **MaxSAT**.

Fonctions :

- `calculer_un_mss(X, y)` : calcule un seul MSS minimal.
  - Construit un WCNF (`WCNF`).
  - Ajoute les clauses strictes de séparabilité entre classes.
  - Ajoute les clauses souples de minimisation du nombre de variables.
  - Résout avec `RC2` et renvoie les indices de variables retenues.

- `generer_plusieurs_mss(X, y, nb_mss=10)` : génère plusieurs MSS différents.
  - Reprend le même encodage.
  - Après chaque solution, ajoute une clause bloquante pour forcer une nouvelle solution.
  - Renvoie une liste de MSS, utile pour RF-LAD.

Entrées : `X, y` binaires.
Sorties : liste d’indices (`mss_indices`) ou liste de listes (`liste_mss`).

#### `svm_comparator.py`

Rôle : entraîner et comparer SVM standard vs SVM restreint au MSS.

Fonctions :

- `evaluer_svm_classique(...)` : SVM linéaire sur toutes les variables.
- `evaluer_svm_lad(...)` : filtre `X_train/X_test` sur `mss_indices`, puis entraîne le même SVM linéaire.

Entrées : split train/test + MSS pour version LAD.
Sortie : `accuracy_score`.

#### `rf_comparator.py`

Rôle : entraîner et comparer RF standard vs RF-LAD.

Fonctions :

- `evaluer_rf_classique(...)` : `RandomForestClassifier` classique avec `K` arbres.
- `evaluer_rf_lad(...)` : implémentation type “forêt personnalisée” :
  - sélection de `K` MSS,
  - bootstrap par arbre,
  - entraînement d’un `DecisionTreeClassifier` par MSS,
  - vote majoritaire final sur l’ensemble de test.

Comportement important : si le nombre de MSS disponibles est inférieur à `K`, le code réduit automatiquement `K`.

Entrées : split train/test + liste de MSS + `K`.
Sortie : `accuracy_score` final.

#### `visualizer.py`

Rôle : produire un résumé visuel des performances.

Fonction :

- `afficher_graphique_resultats(...)` :
  - construit un diagramme en barres (SVM/RF, classique vs LAD),
  - affiche accuracy + nombre de variables dans chaque barre,
  - sauvegarde la figure dans `resultats_comparaison_LAD.png`.

Entrées : accuracies + tailles de jeux de variables.
Sortie : image PNG enregistrée sur disque.

#### `requirements.txt`

Rôle : figer les dépendances minimales du projet.

- `numpy` : calcul matriciel.
- `python-sat` : modélisation et résolution MaxSAT (WCNF/RC2).
- `scikit-learn` : modèles ML et métriques.
- `matplotlib` : visualisation.

---

## 3) Explication méthodologique (LAD + MaxSAT)

### 3.1 Idée LAD

On cherche un **MSS** (*Minimal Support Set*) : un plus petit ensemble de variables permettant de distinguer les classes positives et négatives.

### 3.2 Encodage MaxSAT

Dans `lad_solver.py` :

- **Clauses strictes (hard clauses)** :
  pour chaque paire `(x_pos, x_neg)`, on impose qu’au moins une variable sélectionnée les différencie.
- **Clauses souples (soft clauses)** :
  on pénalise la sélection de chaque variable pour favoriser les solutions minimales.

Le solveur RC2 (`python-sat`) optimise ces contraintes et retourne les indices de variables à garder.

### 3.3 RF-LAD

Pour la forêt LAD :

- on génère plusieurs MSS distincts (avec clauses bloquantes),
- on assigne un MSS à chaque arbre,
- chaque arbre est entraîné sur bootstrap + sous-espace de variables,
- prédiction finale par vote majoritaire.

---

## 4) Prérequis

- Python 3.10+ (testé avec Python 3.11)
- OS Linux/macOS/Windows
- `pip`

---

## 5) Installation

### Option A — environnement virtuel (recommandé)

```bash
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# .\\venv\\Scripts\\activate    # Windows (PowerShell)

pip install --upgrade pip
pip install -r requirements.txt
```

### Option B — environnement existant

Installez simplement :

```bash
pip install -r requirements.txt
```

---

## 6) Reproduire l’expérience

Depuis la racine du projet :

```bash
python3 main.py
```

Le script affiche les accuracies en console et crée :

- `resultats_comparaison_LAD.png`

---

## 7) Choisir le dataset

Dans `main.py`, vous pouvez activer l’un des trois datasets :

- `get_figure1_toy_data()` : mini-exemple pédagogique,
- `get_synthetic_data()` : données binaires synthétiques,
- `get_sklearn_breast_cancer_binarized()` : dataset réel (par défaut).

> Le code attend des variables binaires pour l’étape LAD. Le dataset breast cancer est binarisé par seuil à la moyenne de chaque feature.

---

## 8) Paramètres importants

Dans `main.py` :

- `test_size=0.3` : ratio test,
- `random_state=42` : reproductibilité du split,
- `K=10` : nombre d’arbres RF,
- `N_S=15` : nombre de MSS générés pour RF-LAD.

Dans `rf_comparator.py` :

- si moins de MSS que `K`, le nombre d’arbres est automatiquement réduit.

---

## 9) Interpréter les résultats

- **SVM Classique vs LAD-SVM** : compare précision et nombre de variables.
- **RF Classique vs RF-LAD** : compare précision avec stratégie de sous-espaces LAD.
- Le graphique indique à la fois l’accuracy et la quantité de variables utilisées.

Un bon scénario LAD est :

- accuracy proche du modèle classique,
- nombre de variables significativement réduit.

---

## 10) Reproductibilité et bonnes pratiques

Pour des résultats stables :

1. Fixer `random_state` partout où possible.
2. Conserver les mêmes versions de dépendances (voir `requirements.txt`).
3. Exécuter dans un environnement virtuel propre.
4. Reporter le dataset actif et les paramètres (`K`, `N_S`, `test_size`).

---

## 11) Limites connues

- Le coût MaxSAT peut augmenter rapidement avec la taille des données (paires pos/neg nombreuses).
- La binarisation simple par moyenne peut être sous-optimale selon les données.
- L’évaluation actuelle repose sur un split train/test unique (pas de validation croisée).

---

## 12) Dépannage

### `ModuleNotFoundError: No module named 'pysat'`

```bash
pip install python-sat
```

### Le script est lent

- Réduire `N_S` dans `main.py`.
- Tester avec `get_synthetic_data()` ou le dataset jouet.
- Réduire la taille du dataset.

### Pas de graphique généré

- Vérifier les permissions d’écriture dans le dossier projet.
- Vérifier l’installation de `matplotlib`.

---

## 13) Dépendances

Le projet utilise principalement :

- `numpy`
- `python-sat`
- `scikit-learn`
- `matplotlib`

Voir `requirements.txt` pour installation rapide.

---

## 14) Extensions possibles

- Validation croisée et métriques additionnelles (F1, AUC, rappel/précision).
- Binarisation plus robuste (quantiles, optimisation de seuils).
- Analyse de stabilité des MSS entre exécutions.
- Journalisation des expériences (CSV/JSON) pour comparaisons systématiques.

---

## 15) Licence

Ce projet est distribué sous licence **MIT**.

Voir le fichier `LICENSE` à la racine du dépôt.