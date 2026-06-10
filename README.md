# 🔤 CNN — Reconnaissance de Chiffres & Lettres (0–9 + A–Z)

> Projet IA M1 — Dr Khalid Gaber

Un modèle CNN unifié capable de reconnaître **36 classes** : les chiffres 0–9 et les lettres A–Z, entraîné sur les datasets MNIST et EMNIST fusionnés.

---

## 📋 Table des matières

- [Aperçu](#aperçu)
- [Architecture du modèle](#architecture-du-modèle)
- [Datasets](#datasets)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Résultats](#résultats)
- [Fichiers générés](#fichiers-générés)
- [Structure du projet](#structure-du-projet)

---

## Aperçu

Ce projet implémente un réseau de neurones convolutif (CNN) pour la reconnaissance optique de caractères manuscrits. Le modèle traite des images 28×28 pixels en niveaux de gris et prédit parmi 36 classes possibles.

**Version v2 — Corrections anti-overfitting :**
- L2 regularization sur les couches Dense
- Dropout augmenté (0.60 / 0.50)
- EarlyStopping avec `restore_best_weights=True`
- ModelCheckpoint → sauvegarde automatique du meilleur modèle
- ReduceLROnPlateau plus agressif (`factor=0.3`, `patience=2`)
- Évaluation finale sur `best_model.keras` (pas le dernier epoch)

---

## Architecture du modèle

```
Input (28×28×1)
    ↓
Conv2D → BatchNorm → MaxPool → Dropout
    ↓
Conv2D → BatchNorm → MaxPool → Dropout
    ↓
Conv2D → BatchNorm → Dropout
    ↓
Flatten
    ↓
Dense (L2) → BatchNorm → Dropout (0.60)
    ↓
Dense (L2) → BatchNorm → Dropout (0.50)
    ↓
Output Dense (36 classes) — Softmax
```

---

## Datasets

| Dataset | Classes | Train | Test |
|---|---|---|---|
| MNIST | Chiffres 0–9 | 60 000 | 10 000 |
| EMNIST Letters | Lettres A–Z | 124 800 | 20 800 |

Les deux datasets sont **fusionnés et équilibrés** par classe pour éviter les biais. Techniques appliquées : rééchantillonnage, class weights, data augmentation.

**Téléchargement EMNIST :**
- Option A (recommandée) : fichier `.mat` officiel sur [nist.gov](https://www.nist.gov/itl/products-and-services/emnist-dataset) → placer `emnist-letters.mat` à la racine du projet
- Option B (fallback) : `tensorflow_datasets` (automatique si `.mat` absent)

---

## Installation

### Prérequis

- Python 3.8+
- Jupyter Notebook / JupyterLab

### Dépendances

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn Pillow scipy
# Optionnel (fallback EMNIST)
pip install tensorflow-datasets
```

---

## Utilisation

### 1. Notebook d'entraînement

Ouvrir et exécuter le notebook dans l'ordre :

```bash
jupyter notebook CNN_Reconnaissance_Caracteres_v2.ipynb
```

**Plan du notebook :**
1. Installation & imports
2. Chargement des données (MNIST + EMNIST)
3. Équilibrage et fusion des datasets
4. Prétraitement et augmentation
5. Construction du modèle CNN
6. Callbacks & entraînement
7. Évaluation & courbes d'apprentissage
8. Matrice de confusion & précision par classe
9. Test sur données bruitées
10. Sauvegarde & prédiction rapide

### 2. Interface graphique

```bash
python interface_reconnaissance.py
```

### 3. Prédiction rapide (API)

```python
from tensorflow import keras
import numpy as np

# Charger le modèle
model = keras.models.load_model('best_model.keras')

CLASSES = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]

def predict_char(model, img_array, top_k=5):
    """
    img_array : numpy array (28,28) float32 dans [0,1]
                fond NOIR, tracé BLANC
    """
    arr = img_array.astype('float32').reshape(1, 28, 28, 1)
    probs = model.predict(arr, verbose=0)[0]
    best_idx = int(np.argmax(probs))
    top_k_idxs = np.argsort(probs)[::-1][:top_k]
    return {
        'char'      : CLASSES[best_idx],
        'confidence': float(probs[best_idx]),
        'top_k'     : [(CLASSES[i], float(probs[i])) for i in top_k_idxs],
    }
```

---

## Résultats

| Métrique | Valeur |
|---|---|
| Classes | 36 (0–9 + A–Z) |
| Format d'entrée | 28×28 px, niveaux de gris |
| Framework | TensorFlow 2.21 |
| Robustesse | Testé sous bruit gaussien, sel-poivre et speckle |

---

## Fichiers générés

| Fichier | Description |
|---|---|
| `best_model.keras` | ✅ Meilleur modèle (sauvegardé par ModelCheckpoint) |
| `model_unified.keras` | Alias du meilleur modèle |
| `model_unified.h5` | Alias format HDF5 (compatibilité maximale) |
| `training_log.csv` | Log CSV complet de l'entraînement |
| `distribution_classes.png` | Distribution équilibrée des 36 classes |
| `apercu_donnees.png` | Aperçu des données par classe |
| `augmentation_exemples.png` | Exemples d'augmentation de données |
| `courbes_apprentissage.png` | Courbes Loss & Accuracy (train + validation) |
| `matrice_confusion.png` | Matrice de confusion (absolue + normalisée) |
| `precision_par_classe.png` | Précision par classe (barplot) |
| `robustesse_bruit.png` | Tests de robustesse au bruit |
| `emnist-letters.mat` | Dataset EMNIST (à télécharger séparément) |

---

## Structure du projet

```
CNN_Atelier/
├── CNN_Reconnaissance_Caracteres_v2.ipynb  # Notebook principal
├── interface_reconnaissance.py             # Interface graphique
├── best_model.keras                        # Modèle entraîné
├── training_log.csv                        # Logs d'entraînement
├── emnist-letters.mat                      # Dataset EMNIST
├── README.md                               # Ce fichier
└── *.png                                   # Visualisations générées
```

---

## Technologies

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow)
