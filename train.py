import glob
import os

import pandas as pd
from sklearn import feature_extraction
from sklearn import naive_bayes


def load_txt_files(path):
    """
    Cette fonction ouvre tous les fichiers .txt contenus dans le dossier `path`
    et les retourne sous forme d'une liste.
    """
    file_names = []
    file_txt = []
    for file in glob.glob(os.path.join(path, '*.txt')):
    	file_names.append(file.split('\\')[1])
    	file_txt.append(open(file, encoding='utf-8').read().rstrip('\n'))
    return file_names, file_txt

# On commence par ouvrir les fichiers d'entraînement déjà annotés
train_pos_names, train_pos_txt  = load_txt_files(path='texts/train/positive')
train_neg_names, train_neg_txt = load_txt_files(path='texts/train/negative')

# On concatène les fichiers d'entraînement pour pouvoir entraîner notre modèle
train_names = train_pos_names + train_neg_names
train_txt = train_pos_txt + train_neg_txt

# On définit une liste qui définit le sentiment de chaque texte dans le corpus
y_train = [1] * len(train_pos_txt) + [0] * len(train_neg_txt)

# On va maintenant "vectoriser" les textes dans le corpus d'entraînement. Ceci
# va faire que X_train sera une matrice où les lignes seront des textes et les
# colonnes seront des mots. Chaque cellule dans X_train indiquera le nombre
# d'occurrences du mot dans le texte.
vectorizer = feature_extraction.text.CountVectorizer()
vectorizer.fit(train_txt)
X_train = pd.DataFrame(
	data=vectorizer.transform(train_txt).todense(),
	columns=vectorizer.get_feature_names(),
	index=train_names
)

print(X_train)

# On peut maintenant entraîner un modèle
model = naive_bayes.MultinomialNB()
model.fit(X_train, y_train)

# On ouvre maintenant les fichiers qui doivent être annotés
test_names, test_txt = load_txt_files(path='texts/test')
X_test = pd.DataFrame(
	data=vectorizer.transform(test_txt).todense(),
	columns=vectorizer.get_feature_names(),
	index=test_names
)

# Maintenant que le modèle est entraîné, on peut l'utiliser pour faire des
# prédictions
y_pred = pd.DataFrame(
	data=model.predict_proba(X_test),
	columns=['Negative', 'Positive'],
	index=test_names
)

print(y_pred)
