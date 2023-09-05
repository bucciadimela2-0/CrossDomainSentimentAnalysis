import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, cohen_kappa_score, accuracy_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')

pd.options.mode.chained_assignment = None

# Carica i dati da file
train_data_druglib = pd.read_csv('drugLibTrain_raw.tsv', sep='\t')
test_data_druglib = pd.read_csv('drugLibTest_raw.tsv', sep='\t')
train_data_drugscom = pd.read_csv('drugsComTrain_raw.tsv', sep='\t')
test_data_drugscom = pd.read_csv('drugsComTest_raw.tsv', sep='\t')

# Rimuovi le righe con valori float dalla colonna 'commentsReview'
train_data_drugscom = train_data_drugscom[train_data_drugscom['commentsReview'].apply(lambda x: isinstance(x, str))]
test_data_drugscom = test_data_drugscom[test_data_drugscom['commentsReview'].apply(lambda x: isinstance(x, str))]
train_data_druglib = train_data_druglib[train_data_druglib['commentsReview'].apply(lambda x: isinstance(x, str))]
test_data_druglib = test_data_druglib[test_data_druglib['commentsReview'].apply(lambda x: isinstance(x, str))]

# Applica la tokenizzazione alle recensioni e rimuovi le stopwords
stop_words = set(stopwords.words('english'))

train_data_druglib['commentsReview'] = train_data_druglib['commentsReview'].apply(lambda x: ' '.join([word.lower() for word in nltk.word_tokenize(x) if word not in string.punctuation and word not in stop_words]))
test_data_druglib['commentsReview'] = test_data_druglib['commentsReview'].apply(lambda x: ' '.join([word.lower() for word in nltk.word_tokenize(x) if word not in string.punctuation and word not in stop_words]))

train_data_drugscom['commentsReview'] = train_data_drugscom['commentsReview'].apply(lambda x: ' '.join([word.lower() for word in nltk.word_tokenize(x) if word not in string.punctuation and word not in stop_words]))
test_data_drugscom['commentsReview'] = test_data_drugscom['commentsReview'].apply(lambda x: ' '.join([word.lower() for word in nltk.word_tokenize(x) if word not in string.punctuation and word not in stop_words]))

# Definisci una funzione per etichettare le recensioni in base al rating
def label_sentiment(rating):
    if rating <= 4:
        return 'negativa'
    elif rating < 7:
        return 'neutrale'
    else:
        return 'positiva'

# Applica la funzione di etichettatura alle recensioni
train_data_druglib['sentiment'] = train_data_druglib['rating'].apply(label_sentiment)
test_data_druglib['sentiment'] = test_data_druglib['rating'].apply(label_sentiment)
train_data_drugscom['sentiment'] = train_data_drugscom['rating'].apply(label_sentiment)
test_data_drugscom['sentiment'] = test_data_drugscom['rating'].apply(label_sentiment)

# Dividi il dataset DrugLib in set di addestramento e test
X_druglib = train_data_druglib['commentsReview']
y_druglib = train_data_druglib['sentiment']



# Bilancia il dataset DrugLib utilizzando il campionamento stratificato
X_train_druglib, X_test_druglib, y_train_druglib, y_test_druglib = train_test_split(X_druglib, y_druglib, test_size=0.2, random_state=42, stratify=y_druglib)


# Dividi il dataset DrugsCom in set di addestramento e test
X_drugscom = train_data_drugscom['commentsReview']
y_drugscom = train_data_drugscom['sentiment']

# Bilancia il dataset DrugsCom utilizzando il campionamento stratificato
X_train_drugscom, X_test_drugscom, y_train_drugscom, y_test_drugscom = train_test_split(X_drugscom, y_drugscom, test_size=0.2, random_state=42, stratify=y_drugscom)


# Creare un oggetto CountVectorizer per il dataset DrugLib
vectorizer_druglib = CountVectorizer()

# Addestrare il vectorizer sui dati di addestramento di DrugLib e trasformare i dati
X_train_bow_druglib = vectorizer_druglib.fit_transform(X_train_druglib)
X_test_bow_druglib = vectorizer_druglib.transform(X_test_druglib)


# Creare un oggetto CountVectorizer per il dataset DrugsCom
vectorizer_drugscom = CountVectorizer()

# Addestrare il vectorizer sui dati di addestramento di DrugsCom e trasformare i dati
X_train_bow_drugscom = vectorizer_drugscom.fit_transform(X_train_drugscom)
X_test_bow_drugscom = vectorizer_drugscom.transform(X_test_drugscom)

# Verifica i nomi delle colonne nei dati di addestramento e test per DrugLib
print("Nomi delle colonne nei dati di addestramento di DrugLib:")
print(train_data_druglib.columns)
print("\nNomi delle colonne nei dati di test di DrugLib:")
print(test_data_druglib.columns)


# Verifica i nomi delle colonne nei dati di addestramento e test per DrugsCom
print("\nNomi delle colonne nei dati di addestramento di DrugsCom:")
print(train_data_drugscom.columns)
print("\nNomi delle colonne nei dati di test di DrugsCom:")
print(test_data_drugscom.columns)

# Verifica il numero di colonne (feature) in X_train_bow_druglib
num_features_train = X_train_bow_druglib.shape[1]

# Verifica il numero di colonne (feature) in X_test_bow_druglib
num_features_test = X_test_bow_druglib.shape[1]



# Crea un'istanza del modello Random Forest Classifier
rf_model_dl = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

rf_model_dc = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)


# Addestra il modello Random Forest su DrugLib
rf_model_druglib = rf_model_dl.fit(X_train_bow_druglib, y_train_druglib)


# Addestra il modello Random Forest su DrugsCom
rf_model_drugscom = rf_model_dc.fit(X_train_bow_drugscom, y_train_drugscom)

# Valuta il modello su DrugLib
y_pred_druglib = rf_model_dl.predict(X_test_bow_druglib)
report_druglib = classification_report(y_test_druglib, y_pred_druglib)
print("Classification Report for DrugLib:\n", report_druglib)

# Valuta il modello su DrugsCom
y_pred_drugscom = rf_model_dc.predict(X_test_bow_drugscom)
report_drugscom = classification_report(y_test_drugscom, y_pred_drugscom)
print("Classification Report for DrugsCom:\n", report_drugscom)

domains = ["Birth Control", "Depression", "Pain", "Anxiety", "Diabetes, Type 2"]


# Crea una lista per registrare i risultati
results = []

# Doppio ciclo per confrontare i domini
for domain_origin in domains:
    for domain_destination in domains:
        # Filtra i dati per il dominio di origine
        train_data_origin = train_data_drugscom[train_data_drugscom['condition'] == domain_origin]
        test_data_destination = test_data_drugscom[test_data_drugscom['condition'] == domain_destination]

        if len(train_data_origin) > 0 and len(test_data_destination) > 0:
            # Addestra il modello RandomForestClassifier sul dominio di origine
            rf_model_origin = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
            X_train_origin = vectorizer_drugscom.transform(train_data_origin['commentsReview'])
            y_train_origin = train_data_origin['sentiment']
            rf_model_origin.fit(X_train_origin, y_train_origin)

            # Valuta il modello sul dominio di destinazione
            X_test_destination = vectorizer_drugscom.transform(test_data_destination['commentsReview'])
            y_test_destination = test_data_destination['sentiment']
            y_pred_destination = rf_model_origin.predict(X_test_destination)

            # Calcola l'accuratezza
            accuracy_destination = accuracy_score(y_test_destination, y_pred_destination)

            # Calcola il Cohen's Kappa
            kappa_destination = cohen_kappa_score(y_test_destination, y_pred_destination)

            # Aggiungi i risultati alla lista
            results.append({"Domain Origin": domain_origin, "Domain Destination": domain_destination, "Accuracy": accuracy_destination, "Cohen's Kappa": kappa_destination})
        else:
            print(f"Nessun dato disponibile per la coppia di domini: {domain_origin} -> {domain_destination}")

# Crea un DataFrame dai risultati
results_df = pd.DataFrame(results)

# Stampa la tabella dei risultati
print(results_df)

