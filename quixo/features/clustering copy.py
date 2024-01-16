import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib

def clustering():
        # Carica i dati
    data = np.load("Computational-Intelligence\\quixo\\features\\won_test2.npy")
    data2 = np.load("Computational-Intelligence\\quixo\\features\\lost_test2.npy")

    #join the two data
    data = np.concatenate((data, data2), axis=0)

    # Appiattisci ogni matrice in un vettore di lunghezza 25
    features = data.reshape((data.shape[0], -1))

    # Ora 'features' è un array 2D dove ogni riga è un vettore di lunghezza 25
    # che rappresenta lo stato del gioco

    # Utilizziamo K-Means per il clustering
    kmeans = KMeans(n_clusters=2, random_state=0).fit(features)

    # ora vediamo se un nuovo stato è in uno dei cluster
    new_state = np.array([[1, -1, 1,-1, -1],
                         [-1, -1, -1, 1, 0],
                         [1, -1, 0, 0, 0],
                         [1, -1, -1, 0, 0],
                         [1, 0, 1, 0, -1]])
    new_state = new_state.reshape((1, -1))

    # vediamo se il nuovo stato è in uno dei cluster
    new_prediction = kmeans.predict(new_state)
    
    # Ora, possiamo utilizzare 'kmeans.labels_' per ottenere le etichette dei cluster per ciascuna partita
    labels = kmeans.labels_

    print(new_prediction)
    # Stampa ogni partita con la sua etichetta
    for i in range(len(data)):
        print(data[i])
        print(labels[i])

def svm():
    # Carica i dati
    data = np.load("Computational-Intelligence\\quixo\\features\\won_test2.npy")
    data2 = np.load("Computational-Intelligence\\quixo\\features\\lost_test2.npy")

    #Appiattisci ogni matrice in un vettore di lunghezza 25
    #print(data.shape)
    features = data.reshape((data.shape[0], -1))
    #print(features.shape)
    features2 = data2.reshape((data2.shape[0], -1))
    #print(features2.shape)

    # Metti insieme i dati e assegna correttamente le etichette 1 per i data e 0 per i data2
    features_all = np.concatenate((features, features2), axis=0)
    print(features_all.shape)
    labels = np.concatenate((np.ones(features.shape[0]), np.zeros(features2.shape[0])), axis=0)
    print(labels.shape)

    # Dividi i dati in training e test set
    X_train, X_test, y_train, y_test = train_test_split(features_all, labels, test_size=0.2, random_state=0)

    #Riduci la dimensionalità dei dati
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    #Addestra il classificatore
    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)

    # Salva il modello
    # ...

    # Salva il modello
    joblib.dump(clf, 'modello_svm.pkl')


    #Predici i dati di test
    y_pred = clf.predict(X_test)

    #Calcola l'accuratezza
    print(accuracy_score(y_test, y_pred))



# def convert_board_to_features(board : np.ndarray):
#     # Definiamo le caratteristiche che utilizzeremo
#     features = []

#     # Aggiungiamo la posizione dei pezzi X
#     for i in range(3):
#         features.append(board[i // 2][i % 2])

#     # Aggiungiamo la posizione dei pezzi O
#     for i in range(3):
#         features.append(-board[i // 2][i % 2])

#     # Aggiungiamo la distanza tra i pezzi X e O
#     for i in range(3):
#         for j in range(3):
#             features.append(abs(board[i // 2][i % 2] - board[j // 2][j % 2]))

#     # Aggiungiamo la forma dei gruppi di pezzi X e O
#     for i in range(3):
#         features.append(get_shape(board[i // 2]))

#     return features

# def get_shape(pieces):
#     # Definiamo una lista di possibili forme
#     possible_shapes = ["line", "column", "square", "none"]

#     # Controlliamo se il gruppo è una linea
#     if len(set(pieces)) == 1 and len(pieces) >= 3:
#         return "line"

#     # Controlliamo se il gruppo è una colonna
#     if np.all(pieces[::2] == pieces[0]) and len(pieces) >= 3:
#         return "column"

#     # Controlliamo se il gruppo è un quadrato
#     if np.all(pieces == pieces[0]) and len(pieces) >= 3:
#         return "square"

#     # Se il gruppo non ha una forma definita, restituiamo "none"
#     return "none"



if __name__ == '__main__':
    svm()