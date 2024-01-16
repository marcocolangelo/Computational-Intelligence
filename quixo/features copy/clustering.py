import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def main():
        # Carica i dati
    data = np.load("Computational-Intelligence\\quixo\\features\\won_test2.npy")

    # Appiattisci ogni matrice in un vettore di lunghezza 25
    features = data.reshape((data.shape[0], -1))

    # Ora 'features' è un array 2D dove ogni riga è un vettore di lunghezza 25
    # che rappresenta lo stato del gioco

    # Utilizziamo K-Means per il clustering
    kmeans = KMeans(n_clusters=3, random_state=0).fit(features)

    # Ora, possiamo utilizzare 'kmeans.labels_' per ottenere le etichette dei cluster per ciascuna partita
    labels = kmeans.labels_

    # Stampa ogni partita con la sua etichetta
    for i in range(len(data)):
        print(data[i])
        print(labels[i])


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
    main()