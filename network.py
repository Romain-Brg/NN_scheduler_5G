import numpy as np

def reLU(z):
    return np.maximum(0, z)

def reLU_prime(z):
    # Renvoie 1 pour les valeurs > 0, sinon 0
    return (z > 0).astype(float)


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_Prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def softmax(z):
    # On soustrait le max de z pour éviter l'explosion numérique (overflow)
    # z est un vecteur colonne (n, 1)
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z, axis=0)

class Network:
    def __init__(self, nbUsers, learning_rate=0.1):
        self.nbUsers = nbUsers
        self.lr = learning_rate
        self.nbLayers = 1
        
        # Dimensions
        self.input_size = nbUsers * 2
        self.hidden_size = nbUsers * 4
        self.output_size = nbUsers + 1
        
        # Initialisation des poids (Matrices) et biais (Vecteurs)
        # On utilise une petite variance pour aider la convergence
        self.W1 = np.random.randn(self.hidden_size, self.input_size) * 0.1
        self.b1 = np.zeros((self.hidden_size, 1))
        
        self.W2 = np.random.randn(self.output_size, self.hidden_size) * 0.1
        self.b2 = np.zeros((self.output_size, 1))

    def updateLR(self, newLR):
        self.lr = newLR

    def forward(self, x):
        # x doit être un vecteur colonne (input_size, 1)
        self.z1 = np.dot(self.W1, x) + self.b1
        self.a1 = reLU(self.z1)
        
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    def backward(self, x, y_true):
        """
        C'est ici que la magie de la Backpropagation opère.
        y_true est le vecteur donné par ton Oracle (ex: [0, 1, 0, 0])
        """
        m = 1 # On traite un exemple à la fois (Stochastic Gradient Descent)

        # 1. Erreur sur la couche de sortie (Delta 2)
        # Dérivée de la fonction de coût (MSE) : (a2 - y_true)
        #d_a2 = self.a2 - y_true
        #delta2 = d_a2 * sigmoid_Prime(self.z2)
        delta2 = self.a2 - y_true

        # 2. Erreur sur la couche cachée (Delta 1)
        # On propage l'erreur de la couche 2 vers la couche 1 via W2
        delta1 = np.dot(self.W2.T, delta2) * reLU_prime(self.z1)

        # 3. Calcul des gradients (combien on doit changer les poids)
        dW2 = np.dot(delta2, self.a1.T)
        db2 = delta2
        
        dW1 = np.dot(delta1, x.T)
        db1 = delta1

        # 4. Mise à jour des poids (Apprentissage)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train_step(self, inputs, target):
        # On transforme l'entrée en vecteur colonne pour NumPy
        x = np.array(inputs).reshape(-1, 1)
        y = np.array(target).reshape(-1, 1)
        
        # Passage avant
        self.forward(x)
        # Rétropropagation
        self.backward(x, y)

    def get_oracle_target(self, inputs, nbUsers):
    # On sépare les SNR et les Quantités
        snrs = inputs[0::2]  # Prend les indices 0, 2, 4, 6
        qtes = inputs[1::2]  # Prend les indices 1, 3, 5, 7

        target = np.zeros(nbUsers + 1) # [0, 0, 0, 0, 0]

        # Trouver les utilisateurs qui ont encore des données
        users_avec_data = [i for i, q in enumerate(qtes) if q > 0]

        if not users_avec_data:
            # CAS : Personne n'a de données -> Neurone de repos (le dernier)
            target[-1] = 1.0
        else:
            # CAS : On cherche le max SNR parmi ceux qui ont de la data
            best_user = users_avec_data[0]
            max_snr = snrs[best_user]

            for i in users_avec_data:
                if snrs[i] > max_snr:
                    max_snr = snrs[i]
                    best_user = i

            target[best_user] = 1.0

        return target