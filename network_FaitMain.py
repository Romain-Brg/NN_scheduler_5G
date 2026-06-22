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

class Network_Mano:
    def __init__(self, nbUsers, learning_rate=0.1):
        self.nbUsers = nbUsers
        self.lr = learning_rate
        self.nbLayers = 1
        
        # Dimensions
        self.input_size = nbUsers * 2
        self.hidden_size = nbUsers * 2 + 1
        self.output_size = nbUsers + 1
        
        # Initialisation des poids (Matrices) et biais (Vecteurs)
        # On utilise une petite variance pour aider la convergence
        self.W1 = np.random.randn(self.hidden_size, self.input_size) * 0.0
        self.b1 = np.zeros((self.hidden_size, 1))
    
        

        #user 1
        self.W1[0,0] = 3 #mkn
        self.W1[4,1] = 30 #buffer
        self.W1[8,1] = -15 

        #user 2
        self.W1[1,2] = 3 #mkn
        self.W1[5,3] = 30 #buffer
        self.W1[8,3] = -15

        #user 3
        self.W1[2,4] = 3 #mkn
        self.W1[6,5] = 30 #buffer
        self.W1[8,5] = -15

        #user 4
        self.W1[3,6] = 3 #mkn
        self.W1[7,7] = 30 #buffer
        self.W1[8,7] = -15

        self.W2 = np.random.randn(self.output_size, self.hidden_size) * 0.0
        self.b2 = np.zeros((self.output_size, 1))


        #neurone final 1
        self.W2[0,0] = 3
        self.W2[0,4] = 13

        self.W2[0,5] = -4
        self.W2[0,6] = -4
        self.W2[0,7] = -4

        self.W2[0,8] = -5

        #neurone final 2
        self.W2[1,1] = 3
        self.W2[1,5] = 13

        self.W2[1,4] = -4
        self.W2[1,6] = -4
        self.W2[1,7] = -4
        
        self.W2[1,8] = -5

        #neurone final 3
        self.W2[2,2] = 3
        self.W2[2,6] = 13
        
        self.W2[2,4] = -4
        self.W2[2,5] = -4
        self.W2[2,7] = -4

        self.W2[2,8] = -5

        #neurone final 4
        self.W2[3,3] = 3
        self.W2[3,7] = 13

        self.W2[3,4] = -4
        self.W2[3,5] = -4
        self.W2[3,6] = -4


        self.W2[3,8] = -5

        #neurone final 5
        self.W2[4,4] = -4
        self.W2[4,5] = -4
        self.W2[4,6] = -4
        self.W2[4,7] = -4
        self.W2[4,8] = 12


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
        x = les inputs
        y_true = oracle
        """
        m = 1 # On traite un exemple à la fois (Stochastic Gradient Descent)

        # 1. Erreur sur la couche de sortie (Delta 2)
        # Dérivée de la fonction de coût (MSE) : (a2 - y_true)
        """le cout C"""
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
    
    def get_Tab_erreur(self, inputs, nbUsers):
        resultat = np.empty(nbUsers + 1, dtype=int)        
        snrs = inputs[0::2]  # Prend les indices 0, 2, 4, 6
        qtes = inputs[1::2]  # Prend les indices 1, 3, 5, 7

        #print(f"les SNRS : {snrs} \n les qtes : {qtes}")

        if np.allclose(qtes, 0):
        # Classement par ordre de snrs (max → 0)
            ordre = np.argsort(snrs)[::-1]
            rangs = np.empty(nbUsers, dtype=int)
            rangs[ordre] = np.arange(nbUsers)
            resultat[:nbUsers] = rangs
            resultat[-1] = 0  # dernière case = 0

   
        else:
            avec_qtes = np.where(qtes > 0)[0]   # indices des users avec qtes
            sans_qtes = np.where(qtes == 0)[0]  # indices des users sans qtes

            # Classement des users avec qtes par ordre de snrs
            ordre_avec = avec_qtes[np.argsort(snrs[avec_qtes])[::-1]]

            # Classement des users sans qtes par ordre de snrs
            ordre_sans = sans_qtes[np.argsort(snrs[sans_qtes])[::-1]]

            # Ordre final : avec qtes → sans qtes
            ordre_final = np.concatenate([ordre_avec, ordre_sans])

            rangs = np.empty(nbUsers, dtype=int)
            rangs[ordre_final] = np.arange(nbUsers)
            resultat[:nbUsers] = rangs
            resultat[-1] = nbUsers  # dernière case = nbUsers

        return resultat
    
    def save_NN(self, filepath="poids.npz"):
        np.savez(filepath, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"Poids sauvegardé dans {filepath}")

    def load_NN(self, filepath="poids.npz"):
        data = np.load(filepath)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']

        print("poids chargés")


    def elagage(self):
        self.W1[(-1 <= self.W1 ) & (self.W1 <= 1)] = 0.0
        self.W2[(-1 <= self.W2 ) & (self.W2 <= 1)] = 0.0
    
    def elagage_negatif(self):
        self.W1[ (self.W1 <= 0)] = 0.0
        self.W2[ (self.W2 <= 0)] = 0.0

    def softMaxage_poids(self):

        """Les poids"""
        sumPositif = 0
        sumNegatif = 0
        #comptage
        w1_positifs = self.W1 > 0
        w1_negatifs = self.W1 < 0

        sumPositif = np.sum(self.W1[w1_positifs])
        sumNegatif = (-1) * np.sum(self.W1[w1_negatifs])

        print(sumPositif)
        print(sumNegatif)
            
        if sumPositif > 0:
            self.W1[w1_positifs] /= sumPositif

        if sumNegatif > 0:
            self.W1[w1_negatifs] /= sumNegatif

        sumPositif = 0
        sumNegatif = 0
        #comptage
        w2_positifs = self.W2 > 0
        w2_negatifs = self.W2 < 0

        sumPositif = np.sum(self.W2[w2_positifs])
        sumNegatif = (-1)* np.sum(self.W2[w2_negatifs])

        print(sumPositif)
        print(sumNegatif)
            
        if sumPositif > 0:
            self.W2[w2_positifs] /= sumPositif

        if sumNegatif > 0:
            self.W2[w2_negatifs] /= sumNegatif




        """Les biais"""

        b1_positifs = self.b1 > 0
        b1_negatifs = self.b1 < 0

        sumPositif = np.sum(self.b1[b1_positifs])
        sumNegatif = (-1) * np.sum(self.b1[b1_negatifs])

        print(sumPositif)
        print(sumNegatif)
            
        if sumPositif > 0:
            self.b1[b1_positifs] /= sumPositif

        if sumNegatif > 0:
            self.b1[b1_negatifs] /= sumNegatif

        b2_positifs = self.b2 > 0
        b2_negatifs = self.b2 < 0

        sumPositif = np.sum(self.b2[b2_positifs])
        sumNegatif = (-1) * np.sum(self.b2[b2_negatifs])

        print(sumPositif)
        print(sumNegatif)
            
        if sumPositif > 0:
            self.b2[b2_positifs] /= sumPositif

        if sumNegatif > 0:
            self.b2[b2_negatifs] /= sumNegatif


    