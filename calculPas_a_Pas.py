import numpy as np

from network import Network




net = Network(nbUsers=4,learning_rate=0.01)
net.load_NN("NN_1Layer_4_4U_real_V3.npz")
nbUsers = net.nbUsers

#print(f" etat intitial du NN : \n W1 :\n {net.W1} \n b1 : \n  {net.b1} \n W2 : \n  {net.W2} \n b2: \n {net.b2}")

quantites_reelles = np.array([0,0,0,0]).astype(float)
snrs_reels = np.random.uniform(0, 20, size=nbUsers)

snrs_norm = snrs_reels / 20.0
qtes_norm = quantites_reelles / 400
# 3. Construire le vecteur d'entrée [S1, Q1, S2, Q2...]
inputs = np.empty(nbUsers*2)
inputs[0::2] = snrs_norm
inputs[1::2] = qtes_norm
    


target = net.get_oracle_target(inputs,nbUsers=4)
print(f"inputs :\n {inputs}")
print(f"choix_oracle : {np.argmax(target)}")


prediction = net.forward(inputs.reshape(-1,1))

print(f"choix reseau : {np.argmax(prediction)} \n")

print(f"z1 = inputs * W1 + b1: \n {net.z1}\n")
print(f" activiation a1 des neurones du milieu (a1 = reLu(z1)) :\n {net.a1} \n \n")

print(f"z2 = a1*W2 + b2: \n{net.z2}\n")
print(f" activiation a1 des neurones de la dernière couche (a2 = reLu(z2)) :\n {net.a2}")