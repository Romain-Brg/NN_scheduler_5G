import random
import numpy as np
import matplotlib.pyplot as plt

from network import Network
from network_v2 import Network_2




# Initialisation
nbUsers = 6
vitesseInit = 0.01

regroupementTauxSucc = []
meanTauxSucc =[]
stockageVitesses = []
qteMAx = 2000
net = Network_2(nbUsers=6,learning_rate=0.01)


#Boucle pour tester différentes valeurs de mon LR
#for vitesse in np.linspace(vitesseInit, 0.06, 6):
    #net.updateLR(vitesse)
#vitesse = round(vitesse, 2)

#stockageVitesses.append(vitesse)


tauxSuccesTot = []

for entrainement in range(20000):

    # État initial des quantités (ex: chaque user a entre 0 et 700 unités)
    quantites_reelles = np.random.randint(0, 700, size=nbUsers).astype(float)

    sommeQte= np.sum(quantites_reelles)
    
    # quantites_reelles = []
    # for i in range(nbUsers):
    #     tmp = random.randint(0,qteMAx)
    #     quantites_reelles.append(tmp)
    #     qteMAx -= tmp
    #     if qteMAx < 0: qteMAx = 0

    # quantites_reelles = np.array(quantites_reelles)
    freinage = 0

    window_size = 100
    success_history = [] # Liste pour stocker les succès (1) et échecs (0)
    total_successes = 0
    total_attempts = 0


    for step in range(3000):
        # 1. Générer des SNR aléatoires (ex: entre 0 et 20)
        snrs_reels = np.random.uniform(0, 20, size=nbUsers)
        sumSNR = np.sum(snrs_reels)

        # 2. Normalisation pour le réseau (TRES IMPORTANT)
        snrs_norm = snrs_reels / 20.0
        qtes_norm = quantites_reelles / sommeQte

        # 3. Construire le vecteur d'entrée [S1, Q1, S2, Q2...]
        inputs = np.empty(nbUsers*2)
        inputs[0::2] = snrs_norm
        inputs[1::2] = qtes_norm

        # 4. Prédiction et Oracle
        target = net.get_oracle_target(inputs, nbUsers)

        prediction = net.forward(inputs.reshape(-1, 1))

        # 5. Apprentissage
        net.backward(inputs.reshape(-1, 1), target.reshape(-1, 1))


        """Calcul du taux de succèes"""

        # 1b. On récupère l'indice choisi par le réseau et celui de l'Oracle
        choix_reseau = np.argmax(prediction)
        choix_oracle = np.argmax(target)

        # 2b. On compare
        is_success = 1 if choix_reseau == choix_oracle else 0
        success_history.append(is_success)
        total_attempts += 1
        if is_success: total_successes += 1

        # 3b. On garde seulement les 'window_size' derniers résultats pour le taux glissant
        if len(success_history) > window_size:
            success_history.pop(0)

        # 4b. Calcul des statistiques
        taux_glissant = (sum(success_history) / len(success_history)) * 100
        taux_global = (total_successes / total_attempts) * 100

        # # 5b. Affichage périodique (tous les 50 tours par exemple)
        # if step % 50 == 0:
        #     print(f"Tour {step:4d} | "
        #           f"Précision (100 derniers): {taux_glissant:6.2f}% | "
        #           f"Précision Globale: {taux_global:6.2f}% | "
        #           f"Choix (Réseau vs Oracle): {choix_reseau} vs {choix_oracle}")

        """Fin calcul taux"""


        # 6. Action : On baisse la quantité de l'utilisateur choisi par l'ORACLE 
        # (ou par le réseau si tu es en mode test)
        choix = np.argmax(target)
        if choix < nbUsers: # Si ce n'est pas le neurone de repos
            quantites_reelles[choix] -= 10 # On consomme de la data
            if quantites_reelles[choix] < 0: quantites_reelles[choix] = 0

        #print(f"Step:{step} - Choix Oracle: {choix} - Choix NN :  {choix_reseau} - \n Qtes initiales normalisées : {qtes_norm} - SNR inital: {np.trunc(snrs_reels)} Qtes restantes: {quantites_reelles}")

        if(np.allclose(quantites_reelles, 0)):freinage += 1



        if(freinage >5): break


    if entrainement % 100 == 99:  # tous les 1000 entrainements
        regroupementTauxSucc.append(tauxSuccesTot.copy())
        tauxSuccesTot = []

    tauxSuccesTot.append(taux_global)

    #print(f"tadaaaaa pour entrainement numero {entrainement}, taux de succès global : {taux_global}")



meanTauxSucc.append(round(np.mean(tauxSuccesTot),3))
    
print(f"Taux de succès moyen suite à {entrainement +1} entrainements : {np.mean(tauxSuccesTot)} pour un LR de {net.lr}"
      f"\n avec comme ecart-type {np.nanstd(tauxSuccesTot)} ")



fig, ax = plt.subplots(figsize=(12, 5))

# Aplatir tous les taux en une seule liste
tous_les_taux = [val for groupe in regroupementTauxSucc for val in groupe]
x_global = range(len(tous_les_taux))

# Afficher tous les points
ax.plot(x_global, tous_les_taux, color="blue", marker="o", 
        markersize=2, label="Valeurs", alpha=0.5)

# Courbe de tendance recalculée tous les 1000 entrainements
couleurs = ["red", "green", "orange", "purple", "brown"]
offset = 0
for i, groupe in enumerate(regroupementTauxSucc):
    x_groupe = range(offset, offset + len(groupe))
    
    z = np.polyfit(x_groupe, groupe, 3)
    p = np.poly1d(z)
    x_smooth = np.linspace(x_groupe[0], x_groupe[-1], 300)
    ax.plot(x_smooth, p(x_smooth), color=couleurs[i % len(couleurs)],
            linestyle="--") #label=f"Tendance groupe {i+1} sigma : {np.nanstd(groupe)}
    
    offset += len(groupe)

ax.set_title(f"Taux de succès par entrainement avec valeur moy en fin : {np.mean(p(x_smooth))}")
ax.set_xlabel("Entrainements")
ax.set_ylabel("Taux de succès")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig("graphe.pdf")
plt.show()