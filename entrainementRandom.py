import random
import numpy as np
import matplotlib.pyplot as plt

from network import Network
from network_v2 import Network_2




# Initialisation

vitesseInit = 0.01
entrainementMax = 5000

regroupementTauxSucc = []
meanTauxSucc = []
stockageVitesses = []
qteMAx = 2000
net = Network_2(nbUsers=4,learning_rate=0.0005)
net.load_NN("NN_2Layer4-2_4U.npz")
nbUsers = net.nbUsers

#Boucle pour tester différentes valeurs de mon LR
#for vitesse in np.linspace(vitesseInit, 0.06, 6):
    #net.updateLR(vitesse)
#vitesse = round(vitesse, 2)

#stockageVitesses.append(vitesse)


tauxSuccesTot = []

#Calcul des taux pour les neurones
taux_N_eluSucces = np.zeros(nbUsers + 1)
taux_N_eluErreur = np.zeros(nbUsers + 1)
taux_N_oracle = np.zeros(nbUsers + 1)

#gravité erreur

gravites_Erreurs = []

for entrainement in range(entrainementMax + int(entrainementMax/10)):

    # État initial des quantités (ex: chaque user a entre 0 et 700 unités)
    quantites_reelles = np.random.randint(0, 700, size=nbUsers).astype(float)

    sommeQte= 700.0
    
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

    erreurTic = []


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

       
        
        taux_N_oracle[choix_oracle] += 1.0

        # 2b. On compare
        if choix_reseau == choix_oracle:
            #print(f"{prediction}")
            #print(f"{target}")
            is_success = 1
            total_successes += 1 
            success_history.append(is_success)
            taux_N_eluSucces[choix_reseau] += 1

        else: 
            is_success = 0
            taux_N_eluErreur[choix_reseau] += 1.0
 
            #print(f"{prediction}")
        
        total_attempts += 1
        classementInputs = net.get_Tab_erreur(inputs, nbUsers)
        #print(f"Les classements sont : {classementInputs}")
        erreurTic.append(classementInputs[choix_reseau])
        

        # 3b. On garde seulement les 'window_size' derniers résultats pour le taux glissant
        # if len(success_history) > window_size:
        #     success_history.pop(0)

        # 4b. Calcul des statistiques
        #taux_glissant = (sum(success_history) / len(success_history)) * 100
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


        if(freinage > 12): break

    gravites_Erreurs.append(np.mean(erreurTic))


    if entrainement % 100 == 99:  # tous les 1000 entrainements
        regroupementTauxSucc.append(tauxSuccesTot.copy())
        tauxSuccesTot = []

        


    #paliers = [int(entrainementMax * p) for p in [0.75,0.80,0.85,0.90]]

    # if entrainement in paliers:
    #     net.updateLR(newLR=net.lr / 1.25)
    #     print(f"Palier {entrainement} : nouveau LR = {net.lr}")

    tauxSuccesTot.append(taux_global)

        #print(f"tadaaaaa pour entrainement numero {entrainement}, taux de succès global : {taux_global}")



meanTauxSucc.append(round(np.mean(tauxSuccesTot),3))
    
print(f"Taux de succès moyen suite à {entrainement +1} entrainements : {np.mean(tauxSuccesTot)} pour un LR de {net.lr}"
      f"\n avec comme ecart-type {np.nanstd(tauxSuccesTot)} ")

net.save_NN("NN_2Layer4-2_4U.npz")

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

ax.set_title(f"Taux de succès par entrainement avec valeur moy en fin : {np.round(np.mean(p(x_smooth)),2)}")
ax.set_xlabel("Entrainements")
ax.set_ylabel("Taux de succès")
ax.legend()
ax.grid(True)



"""graphe sur les neurones """

fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)


resultat1 = taux_N_eluErreur / taux_N_oracle
print(f"NbE / Nb-devraitE : {resultat1}")

ax2.bar(range(len(resultat1)), resultat1, color="blue", alpha=0.7)
ax2.set_title("Nb fois elu par le NN / nb fois de l'oracle")
ax2.set_xlabel("Neurone")
ax2.set_ylabel("Taux")
ax2.grid(True, axis="y")



resultat2 = taux_N_eluSucces / taux_N_oracle
print(f"NbE_succes / Nb-devraitE : {resultat2}")

ax3.bar(range(len(resultat2)), resultat2, color="green", alpha=0.7)
ax3.set_title("Nb fois elu succès par le NN / nb fois de l'oracle")
ax3.set_xlabel("Neurone")
ax3.set_ylabel("Taux")
ax3.grid(True, axis="y")

"""graphe sur les erreurs"""
fig3, ax4 = plt.subplots(figsize=(12, 5))

ax4.plot(gravites_Erreurs, color="purple", marker="o", 
        markersize=2, label="Erreurs", alpha=0.5)

ax4.set_title(f"Evolution du degré de gravité des erreurs au cours des tics")
ax4.set_xlabel("Tics")
ax4.set_ylabel("Gravité")


plt.tight_layout()
#plt.savefig("graphe.pdf")
plt.show()