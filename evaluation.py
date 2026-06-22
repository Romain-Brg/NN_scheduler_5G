import random
import numpy as np
import matplotlib.pyplot as plt
import json
import os

from network import Network
from network_v2 import Network_2
from network_PF import Network_PF





# Initialisation

vitesseInit = 0.01
stepMax = 10000
regroupementTauxSucc = []
meanTauxSucc = []
stockageVitesses = []
net = Network(nbUsers=4,learning_rate=0.01)
net.load_NN("NN_1Layer_4_4U_real_v5_0_aux_grospoids.npz")
# net.elagage()
# net.save_NN("NN_1Layer_4_4U_real_V3_elague.npz")

nbUsers = net.nbUsers
nn_est_PF = False

if(net.__class__ == Network_PF):
    print("c bon")
    nn_est_PF = True
    fichier_erreurs = f"erreurs_PF_{net.nbLayers}midLayer{net.hidden_size}_{net.nbUsers}User_V3.json"

elif(net.__class__ == Network_2):
    fichier_erreurs = f"erreurs_{net.nbLayers}midLayer{net.hidden_1_size}-{net.hidden_2_size}_{net.nbUsers}User_V3.json"
elif(net.__class__ == Network):
    fichier_erreurs = f"erreurs_{net.nbLayers}midLayer{net.hidden_size}_{net.nbUsers}User_V3.json"






#Calcul des taux pour les neurones
taux_N_eluSucces = np.zeros(nbUsers + 1)
taux_N_eluErreur = np.zeros(nbUsers + 1)
taux_N_oracle = np.zeros(nbUsers + 1)

#gravité erreur

gravites_Erreurs = []

#stocckage erreurs

erreurs = []


#taux de bits transmis

tauxBits_transmis_Oracle = 0
tauxBits_transmis_NN = 0

tauxbits_transmis = []

#bits demandes

qteBits_tab_demandes = []
qteBits_tab_Transmis = []
qteBits_demandes = 0
qteBits_Transmis = 0

# État initial des quantités (ex: chaque user a entre 0 et 700 unités)
quantites_reelles = np.random.randint(0, 400, size=nbUsers).astype(float)
#quantites_reelles = np.array([0,0,0,0]).astype(float)
sommeQte= 400.0
qteBits_demandes += np.sum(quantites_reelles)
qteBits_tab_demandes.append(qteBits_demandes)

freinage = 0
window_size = 100
success_history = [] # Liste pour stocker les succès (1) et échecs (0)
total_successes = 0
total_attempts = 0
erreurTic = []


for step in range(stepMax):

    # 1. Générer des SNR aléatoires (ex: entre 0 et 20)
    snrs_reels = np.random.uniform(0, 20, size=nbUsers)
    sumSNR = np.sum(snrs_reels)

    if(nn_est_PF):
        snrs_reels_1 = np.random.uniform(0, 20, size=int(nbUsers/2))
        snrs_reels_2 = np.random.uniform(0,14,size=int(nbUsers/2))
        snrs_reels = np.hstack([snrs_reels_1,snrs_reels_2])


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
    
    """Partie Evaluation donc pas d'apprentissage"""
    # # 5. Apprentissage
    # net.backward(inputs.reshape(-1, 1), target.reshape(-1, 1))


    """Calcul du taux de succèes"""
    # 1b. On récupère l'indice choisi par le réseau et celui de l'Oracle
    choix_reseau = np.argmax(prediction)
    choix_oracle = np.argmax(target)
   
    
    total_attempts += 1
    

    taux_N_oracle[choix_oracle] += 1.0
    # 2b. On compare
    if choix_reseau == choix_oracle:
        #print(f"{prediction}")
        #print(f"{target}")
        is_success = 1
        total_successes += 1
        
        taux_N_eluSucces[choix_reseau] += 1
    else: 
        is_success = 0
        taux_N_eluErreur[choix_reseau] += 1.0

        classementInputs = net.get_Tab_erreur(inputs, nbUsers)
        erreurTic.append(classementInputs[choix_reseau])
        # #Stockage de l'erreur
        
        #print(f"{prediction}")

    success_history.append(is_success)
    
    total_attempts += 1
    

    """Fin calcul taux"""
    
    if(choix_oracle < nbUsers):
        tauxBits_transmis_Oracle = snrs_reels[choix_oracle]
    else:
        tauxBits_transmis_Oracle = 1

    if(choix_reseau < nbUsers and quantites_reelles[choix_reseau] > 0 ):
        tauxBits_transmis_NN = snrs_reels[choix_reseau]
        qteBits_Transmis += np.minimum(snrs_reels[choix_reseau], quantites_reelles[choix_reseau])
        qteBits_tab_Transmis.append(qteBits_Transmis)
    else:
        if(choix_reseau == choix_oracle):
            tauxBits_transmis_NN = 1
        else:
            tauxBits_transmis_NN = 0 
        
        qteBits_Transmis += 0
        qteBits_tab_Transmis.append(qteBits_Transmis)



    

    

    if step % 100 == 99:
        #ajout de qte (+ réaliste)
        quantites_reelles += np.random.randint(0, 400, size=nbUsers).astype(float)
        #quantites_reelles += 0
        sommeQte= 400
        
        #qteBits_tab_demandes.append(qteBits_demandes)
        qteBits_demandes += np.sum(quantites_reelles)
        regroupementTauxSucc.append(np.mean(success_history)*100)
        success_history = []
    

    qteBits_tab_demandes.append(qteBits_demandes)

    
    gravites_Erreurs.append(np.mean(erreurTic))
    erreurTic =[]
    tauxbits_transmis.append((np.sum(tauxBits_transmis_NN)/ np.sum(tauxBits_transmis_Oracle))*100)
    if(tauxbits_transmis[-1] > 100):
        erreur = {
            "step": int(step),
            "choix_reseau": int(choix_reseau),
            "choix_oracle": int(choix_oracle),
            "gravite_erreur": int(classementInputs[choix_reseau]),
            "snrs": snrs_reels.tolist(),
            "quantites": quantites_reelles.tolist(),
            "valeurs_neurones_sortie": prediction.flatten().tolist()
        }
        erreurs.append(erreur)

    tauxBits_transmis_Oracle = 0
    tauxBits_transmis_NN = 0
    
    ##Sauvegarde des erreurs
    if erreurs:
        if os.path.exists(fichier_erreurs):
            with open(fichier_erreurs, "r") as f:
                existant = json.load(f)
        else:
            existant = []
        existant.extend(erreurs)
        with open(fichier_erreurs, "w") as f:
            json.dump(existant, f, indent=4)
        erreurs = []  # reset

    # 6. Action : On baisse la quantité de l'utilisateur choisi par l'ORACLE 

    choix = np.argmax(target)
    if choix < nbUsers: # Si ce n'est pas le neurone de repos
        quantites_reelles[choix] -= snrs_reels[choix_oracle] # On consomme de la data
        if quantites_reelles[choix] < 0: quantites_reelles[choix] = 0

    
    




    
print(f"Taux de succès moyen suite à {step +1} tics : {np.mean(regroupementTauxSucc)} pour un LR de {net.lr}"
      f"\n avec comme ecart-type {np.nanstd(regroupementTauxSucc)} ")


fig, ax = plt.subplots(figsize=(12, 5))

# Aplatir tous les taux en une seule liste
x_global = [i * 100 for i in range(len(regroupementTauxSucc))]

# Afficher tous les points
ax.plot(x_global,regroupementTauxSucc, color="blue", marker="o", 
        markersize=2, label="Valeurs", alpha=0.5)

ax.set_xticks(x_global[::10])


ax.set_title(f"Taux de succès tous les 1000 tics")
ax.set_xlabel("Tics")
ax.set_ylabel("Taux de succès")
ax.legend()
ax.grid(True)



"""graphe sur les neurones """

fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)


resultat1 = taux_N_eluErreur / taux_N_oracle
print(f"NbE / Nb-devraitE : {resultat1}")

ax2.bar(range(len(resultat1)), resultat1, color="blue", alpha=0.7)
ax2.set_title("Nb fois elu par le NN, alors que erreur / nb fois de l'oracle")
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
print(f"{taux_N_eluErreur + taux_N_eluSucces}")



"""graphe sur les erreurs"""
fig3, ax4 = plt.subplots(figsize=(12, 5))

ax4.plot(gravites_Erreurs, linestyle='none',color="purple", marker="o", 
        markersize=2, label="Erreurs", alpha=0.5)

ax4.set_title(f"Evolution du degré de gravité des erreurs au cours des tics")
ax4.set_xlabel("Tics")
ax4.set_ylabel("Gravité")



"""graphe sur transmssion de bits"""
fig4, ax5 = plt.subplots(figsize=(12, 5))


ax5.plot(tauxbits_transmis, color="blue", marker="o",linestyle='none', 
        markersize=2, label="bits_transmis", alpha=0.5)

ax5.set_title(f"Evolution des bits transmis cours des tics")
ax5.set_xlabel("Tics")
ax5.set_ylabel("taux de transmissions")



"""graphe sur les quantités de bits transmises"""

print(f"{qteBits_Transmis}")

fig5, ax6 = plt.subplots(figsize=(12, 5))

ax6.plot(qteBits_tab_demandes,color='blue',label='quantités demandées')
ax6.plot(qteBits_tab_Transmis, color='green',label='quantités bits transmises')
ax6.set_title("Comparaison entre les bits transmis (vert) et les bits demandés (bleu)")
ax6.set_xlabel("Tics")
ax6.set_ylabel("quantités transmises")


plt.tight_layout()
#plt.savefig("graphe.pdf")
plt.show()