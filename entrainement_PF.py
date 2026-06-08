import random
import numpy as np
import matplotlib.pyplot as plt


from network_PF import Network_PF




# Initialisation

vitesseInit = 0.01
stepMax = 2000000
regroupementTauxSucc = []
meanTauxSucc = []
stockageVitesses = []
qteMAx = 2000
net = Network_PF(nbUsers=4,learning_rate=0.000125)
net.load_NN("NN_PF_1Layer_4_4U_real_V3.npz")
nbUsers = net.nbUsers

#Boucle pour tester différentes valeurs de mon LR
#for vitesse in np.linspace(vitesseInit, 0.06, 6):
    #net.updateLR(vitesse)
#vitesse = round(vitesse, 2)

#stockageVitesses.append(vitesse)



#Calcul des taux pour les neurones
taux_N_eluSucces = np.zeros(nbUsers + 1)
taux_N_eluErreur = np.zeros(nbUsers + 1)
taux_N_oracle = np.zeros(nbUsers + 1)

#gravité erreur

gravites_Erreurs = []





# État initial des quantités (ex: chaque user a entre 0 et 700 unités)
quantites_reelles = np.random.randint(0, 400, size=nbUsers).astype(float)
sommeQte= 400

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

nbErreur = 0
for step in range(stepMax):

    # 1. Générer des SNR aléatoires (ex: entre 0 et 20)
    
    snrs_reels_1 = np.random.uniform(0, 20, size=int(nbUsers/2))
    sumSNR_1 = np.sum(snrs_reels_1)
    #print(f"{snrs_reels_1}")
    snrs_reels_2 = np.random.uniform(0,14,size=int(nbUsers/2))
    #print(f"{snrs_reels_2}")


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
        
        taux_N_eluSucces[choix_reseau] += 1
    else: 
        is_success = 0
        nbErreur += 1
        taux_N_eluErreur[choix_reseau] += 1.0
        classementInputs = net.get_Tab_erreur(inputs, nbUsers)
        # print(f"\nSNRS : {inputs[0::2]} ; QTES {inputs[1::2]}")
        # print(f"Oracle :{target} ; Prediction NN : {prediction}")
        # print(f"classement :{classementInputs}")
        erreurTic.append(classementInputs[choix_reseau])
        #print(f"{prediction}")

    success_history.append(is_success)
    
    total_attempts += 1
    
    

    """Fin calcul taux"""


    # 6. Action : On baisse la quantité de l'utilisateur choisi par l'ORACLE 
    choix = np.argmax(target)
    if choix < nbUsers: # Si ce n'est pas le neurone de repos
        quantites_reelles[choix] -= snrs_reels[choix_oracle] # On consomme de la data
        if quantites_reelles[choix] < 0: quantites_reelles[choix] = 0
    #print(f"Step:{step} - Choix Oracle: {choix} - Choix NN :  {choix_reseau} - \n Qtes initiales normalisées : {qtes_norm} - SNR inital: {np.trunc(snrs_reels)} Qtes restantes: {quantites_reelles}")
    

    

    if step % 100 == 99:
        #ajout de qte (+ réaliste)
        quantites_reelles += np.random.randint(0, 400, size=nbUsers).astype(float)
        sommeQte= 400
        gravites_Erreurs.append(np.mean(erreurTic))
        erreurTic =[]

    if step % 1000 == 999:
        regroupementTauxSucc.append(np.mean(success_history)*100)
        success_history = []
        # Sauvegarde des erreurs

    
    





    
print(f"Taux de succès moyen suite à {step +1} tics : {np.mean(regroupementTauxSucc)} pour un LR de {net.lr}"
      f"\n avec comme ecart-type {np.nanstd(regroupementTauxSucc)} et avec {nbErreur} erreurs faites ")


net.save_NN("NN_PF_1Layer_4_4U_real_V3.npz")




"""calcul des graphes"""

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