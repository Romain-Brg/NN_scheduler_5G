import random
import numpy as np
import matplotlib.pyplot as plt

from network import Network
from network_v2 import Network_2
from entrainementMemeSeed import entrainementMEME_SEED


tabAffichage = entrainementMEME_SEED(nbUsersParam= 8 , nbLayers= 1, lr=0.01, qteMaxParam=2000, nbEntrainement=600, nbIteration=100)


fig, ax = plt.subplots(figsize=(12, 5))
couleurs = plt.cm.tab20(np.linspace(0, 1, len(tabAffichage)))  # 20 couleurs distinctes, cyclées sur 100

# couleurs = ["red", "blue", "green", "orange", "purple", "brown", "pink", "cyan"]

for v, iteration in enumerate(tabAffichage):
    tous_les_taux = [val for groupe in iteration for val in groupe]
    x_global = range(len(tous_les_taux))
    #couleur = couleurs[v]
    couleur = couleurs[v % len(couleurs)]

    # Points
    ax.plot(x_global, tous_les_taux, color=couleur, marker="o",
            markersize=2, alpha=0.3,
            label=f"Iter {v+1} (mean: {round(np.mean(tous_les_taux), 2)}%)")

# Tendance globale de toutes les itérations avec un pas de 100
tous_les_taux_all = [val for iteration in tabAffichage 
                         for groupe in iteration 
                         for val in groupe]

# Moyenne par pas de 100
pas = 100
moyennes = []
x_moyennes = []
for i in range(0, len(tous_les_taux_all), pas):
    fenetre = tous_les_taux_all[i:i+pas]
    moyennes.append(np.mean(fenetre))
    x_moyennes.append(i + pas // 2)  # centre du pas

# Courbe de tendance sur les moyennes   
deg = min(3, len(x_moyennes) - 1)  # degré adapté automatiquement
z = np.polyfit(x_moyennes, moyennes, deg)
p = np.poly1d(z)
x_smooth = np.linspace(x_moyennes[0], x_moyennes[-1], 300)
ax.plot(x_smooth, p(x_smooth), color="black", linestyle="-", 
        linewidth=2, label="Tendance globale (pas 100)")

# Afficher les moyennes par pas
ax.plot(x_moyennes, moyennes, color="black", marker="x", 
        markersize=5, linestyle="none", label="Moyennes par 100")

#TODO changemnt a faire pour graphe
ax.set_xlim(0, 600)
ax.set_title(f"Taux de succès par entrainement avec même SEED")
ax.set_xlabel("Entrainements")
ax.set_ylabel("Taux de succès")
ax.legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc="upper left")
ax.grid(True)

plt.tight_layout()
plt.show()