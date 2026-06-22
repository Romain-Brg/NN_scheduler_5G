# Projet Stage L3 Info - Réseau de Neurones et scheduling

Un réseau de neurones codé exclusivement en Python à l'aide la bibliothèque NumPy. <br>
<br>
<br>

## Faire fonctionner le programme

Il faut d'abord avoir python installé, version 3.14 minimum

Une bibliothèque python est nécessaire, il s'git de NumPy. Si elle n'est pas installée, alors il faut utiliser la commande :
```bash
pip install numpy
``` 

Une auter bibliothèque est très fortement reccomandée pour l'affichage des graphes,  :

```bash
pip install matplotlib
```

Ou plus simplement :
```bash
pip install -r requirements.txt
```

<br><br><br>
## Les différentes versions

- V2 =  ajout de qtes pendant l'entrainement

- V3 = correspond à la V2 mais avec une meilleure normalisation qui permet d'améliorer la convergence. 
    Problème : Pendant les 99 premiers tics, le réseau reçoit des quantités normalisées de manière absolue (divisées par 700 la qte max attribuable à un user). Mais à partir du tic 100, la variable sommeQte devient la somme réelle des files d'attente (qui peut valoir 1500 ou 2000). Le réseau reçoit des données normalisées de manière relative.
    Le réseau doit gérer deux types de signaux totalement différents pour la même entrée. Cela nuit gravement à sa convergence. 
    Solution : On change en améliorant la normalisation.

- V4 et V5 sont des versions similaires à la V3 mais à des fins de présentations
<br>
<br>
## Comment entraîner un réseau de neurones ?

### 1ère étape :
Choisir un entrainement, nous recommendons de prendre entrainementRandomWajout.py

    python entrainementRandomWajout.py

Cet entrainement le plus complet entraîne de manière aléatoire et ajoute tout les 100 tics des datas dans les buffers
<br><br>

### 2ème étape :

Choisir le réseau souhaité entre :

* NN avec 1 étage au milieu : ``` network.py```

* NN avec 2 étages au milieu : ``` network_v2.py```   

* NN adapté pour le PF (plus compliqué à paramétrer - non-recommandé) : ``` network_PF.py```


Puis dans le fichier d'entrainement choisi, modifier si besoin la création du NN

```python
#dans le fichier entrainement
net = Network(nbUsers=4,learning_rate=0.01) 

""" ou alors : """
net = Network_2(nbUsers=4,learning_rate=0.01) 

```


#### Important : 
S'il est nécessaire de charger un réseau existant (`exemple_reseau_existant.npz`), décommentez (enlever le #) la ligne :

```python
net.load_NN("exemple_reseau_existant.npz")
```

Cela peut être utile lors de l'entrainement et donc le besoin de réutiliser une réseau déjà entrainé.
<br><br>

### 3ème étape :

Il est possible de sauvegarder le NN a tout moment dans le code en utilisant la ligne : 

```python
net.save_NN("exemple_reseau_entraine.npz")
```

Il faut bien penser à ajouter le .npz à la fin

### Poursuite de l'entrainement :
Si vous souhaitez poursuivre l'entrainement mais en changeant le learning rate, il suffit de faire les mêmes manipulations que celles précédentes en changeant dans la ligne 
```python 
net = Network(nbUsers=4,learning_rate=0.01) 
```
le paramètre `learning_rate`.

Et il faudra recharger un réseau de neurones déjà existant et entrainé (cf : 2ème étape)

<br><br>
## Evaluation d'un NN

A l'aide du fichier `evaluation.py `, on réalise les mêmes choses que pour un entrainement au niveau de l'implémentation du réseau, pour le reste il n'y a rien à faire les différents graphes s'afficheront. <br><br><br>


## Visualisation d'un NN en HTML

Il faut donc utiliser la commande :
```bash
python ~/cheminVersFichier/visualiseur_NN.py mon_reseau_existant.npz -o schema_mon_reseau.html
```
Cela produira un fichier HTML représentant le NN.