La regression logistique est un modele lineaire qui permet de resoudre es problemes de **classification**
On ne cherche plus a predire une valeur continue mais une variable discrete (on essaye de trouver la classe d'un
individu)

Le resultat est l'expression d'une droite avec autant de variables que l'on a de features
Cette droite sert a separer nos classes

Ici, on veut un nombre entre 0 et 1. On va donc utiliser la **fonction sigmoide** :

- plus y est proche de 1, plus on est sur que l'individu appartient a la classe 1

Les etapes :

- lecture du dataset
- separation des donnees (X et y)
- standardisation des donnees (soustraire la moyenne et diviser par l'ecart type)

- On initialise le vecteur de poids (selon la dimension de X)
- on predit et on retourne les poids grace a la fonction sigmoide
- on calcule la fonction cout (cross entropy loss)
- fit : on va faire une descente de gradient pour optimiser les poids afin de minimiser la fonction cout
