# Filtrage-collaboratif

Ce dépôt git contient les fichiers suivants :

- Un fichier `.gitignore` afin de garder ce dépôt propre.

- Un fichier `requirements.txt` contenant les librairies nécessaires pour tester ce projet.

- Un fichier `source.py` contenant un exemple simple d'application de 3 méthodes (NMF, SVD et complétion de matrices) sur une matrice $X$ représentant les scores, compris entre 1 et 5, donnés à des séries par 5 individus.

### Première étape :

Ce Projet a été effectué sur **VsCode** , avant de pouvoir
expérimenter notre code, vous devez d'abord vous assurer d'avoir un
environnement de travail fonctionnel (si cela est déja fait vous pouvez passer
à la deuxième étape).

- Installer `VsCode` en suivant les instructions sur [ce lien](https://code.visualstudio.com/download).

- Installer l'extension `Python` en suivant les instructions sur [ce lien](https://www.pythontutorial.net/getting-started/setup-visual-studio-code-for-python/).

### Deuxième étape :

- Ouvrez un terminal et clônez le référentiel via la commande suivante :

```bash
$ git clone https://github.com/sarahmatoub/Filtrage-collaboratif.git
```

- Téléchargez les modules présents dans le fichier requirements.txt via la commande `pip` suivante :

```bash
$ pip install -r requirements.txt 
``` 
## Références :

- Méthode NMF : https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html

- Méthode SVD : https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html

- Méthode de complétion de matrices : https://pypi.org/project/fancyimpute/

## Autrices :

- Pauline Dusfour--Castan : pauline.dusfour-castan@etu.umontpellier.fr

- Sarah Matoub : sarah.matoub@etu.umontpellier.fr

- Thamara Renoir : thamara.renoir@etu.umontpellier.fr

