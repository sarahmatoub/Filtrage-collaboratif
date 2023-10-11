#%%
import numpy as np
from sklearn.decomposition import NMF


# Chaque ligne représente les notes données par un individu à chaque film
import pandas as pd

data = {'Individus': [1, 2, 3, 4, 5],
        'Good Doctor': ["na", 4, 4, 4, "na"],
        'Murder': [4, "na", "na", 2, 1],
        'The Recruit': ["na", 1, 5, "na", "na"],
        'Manifest': [3, 5, "na", 4, 2]
}

df = pd.DataFrame(data)

# Application de la méthode de factorisation par matrices non-négatives (NMF)

# Remplacer les donnees manquantes par des 0
df = df.replace("na", 0)

# On définit la matrice X
X = df.values[:, 1:]

# On crée un objet NMF avec 2 composantes, une initialisation aléatoire et une graine aléatoire fixée à 0
model = NMF(n_components=2, init="random", random_state=0)

# On applique la factorisation NMF à la matrice X
W = model.fit_transform(X) # Matrice des poids (films classés)
H = model.components_ # Matrice des coefficients (goûts des individus)

# On reconstruit la matrice X à partir des matrices des poids et des coefficients obtenues par NMF
X_nmf = np.dot(W, H)

print("Matrice des poids (films classés) : \n", W)
print("Matrice des coefficients (goûts des individus) : \n", H)
print("Matrice X reconstruite : \n", X_nmf)


# On calcule l'erreur de prediction pour nmf
error_nmf = np.sqrt(np.mean((X - X_nmf)**2))
error_nmf

#%% 
# Méthode SVD (https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)
import numpy as np

U, D, V = np.linalg.svd(X, full_matrices=False)

# Check that left singular matrix is orthonornal (unitary)
np.set_printoptions(precision=3, suppress=True)
print(U @ U.transpose())

# Check that left singular matrix is orthonornal (unitary)
np.set_printoptions(precision=3, suppress=True)
print(V @ V.transpose())


# Reconstruct the original matrix using the SVD components
X_svd = U @ np.diag(D) @ V


print("Reconstructed matrix using SVD:\n", X_svd)
error_svd = np.sqrt(np.mean((X - X_svd)**2))
error_svd

#%%
# Méthode de complétion de matrice (https://pypi.org/project/fancyimpute/)

from fancyimpute import NuclearNormMinimization

df2 = pd.DataFrame(data)
X2 = df2.values[:, 1:]
# Replace missing values with NaN
X2[X2 == "na"] = np.nan

#%%
# matrix completion using convex optimization to find low-rank solution
# that still matches observed values. Slow!
X_filled_nnm = NuclearNormMinimization().fit_transform(X2)

nnm_mse = ((X_filled_nnm - X) ** 2).mean()
print("Nuclear norm minimization MSE: %f" % nnm_mse)













# %%
