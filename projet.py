#Imports
import numpy as np
from matplotlib import pyplot as plt
import math

# =============================================================================
# Définition des paramètres
# =============================================================================
phi1=1
phi2=0.5
#Nb de particules
n=50
#Valeur minimale pour x et y
xMin=-5.12
#Valeur maximale pour x et y
xMax=5.12
#Poids initial pour newVitesse (pondération d'inertie)
wInit=0.8
#Poids final pour newVitesse (pondération d'inertie)
wFinal=0.2 
#Initalisation du poids (pondération d'inertie)
w=wInit
#Nombre d'itérations max qui sera notre condition d'arrêt
nbIterations=500

# =============================================================================
# Définition des fonctions
# =============================================================================
def f(listePoint):
    """
    On définit la fonction de Rastrigin définie de R^2 dans R
    On calcule l'image par cette fonction pour chaque particule: f((x,y)) où (x,y) est la coordonnée de la particule
    
    ----------
    listePoint : Matrice n lignes et 2 colonnes
        Chaque composante est la coordonnée (x,y) d'une particule.

    Returns
    -------
    rep : 
        Vecteur colonne de n composantes qui sont le calcul du critère de chaque particule par f

    """
    rep=[]
    for i in range(n):
        xi= listePoint[i][0]
        yi= listePoint[i][1]
        fi= 10*2+(xi*xi-10*math.cos(2*math.pi*xi))+(yi*yi-10*math.cos(2*math.pi*yi))
        rep.append(fi)
    return rep

def set():
    """
    Initialisation de notre algorithme
    Génération des points pour t=0 (à t=0, v=0)
    On attribue à chaque particule une position initiale aléatoire dans notre espace de recherche [-5.12,5.12]^2

    Returns
    -------
    list
        Première composante est une matrice n lignes et 2 colonnes où chaque composante est la coordonnée initiale d'une particule.
        Deuxième composante est la matrice nulle à n lignes et 2 colonnes correspondant à la vitesse initiale de chaque particule.

    """
    v=[[0,0]]
    y= (np.random.random() * 10.24 - 5.12)
    z=(np.random.random() * 10.24 - 5.12)
    x=[[y,z]]
    for i in range (n-1):
        v.append([0,0])
        y=(np.random.random() * 10.24 - 5.12)
        z=(np.random.random() * 10.24 - 5.12)
        x.append([y,z])
    return [x,v]

def best(listePoint):
    """
    Renvoie l'indice correspondant aux coordonnées de la particule la mieux placée
    La meilleure position de l’essaim 

    Parameters
    ----------
    listePoint : Matrice n lignes et 2 colonnes
        Chaque composante est la coordonnée (x,y) d'une particule.

    Returns
    -------
    rep : Entier compris entre 0 et n-1
        l'indice correspondant aux coordonnées de la particule la mieux placée

    """
    fx=f(listePoint)
    actBest=fx[n-1] #Le best actuel
    rep=n-1
    for i in range(n-1):
        if (fx[i]<actBest):
            rep=i
            actBest=fx[i]
    return rep

def personnalBest(x,pb):
    """
    Met à jour la liste contenant la meilleures position personnelle de chaque particule

    Parameters
    ----------
    x : Matrice n lignes et 2 colonnes
        Chaque composante est la coordonnée (x,y) d'une particule.
    pb : Matrice n lignes et 2 colonnes
        Chaque composante est la coordonnée (x,y) de la meilleure position d'une particule
        
    Returns
    -------
    pb : Matrice n lignes et 2 colonnes
        Chaque composante est la coordonnée (x,y) de la meilleure position d'une particule

    """
    for i in range(n):
        if (f(x)[i]<f(pb)[i]):
            pb[i]=x[i]
    return pb

def newVitesse(vAvant,x,k,pb):
    """
    Calcule les valeurs des nouvelles vitesses pour chaque particule via la formule établie dans le diapo

    Parameters
    ----------
    vAvant : Matrice n lignes et 2 colonnes
        Chaque composante est la vitesse actuel d'une particule.
    x : Matrice n lignes et 2 colonnes
        Chaque composante est la coordonnée (x,y) d'une particule.
    k : Entier compris entre 0 et n-1
        indice de la particule qui dicte la meilleure position de l’essaim .
    pb : Matrice n lignes et 2 colonnes
        Chaque composante est la coordonnée (x,y) de la meilleure position d'une particule

    Returns
    -------
    vApres : Matrice n lignes et 2 colonnes
        Chaque composante est la vitesse à la prochaine itération d'une particule.

    """
    vApres=vAvant
    gbX=x[k][0]
    gbY=x[k][1]
    for i in range (n):
        pbX=pb[i][0]
        pbY=pb[i][1]
        vApres[i][0]= w * (vApres[i][0] + phi1 * np.random.random() * (pbX-x[i][0])+ phi2 * np.random.random() * (gbX-x[i][0]))
        vApres[i][1]= w * (vApres[i][1] + phi1 * np.random.random() * (pbY-x[i][1])+ phi2 * np.random.random() * (gbY-x[i][1]))
        if (k==i):
            vApres[i]=[0,0]
    return vApres

def newPosition(listePoint,newListeVitesse):
    """
    Calcule les valeurs des nouvelles positions pour chaque particule via la formule établie dans le diapo

    Parameters
    ----------
    listePoint : Matrice n lignes et 2 colonnes
        Chaque composante est la coordonnée (x,y) actuelle d'une particule.
    newListeVitesse : Matrice n lignes et 2 colonnes
        Chaque composante est la vitesse à la prochaine itération d'une particule.

    Returns
    -------
    rep : Matrice n lignes et 2 colonnes
        Chaque composante est la coordonnée (x,y) à la prochaine itération d'une particule.

    """
    rep=listePoint
    for i in range(n):
        rep[i][0]=max(min(rep[i][0]+newListeVitesse[i][0],xMax),xMin)
        rep[i][1]=max(min(rep[i][1]+newListeVitesse[i][1],xMax),xMin)
    return rep
    

def edit(xAvant,xApres,vAvant,vApres):
    """
    Met à jour les valeurs des positions et des vitesses pour chaque particule pour la prochaine itération

    Parameters
    ----------
    xAvant : Matrice n lignes et 2 colonnes
        Chaque composante est la coordonnée (x,y) avant l'itération d'une particule.
    xApres : Matrice n lignes et 2 colonnes
        Chaque composante est la coordonnée (x,y) pour la prochaine itération d'une particule.
    vAvant : Matrice n lignes et 2 colonnes
        Chaque composante est la vitesse avant l'itération d'une particule.
    vApres : Matrice n lignes et 2 colonnes
        Chaque composante est la vitesse pour la prochaine itération d'une particule.

    Returns
    -------
    list
        Première composante est une matrice n lignes et 2 colonnes où chaque composante est la coordonnée pour la prochaine itération d'une particule.
        Deuxième composante est la matrice nulle à n lignes et 2 colonnes correspondant à la vitesse pour la prochaine itération de chaque particule.

    """
    newX=xAvant
    newV=vAvant
    y=f(xAvant)
    z=f(xApres)
    for i in range (n):
        if  y[i]>z[i]:
            newX[i]=xApres
            newV[i]=vApres
    return [newX,newV]

def main():
    """
    Programme principale (voir diapo pour plus d'infos sur les étapes)
    Il va aussi compléter les graphes à chaque itération

    Returns
    -------
    list
        Première composante est une matrice n lignes et 2 colonnes où chaque composante est la coordonnée d'une particule obtenue lors de la dernière itération.
        Deuxième composante est la matrice nulle à n lignes et 2 colonnes correspondant à la vitesse d'une particule obtenue lors de la dernière itération.
        Troisième composante est un entier compris entre 0 et n-1 correspondant à l'indice de la particule qui dicte la meilleure position de l’essaim obtenue lors de la dernière itération.
        Quatrième composante est une matrice n lignes et 2 colonnes dont chaque composante est la coordonnée (x,y) de la meilleure position d'une particule obtenue lors de la dernière itération.
    """
    [x,v]=set()
    k=best(x)
    pb=x
    xPlot=[]
    yPlot=[]
    for i in range(nbIterations):
        w=wInit - i/nbIterations * (wInit-wFinal)
        newV=newVitesse(v,x,k,x)
        newX=newPosition(x,newV)
        [x,v]=edit(x,newX,v,newV)
        pb=personnalBest(x,pb)
        k=best(x)
        xPlot.append(x[0][0])
        yPlot.append(x[0][1])
    plt.plot(xPlot,yPlot,linestyle = 'none', marker = 'o', c = 'lime', markersize = 10) #Affiche le minimum personnel la particule à chaque étape en vert
    plt.plot(x[k][0],x[k][1],linestyle = 'none', marker = 'o', c = 'blue', markersize = 12) #Affiche le minimum obtenu en bleu
    plt.plot(xPlot[-1],yPlot[-1],linestyle = 'none', marker = 'o', c = 'red', markersize = 10) #Affiche l'emplacement de la particule à la dernière étape
    plt.axis([xMin,xMax,xMin,xMax])

    plt.show()
    return [x,v,k,pb]

# =============================================================================
# Exécution de notre code et génération de graphe
# =============================================================================
[x,v,k,pb]=main() #Fonction main
print(x[k][0],';',x[k][1],';',f(x)[best(x)]) #Test Coordonnées du minimum X* et f(X*) 
print(pb[0]) #Test minimum pb pour un point quelconque
