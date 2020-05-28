##Initialisation
#Imports
import numpy as np
from matplotlib import pyplot as plt
import math

#Constantes
phi1=1
phi2=0.5
n=50 #Nb de points
xMin=-5.12 #Valeur minimale pour x1 et x2
xMax=5.12 #Valeur maximale pour x1 et x2
wInit=0.8 #Poids initial pour newVitesse
wFinal=0.2 #Poids final pour newVitesse

#Initalisation du poids
w=wInit

#Conditions d'arrêt
nbIterations=500 #Nombre d'itérations max

#Applique la fonction de Rastrigin
def f(listePoint): #Moi
    rep=[]
    for i in range(n):
        xi= listePoint[i][0]
        yi= listePoint[i][1]
        fi=10*2+(xi*xi-10*math.cos(2*math.pi*xi))+(yi*yi-10*math.cos(2*math.pi*yi))
        rep.append(fi)
    return rep

#Génération des points pour t=0 (à t=0, v=0)
def set():
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


##Passage à l'étape suivante


#Renvoie les coordonnées de la particule la mieux placée
def best(listePoint):
    fx=f(listePoint)
    actBest=fx[n-1] #Le best actuel
    rep=n-1
    for i in range(n-1):
        if (fx[i]<actBest):
            rep=i
            actBest=fx[i]
    return rep

#Met à jour la liste contenant la meilleures position personnelle de chaque particule
def personnalBest(x,pb):
    for i in range(n):
        if (f(x)[i]<f(pb)[i]):
            pb[i]=x[i]
    return pb

#Définit la nouvelle vitesse
def newVitesse(vAvant,x,k,pb):
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

#Définit la nouvelle position
def newPosition(listePoint,newListeVitesse):
    rep=listePoint
    for i in range(n):
        rep[i][0]=max(min(rep[i][0]+newListeVitesse[i][0],xMax),xMin)
        rep[i][1]=max(min(rep[i][1]+newListeVitesse[i][1],xMax),xMin)
    return rep
    
#Ajoute la nouvelle position et la nouvelle vitesse
def edit(xAvant,xApres,vAvant,vApres):
    newX=xAvant
    newV=vAvant
    y=f(xAvant)
    z=f(xApres)
    for i in range (n):
        if  y[i]>z[i]:
            newX[i]=xApres
            newV[i]=vApres
    return [newX,newV]

##Fonction principale
def main():
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
    # plt.axes()
    # plt.title("Position d'une particule quelconque donnée pour chaque itération")
    plt.show()
    return [x,v,k,pb]
  
##Tests
#[x,v]=set()
#print(f(x)) #Test f
#print(newVitesse(v,x,n-1,x)) #Test newVitesse
#print(best(x)) #Test best
#print(newPosition(x,newVitesse(v,x,n-1,x))) #Test newPosition
#[x,v,k,pb]=main() #Fonction main
[x,v,k,pb]=main() #Fonction main
print(x[k][0],';',x[k][1],';',f(x)[best(x)]) #Test Coordonnées du minimum X* et f(X*) 
print(pb[0]) #Test minimum pb pour un point quelconque