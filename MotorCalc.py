import pandas as pd
import json
import time
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#import requests
#resp = requests.get('http://ip-api.com/json/208.80.152.201')
#json.loads(resp.content)
#pd.dafra



#from json_database import JsonDatabase

#db_path = "users.db"

#with JsonDatabase("users", db_path) as db:
    # add some users to the database

    #for user in [
       # {"name": "bob", "age": 12},
       # {"name": "bobby"},
        #{"name": ["joe", "jony"]},
        #{"name": "john"},
        #{"name": "jones", "age": 35},
        #{"name": "joey", "birthday": "may 12"}]:
        #db.add_item(user)

    # pretty print database contents
   # db.print()
X = 0
K=0

Dat = []
Dat2 = []
Dateliminada = []
def MotorCalc(listXtM, listXvM, listXrM):
    Dat = []
    Dat2 = []
    Dateliminada = []
    X = 0
    # --------------Elimino datos anomalos-------------------------
    listamadre = MotorCalc2(listXvM,listXrM,listXtM)
    listXvM=  listamadre[0]
    listXrM = listamadre[1]
    listXtM = listamadre[2]
    # ---------------Ordeno los datos de menor a mayor-------------
    listXv = list(set(sorted(listXvM)))
    listXvr = listXv
    listXr = list(set(sorted(listXrM)))
    #--------------Elimino datos anomalos-------------------------
    coun = 0
    #while coun < len(listXv):
     #   if listXv[coun] > 25:
      #      listXv.pop(coun)
       # coun = coun + 1
    #print(listXv)
    #coun = 0
    #while coun < len(listXr):
     #   if listXr[coun] > 250:
      #      listXr.pop(coun)
      #  coun = coun + 1
    #print(listXr)
    #listXv.pop(len(listXv)-1)
    #print(listXv)
    #print(len(listXv))
    #print(len(listXr))
    #listXr = list(sorted(listXr))
    #print(listXr)
###---------Calculos las deribadas Numericas-------------
    X = 0
    for V in listXv:

        #time.sleep(0.01)

        if V >= 0 and V <= 1100:
            X = X + 1
            if X != 0 and X < len (listXv) -1:
                dat = float((listXr[X]-float(listXr[X-2])))/(2)
                Dat.append(dat)

            elif X == 1:
                dat = float((listXr[X]) - float(listXr[X-1]))

                Dat.append(dat)
            else:
                dat = float((listXr[X-1])) - float(listXr[X-2])
                Dat.append(dat)

        else:
            print("Dato Anomalo:" + str(X))

            #K = K + 1
            Dateliminada.append(V)


    X2 = 0
    for V in Dat:
         #------------>Son minusculas
        #X2 = x2 - 1 #-------------> Son mayusculas
        #time.sleep(0.01)
        if X2 != 0 and X2 < len (Dat)-1:
            dat2 = (float((Dat[X2+1]))-float(Dat[X2-1]))/2
            Dat2.append(dat2)
        elif X2 == 1:
            dat2 = float((Dat[X2 + 1]) - float(Dat[X2]))
            Dat2.append(dat2)
        else:
            dat2 = float((Dat[X2]) - float(Dat[X2-1]))
            Dat2.append(dat2)

        X2 = X2 + 1
    print("Cantidad de datos anomalos:" + str(K))
    print(Dat)
    print("Cantidad de datos anomalos:")
    print(Dat2)

    #DN------> Deribada numerica es regresionada---- Se pasa a ARRAY
    G = len(listXv) * [1]
    print("Largo de la matriz:" + str(len(listXv)))
    for g in G:
        listXv.append(g)
    listXvA = np.transpose(np.array(listXv).reshape(2, int(float(len(listXv))*0.5)))
    print((listXv))

    #print(np.array(listXv,G))
    DatA = np.array(Dat)


    #-------------Regrecion Lineal derivada primera---------------------ATA
    ATA = np.dot(np.transpose(listXvA),listXvA)
    ATY = np.dot(np.transpose(listXvA), DatA)
    print("ATA")
    print(ATA)
    print(np.linalg.inv(ATA))
    print("ATY")
    print(ATY)

    Parametros = np.dot(np.linalg.inv(ATA),(ATY))
    print("Parametros")
    print(Parametros)

    parlis = Parametros.tolist()
    parlism = parlis[0]
    parlisI = parlis[1]

    Residuo = []
    Valteo = []
    Xw = 0
    while Xw != len(listXv) or Xw > len(listXv):
        if Xw < len(listXr):
            R = float(Dat[Xw-1])-(float(listXv[Xw-1])*float(parlism) + float(parlisI))

            teo = (float(listXv[Xw]) * float(parlism)) + float(parlisI)
            #print(listXr[Xw])
            Residuo.append(R)
            Valteo.append(teo)
        Xw = Xw + 1
    print("Valores Empiticos:")
    print(Dat)
    print("valores TEoricos:")
    print(Valteo)
    print("Residuo:")
    print(Residuo)

    # -------------Regrecion Lineal derivada segunda---------------------ATA

    Dat2A = np.array(Dat2)
    # print(np.array(listXv,G))
    ATY2 = np.dot(np.transpose(listXvA), Dat2A)
    print("ATA2")
    print(ATA)
    print("ATY2")
    print(ATY2)

    Parametros2 = np.dot(np.linalg.inv(ATA), (ATY2))
    print("Parametros2")
    print(Parametros2)

    parlis2 = Parametros2.tolist()
    parlism2 = parlis2[0]
    parlisI2 = parlis2[1]

    Residuo2 = []
    Valteo2 = []
    Xw2 = 0
    while Xw2 != len(listXv) or Xw2 > len(listXv):
        if Xw2 < len(listXr)-1:
            R2= float(Dat[Xw2]) - (float(listXv[Xw2]) * float(parlism2) + float(parlisI2))

            teo2 = (float(listXv[Xw-1]) * float(parlism2)) + float(parlisI2)
            # print(listXr[Xw2])
            Residuo2.append(R2)
            Valteo2.append(teo2)
        Xw2 = Xw2 + 1
    print("Valores Empiticos2:")
    print(Dat2)
    print("valores TEoricos2:")
    print(Valteo2)
    print("Residuo2:")
    print(Residuo2)

    # ---------------Optimizacion del Proceso----------------------

    # --------------- Metodo newton rapson----------------------------------------
    var = 10
    var2 = 0
    deri = 30
    deri2 = ("No hay valor")
    deri3 = ("No hay valor")
    vaf1 = ("No hay valor")
    vaf2 = ("No hay valor")

    r = 0
    while int(deri) >= 4 or int(deri) <= -4:

       # print("DER:"+str(deri))
        r=r+1
       # print ("R ="+ str(r))
        if var < 0:
            var2 = float(var) - (var*float(parlism) + float(parlisI))/\
                   (float(var) * float(parlism2) + float(parlisI2))
            deri = float(var2) * float(parlism) + float(parlisI)
            deri2 = deri
            var = var2
            print(deri)
            print("VAR1:" + str(var))
            print("EJECUTO1")
        if var > 0:


            var2 = var - (var * float(parlism) + float(parlisI))/(float(var) * float(parlism2) + float(parlisI2))
            deri = var2 * float(parlism) + float(parlisI)
            var = var2
            deri3 = deri
            print("VAR2:" + str(var))
            print("EJECUTO2")
        if  int(var) == 1000 or r == 5000:
            #print("No se encontro valor")
            deri2 = deri
            var = -10
            vaf1 = var2
            print("EJECUTO3")
        if  int(var) == -1000 or r == 5000:
            print("No se encontro valor")
            var = 100
            deri3 = deri
            print("EJECUTO4")
            vaf2 = var2
        if r == 150000:
            #print("BREAK")
            #print("EJECUTO5")
            break



    print(deri)
    print(deri2)
    print(deri3)
    print(vaf1)
    print(vaf2)
    return ("Valor optimo:"+ str(var2),
           "Valor optimo:"+ str(vaf1),
            "Valor optimo:" + str(vaf2),
            "1er deribada:"+"X*"+str(parlism) + "+" + str(parlisI),
            "2da deribada:"+"X*"+str(parlism2) + "+" + str(parlisI2),
            "Datos trabajados:" + str(listXv) + ";"+str(listXr))





        # listXv_train, listXv_test ,Dat_train, Dat_test = train_test_split(listXvA, DatA, test_size = 0.2)

    #lr = linear_model.LinearRegression()

    #lr.fit(listXv_train, Dat_train)

    #Y_pred = lr.predict(listXv_test)

    #print(Y_pred)



#MotorCalc([1,2,11,12,13,14,19,20,28,29],
          #[4,7,11,7,10,6,9,5,100,3],
          #[11,41,109,41,89,29,71,19,9899,5])




#Filtros de anomalis minimos cuadrados

def MotorCalc2(numeros1, numeros2, Tiempo):
    largox = len(numeros1)
    largoy = len(numeros2)
    anterior = 0
    anterior2 = 0
    Filtrados = []
    Desechados = []
    TiempoN = []
    Filtrados2 = []
    Desechados2 = []
    kount = 0
    for X in numeros1:
        sumaX = int(X)+int(anterior)
        #print(sumaX)
        anterior = sumaX

    Promedio = sumaX / int(len(numeros1))
# Calculo de desvio estandar
    for X2 in numeros1:
        min2 = (float(X2)-float(Promedio))*(float(X2)-float(Promedio))
        #print(min2)
        sumaX2 = (float(min2) + float(anterior2))
        anterior2 = sumaX2
    min2 = ((sumaX2 / float(len(numeros1))))**.5
    print (min2)
    print (Promedio)
   # print (numeros1)
    prommin2sum = Promedio + 3 * min2
    prommin2res = Promedio - 3 * min2
    for X3 in numeros1:
        if X3 <= (prommin2sum) and X3 >= (prommin2res):
            Filtrados.append(X3)
            Filtrados2.append(numeros2[kount])
            TiempoN.append(Tiempo[kount])
        else :
            Desechados.append(X3)
            Desechados2.append(numeros2[kount])
        kount = kount + 1
    print("Volores de variavle filtrada:")
    #print(Filtrados)
    print("Volores de respuesta filtrada:")
    #print(Filtrados2)
    print("Volores de desechos respuesta2:")
    print(Desechados)
    print("Volores de desechosrespuestac2:")
    print(Desechados2)
    print("Volores de tiempo:")
    #print(TiempoN)
    print("valore de decision")
    print(prommin2sum)
    print(prommin2res)
    print("valore de promedio")
    print(Promedio)
    print("min2")
    print(min2)
    print(9**.5)
    Filtradoslist = [Filtrados, Filtrados2, TiempoN]
    return (Filtradoslist)

#MotorCalc2([1,2,101,12000,13,14,19,200,28,290],
         #[4,7,11,7,10,6,9,5,100,3],
          # [4, 7, 11, 7, 10, 6, 9, 5, 100, 3]
           #)