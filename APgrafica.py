

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow
import MotorCalc as Motor
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QProgressBar
import json
import time
#import PyQt5.QtWidgets
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import sys
import random
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
import matplotlib.backends
matplotlib.use('Qt5Agg')
import pandas as pd
import json

import numpy as np

from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5 import QtCore, QtWidgets
import MotorCalc


listX  = []
listXt = []
listXv = []
listXr = []

fileName = 0


class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        title = "Base de datos"
        self.setWindowTitle(title)
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1400, 700)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        ####boton 1
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(550, 640, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.Cargar)
        ####Coneccion de boton 2
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(650, 640, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.Detener)
        ####Coneccion de boton 3
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(750, 640, 75, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.Procesar)
        self.pushButton_3.clicked.connect(self.Guardar)
        self.pushButton_3.clicked.connect(self.motorcalc)

        # self.PlotM(plt, listXr2)
        #self.pushButton_3.clicked.connect(self.PlotM)
######-------------------Datos de optimizacion----------------
        self.listWidget = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(300, 500, 1000, 70))
        self.listWidget.setObjectName("listWidget")

        #####---------Texto de Widget------------------
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_2 = QtWidgets.QComboBox(self.centralwidget)
        #self.comboBox_2.setGeometry(QtCore.QRect(500, 370, 200, 22))
        self.comboBox_2.setGeometry(QtCore.QRect(500, 37, 200, 22))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("Lista de base de datos prosesada")
        self.lineEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        #self.lineEdit.setGeometry(QtCore.QRect(300, 395, 1000, 200))
        self.lineEdit.setGeometry(QtCore.QRect(300, 59, 1000, 400))
        self.lineEdit.setObjectName("lineEdit")

     ####-----------Fin de lista texto-----------
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        #self.graphicsView.setGeometry(QtCore.QRect(10, 30, 1300, 335))
        self.graphicsView.setGeometry(QtCore.QRect(1, 3, 1, 3))
        self.graphicsView.setObjectName("graphicsView")

        ####Calacteristica a la barra#####
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(350, 600, 700, 23))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        ####Calacteristica a la barra por mi#####
        self.step = 0
        self.progressBar.setValue(0)


        ####Calacteristica a la barra fin#####

        self.lcdNumber_1 = QtWidgets.QLCDNumber(self.centralwidget)
        #self.lcdNumber_1.setGeometry(QtCore.QRect(10, 395, 245, 75))
        self.lcdNumber_1.setGeometry(QtCore.QRect(10, 145, 245, 75))
        self.lcdNumber_1.setObjectName("lcdNumber_1")

        self.lcdNumber_2 = QtWidgets.QLCDNumber(self.centralwidget)
        #self.lcdNumber_2.setGeometry(QtCore.QRect(10, 495, 245, 75))
        self.lcdNumber_2.setGeometry(QtCore.QRect(10, 245, 245, 75))
        self.lcdNumber_2.setObjectName("lcdNumber_2")

        self.lcdNumber_3 = QtWidgets.QLCDNumber(self.centralwidget)
        #self.lcdNumber_3.setGeometry(QtCore.QRect(10, 600, 245, 75))
        self.lcdNumber_3.setGeometry(QtCore.QRect(10, 345, 245, 75))
        self.lcdNumber_3.setObjectName("lcdNumber_3")

        self.toolButton = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton.setGeometry(QtCore.QRect(0, 5, 200, 22))
        self.toolButton.setObjectName("toolButton")

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.pushButton_2, self.pushButton_3)
        MainWindow.setTabOrder(self.pushButton_3, self.graphicsView)
        MainWindow.setTabOrder(self.graphicsView, self.listWidget)
        MainWindow.setTabOrder(self.listWidget, self.pushButton)

    #####----------setpoint-------------------




    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Cargar"))
        self.pushButton_2.setText(_translate("MainWindow", "Borrar"))
        self.pushButton_3.setText(_translate("MainWindow", "Procesar"))
        self.toolButton.setText(_translate("MainWindow", "Opciones"))



    def Cargar(self):

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*)Json Files (*.json);;All Files (*)", options=options)
        if fileName != "":
            print(fileName)
            print(type(fileName))

            ODataFrame = fileName
        #pd.dat_json('dat.json', lines=True)
        #file = open('dat.json', 'w')

        #file=list(range(1,300))
        #ODataFrame, _filter = QtWidgets.QFileDialog.getOpenFileName(None, "Open " +" Data File", '.', "(*.json)")
            with open(ODataFrame) as file:

                data = json.load(file)

                for datos in data:
                    T = (datos['Tiempo'])
                    V = (datos['Variable'])
                    R = (datos["Respuesta"])


                    X = '[' + "Tiempo:" + str(T) + " " + "Variable:" + str(V) + " " + "Respuesta:"+ str(R) + ']'
                    listX.append(X)
                    listXt.append(T)
                    listXv.append(V)
                    listXr.append(R)

            print(listXv)

            return (T , V , R, listX , listXt, listXv, listXr)

    def Detener(self):
        self.lcdNumber_1.display(str(0))
        self.lcdNumber_2.display(str(0))
        self.lcdNumber_3.display(str(0))
        fileName = ""
        options = ""
        file = ""
        T = 0
        V = 0
        R = 0
        listX = []
        listXt = []
        listXv = []
        listXr = []
        #var2 = MotorCalc.MotorCalc(1, 1, 1)
        #var20 = MotorCalc.MotorCalc2(1, 1, 1)
        filtrados = [1,1,1]
        self.PlotM(plt, plt2, filtrados)
        #options = QFileDialog.Options()
        #options |= QFileDialog.DontUseNativeDialog



        return (T, V, R, listX, listXt, listXv, listXr,
                self.lcdNumber_1.display(str(0)),
                self.lcdNumber_2.display(str(0)),
                self.lcdNumber_3.display(str(0)),
                self.lineEdit.clear(), fileName, options,
                file)


    def Procesar(self):
        count = 0
        print("Ejecuto")

        G = 0
        Glen = len(listXt)
        for T in listXt:
            Event = True

            G = G + 1
            Gpo = (G/Glen)*100

            #time.sleep(0.05)
            V = listXv[count]
            print(V)


            print("Ejecuto")
            R = listXr[count]
            count = count + 1
            if Event == True:
                self.lcdNumber_1.display(str(T))
                self.lcdNumber_2.display(str(V))
                self.progressBar.setValue(Gpo)
                self.lcdNumber_3.display(str(R))


                #self.progressBar.event(Event)
                self.Procesar2

        return  (self.lcdNumber_1.display(str(T)),
                self.lcdNumber_2.display(str(V)),
                self.lcdNumber_3.display(str(R)),
                self.lineEdit.insertPlainText(str(listX)))


    def Procesar2(self, T, V, R, Event=True):

        self.lcdNumber_1.display(str(T))
        self.lcdNumber_1.event(Event)
        self.lcdNumber_2.display(str(V))
        self.lcdNumber_2.event(Event)
        self.lcdNumber_3.display(str(R))
        self.lcdNumber_3.event(Event)

    def Guardar(self):
        df = pd.DataFrame({'"Tiempo"': listXt,
                           'Variable': listXv,
                           'Respuesta': listXr})
        df = df[['"Tiempo"', 'Variable', 'Respuesta']]
        df.to_excel('Datosordenados.xlsx', sheet_name='Datosordenados2')
        print(df)


        return (self.lineEdit.insertPlainText(str("\n")),
                self.lineEdit.insertPlainText(str(df)),
                df)


    def motorcalc(self):

        var2 = Motor.MotorCalc(listXt, listXv, listXr)
        print(var2)

        filtrados= Motor.MotorCalc2(listXv, listXr, listXt)


        self.PlotM(plt,plt2, filtrados)

        return (var2, self.listWidget.insertPlainText(str("\n")),
                self.listWidget.insertPlainText("Valor optimo de proceso:" + str(var2)))

    def PlotM(self, plt, plt2, filtrados):

        #Se filtran valores anomalos#
        listXtf2 = filtrados[2]
        listXvf2 = filtrados[0]
        listXrf2 = filtrados[1]
        #print (len(listXtf2)," " ,len(listXvf2)," ", len(listXrf2) )
        #print((listXtf2), " \n ", (listXvf2), "\n ", (listXrf2))

        Xp = listXtf2
        Yp = listXvf2
        Zp = listXrf2
        print("valor")
        #print(filtrados[0])
        print("valor2----------------------------------------------------------------------"
              "--------------------------------------------------------------------------")
        #print(filtrados[1])
        plt.plot(Xp, Yp,"-o")
        plt2.plot(Xp, Zp)
        plt.show()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())
