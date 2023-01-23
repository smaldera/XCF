import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, QTabWidget,QVBoxLayout,  QSpinBox, QFileDialog,  QLineEdit, QLabel,QGridLayout, QGroupBox, QFormLayout
from PyQt5.QtGui import QIcon, QIntValidator,QDoubleValidator,QFont
from PyQt5.QtCore import pyqtSlot, Qt

import acq_loop

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'PROVA SIMO!!!!!!'
        self.left = 0
        self.top = 0
        self.width = 600
        self.height = 200
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.myDAQwindow =DAQwindow(self)
        self.setCentralWidget(self.myDAQwindow)
        
        self.show()

        
class DAQwindow(QWidget):
    
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        #self.layout = QVBoxLayout(self)
        mainLayout = QGridLayout()

        self.pedFilePath=""
        self.outDataPath=""

        self.gain=''
        self.l7=QLabel(self)
        
        # input ped file 
        self.l1=QLabel(self)
        self.l1.setText("input pedestal file:")
        #self.layout.addWidget(self.l1)
        self.e1 = QLineEdit(self)
        #e1.setValidator(QIntValidator())
        #e1.setMaxLength(4)
       # self.e1.setAlignment(Qt.AlignRight)
       # self.e1.setFont(QFont("Arial",20))
        self.e1.editingFinished.connect(self.e1_finishEdit)
        #self.layout.addWidget(self.e1)

        self.btn1 = QPushButton('choose file')
        self.btn1.clicked.connect(self.on_click)
        #self.layout.addWidget(self.btn1)
       

        # input data path:
        self.l2=QLabel(self)
        self.l2.setText("data folder:")
        #self.layout.addWidget(self.l2)
        self.e2 = QLineEdit(self)
        #e1.setValidator(QIntValidator())
        #e1.setMaxLength(4)
       # self.e1.setAlignment(Qt.AlignRight)
       # self.e1.setFont(QFont("Arial",20))
        self.e2.editingFinished.connect(self.e2_finishEdit)
        #self.layout.addWidget(self.e2)

        self.btn2 = QPushButton('choose folder')
        self.btn2.clicked.connect(self.on_click2)
        #self.layout.addWidget(self.btn2)



        
        #inputFieldsLayout=QGridLayout()
        inputFieldsLayout= QFormLayout()
        
        self.inputFieldsBox = QGroupBox("parameters")
        
        # input  Gain:
        #self.l5=QLabel(self)
        #self.l5.setText("Gain:")
        #self.layout.addWidget(self.l5)
        self.e3 = QLineEdit(self)
        self.e3.setValidator(QIntValidator())
        self.e3.setMaxLength(3)
       # self.e1.setAlignment(Qt.AlignRight)
       # self.e1.setFont(QFont("Arial",20))
        pippo=5
       # self.e3.editingFinished.connect(lambda: self.e3_finishEdit(pippo))

        self.e3.editingFinished.connect(lambda: self.lineEdit_finshGenertic(self.e3,self.gain,self.l7))
        #self.layout.addWidget(self.e3) 

        
        inputFieldsLayout.addRow("Gain", self.e3)

        
        # input  time:
       # self.l6=QLabel(self)
       # self.l6.setText("Exp. time:")
        #self.layout.addWidget(self.l6)
        self.e4 = QLineEdit(self)
        self.e4.setValidator(QIntValidator())
        #e1.setMaxLength(4)
       # self.e1.setAlignment(Qt.AlignRight)
       # self.e1.setFont(QFont("Arial",20))
        self.e4.editingFinished.connect(self.e4_finishEdit)
        #self.layout.addWidget(self.e4) 

      #  inputFieldsLayout.addWidget( self.l6,1,0)
      #  inputFieldsLayout.addWidget( self.e4,1,1)
        inputFieldsLayout.addRow("Exp time [us]", self.e4)


        #N. events
        #self.l77=QLabel(self)
        #self.l77.setText("N. shots")
        self.e5 = QLineEdit(self)
        self.e5.setValidator(QIntValidator())
        self.e5.editingFinished.connect(self.e5_finishEdit)

        #inputFieldsLayout.addWidget( self.l77,2,0)
        #inputFieldsLayout.addWidget( self.e5,2,1)
        inputFieldsLayout.addRow("N. shots ", self.e5)
        
        #N. loops
        #self.l78=QLabel(self)
        #self.l78.setText("N.loops")
        self.e6 = QLineEdit(self)
        self.e6.setValidator(QIntValidator())
        self.e6.editingFinished.connect(self.e6_finishEdit)

        #inputFieldsLayout.addWidget( self.l78,3,0)
        #inputFieldsLayout.addWidget( self.e6,3,1)
        inputFieldsLayout.addRow("n loops", self.e6)
        

        self.inputFieldsBox.setLayout(inputFieldsLayout) 
        

        
        #resume parameters
        
        self.l3=QLabel(self)
        self.l3.setText("selected file: ")
        #self.layout.addWidget(self.l3)

        self.l4=QLabel(self)
        self.l4.setText("out folder: ")
        #self.layout.addWidget(self.l4)

        #self.l7=QLabel(self)
        self.l7.setText("Gain: ")
        #self.layout.addWidget(self.l7)

        self.l8=QLabel(self)
        self.l8.setText("Exp.time: ")
        #self.layout.addWidget(self.l8)

        self.l9=QLabel(self)
        self.l9.setText("N_shots: ")

        self.l10=QLabel(self)
        self.l10.setText("N_loops: ")


         # pulsante finale!
        self.btn3 = QPushButton('GO!')
        self.btn3.clicked.connect(self.on_click3)

        
        #SET ELEMETS!!

        

        
        mainLayout.addWidget(self.l1,0,0)
        mainLayout.addWidget(self.e1,1,0,1,3)
        mainLayout.addWidget(self.btn1,1,4 )

        mainLayout.addWidget(self.l2,2,0 )
        mainLayout.addWidget(self.e2,3,0,1,3 )
        mainLayout.addWidget(self.btn2,3,4,1,1 )
      
        mainLayout.addWidget(self.inputFieldsBox,4,0)
        
        
        mainLayout.addWidget(self.l3,5,0 )
        mainLayout.addWidget(self.l4,6,0 )
        mainLayout.addWidget(self.l7,7,0 )
        mainLayout.addWidget(self.l8,8,0 )
        mainLayout.addWidget(self.l9,9,0 )
        mainLayout.addWidget(self.l10,10,0 )
       
        mainLayout.addWidget(self.btn3,11,0)
        
        
       
       
       
        # Add tabs to widget
       
       
        self.setLayout(mainLayout)
       # self.setLayout(inputFieldsLayout)
       
    @pyqtSlot()
    def on_click(self):
        self.pedFilePath=QFileDialog.getOpenFileName(self,"choose file")[0]
        self.l3.setText('selected file= '+self.pedFilePath)
        self.e1.setText(self.pedFilePath)
        print("filepath=",self.pedFilePath)

    def on_click2(self):
        self.outDataPath=QFileDialog.getExistingDirectory(self,"choose folder")
        self.l4.setText('out folder= '+self.outDataPath)
        self.e2.setText(self.outDataPath)
        print("filepath=",self.outDataPath)
             
    def e1_finishEdit(self):
        #print("Enter pressed")
        myText=self.e1.text()
        #print("text = ",myText)
        self.l3.setText('selected file= '+myText)
        self.pedFilePath=myText
        
    def e2_finishEdit(self):
        #print("Enter pressed")
        self.outDataPath=self.e2.text()
        self.outDataPath=self.outDataPath+'/'
        self.l4.setText('out folder '+ self.outDataPath)

    def e3_finishEdit(self,pippo):
        print("pippo=",pippo)
        self.gain=self.e3.text()
        self.l7.setText('Gain '+ self.gain)

    
    def e4_finishEdit(self):
        #print("Enter pressed")
        self.ExpTime=self.e4.text()
        self.l8.setText('Exp.time '+ self.ExpTime)

    def e5_finishEdit(self):
        #print("Enter pressed")
        self.Nshots=self.e5.text()
        self.l9.setText('N_shots '+ self.Nshots)

    def e6_finishEdit(self):
        #print("Enter pressed")
        self.Nloops=int(self.e6.text())
        self.l10.setText('N_loops '+ str(self.Nloops))

        
    def on_click3(self):
       print("starting acqusition!!!!")
       nLoops=100
       acq_loop.run_acq_loop(self.ExpTime,self.gain, self.Nshots, self.Nloops,    self.outDataPath, self.pedFilePath)


       
    def lineEdit_finshGenertic(self,e,var,label):
        
        var=e.text()
        label.setText(var)
       
       
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
