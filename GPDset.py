import os
import subprocess

class MyDaq:
    def __init__(self, gpd=35):
        self.mask = self.TrgMask(gpd)
        self.Process = 0
        #self.Bat()

    def Acquire(self, time=300):
        command='ixpedaq -v -b -s '+ str(time) + ' -m ' + self.mask
        #os.system(f"start /wait /b cmd /c {command}")
        self.Process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.Process.communicate()
        print(command)
        #print(stdout.decode())
        #if stderr:
        #    print(stderr.decode())

    def Bat(self):
        os.system("C:/xro/gpdsw/setup.bat")
        #subprocess.run("setup.bat", shell=True)
        print()
        return 1

    def PrintOut(self, log_file):
        stdout, stderr = self.Process.communicate()
        log_file.write(stdout.decode())
        if stderr:
            log_file.write(stderr.decode())

    def TrgMask(self, gpd):
        path = 'C:/xro/gpdsw/Daq/config/trg_'+ str(gpd) +'.mask'
        return path
