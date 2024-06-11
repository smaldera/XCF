import subprocess
import glob
from datetime import datetime as dt

def callps1(path='./', livetime='10', name=''):
    powerShellPath = r'C:\WINDOWS\system32\WindowsPowerShell\v1.0\powershell.exe'
    powerShellCmd = './script.ps1'
    print('Ready')
    #call script with argument '/mnt/c/Users/aaa/'
    p = subprocess.Popen([powerShellPath, '-ExecutionPolicy', 'Unrestricted', powerShellCmd, path, livetime, name]
                         , stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = p.communicate()
    return 1

def CheckForFile(path):
    print("checking...")
    path = 'C:/Users/XCF/'+path
    while True:
        check = glob.glob(path+f'/file_{dt.now().year}_{dt.now().month}_{dt.now().day}_{dt.now().hour}_{dt.now().minute}.npz')
        if len(check)>0:
            print('Done')
            return 1
