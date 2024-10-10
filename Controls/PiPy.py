from pipython import GCSDevice

def Connect():
    master = GCSDevice()
    master.OpenRS232DaisyChain(comport=4, baudrate=115200)
    daisychainid = master.dcid
    master.ConnectDaisyChainDevice(1, daisychainid)
    c12 = GCSDevice()
    c12.ConnectDaisyChainDevice(12, daisychainid)
    c13 = GCSDevice()
    c13.ConnectDaisyChainDevice(13, daisychainid)
    #print('\n{}:\n{}'.format(master.GetInterfaceDescription(), master.qIDN()))
    #print('\n{}:\n{}'.format(c12.GetInterfaceDescription(), c12.qIDN()))
    #print('\n{}:\n{}'.format(c13.GetInterfaceDescription(), c13.qIDN()))
    print(c12)
    return master, c12, c13

def Close(device):
    device.CloseConnection()
    return 0

master, yUp, xUp = Connect()
yUp.MOV(1,20)
Close(yUp)
Close(xUp)
Close(master)
