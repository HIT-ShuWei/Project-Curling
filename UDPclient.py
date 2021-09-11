import socket
import numpy as np
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#不需要建立连接：
data=np.array([[640.8,2,3,1,2],[4,5,6,4,5],[7,8,9,8,9]])
data = data.tostring()
#for data in [b'Michael', b'ALice', b'FF']:
    #发送数据到客户端：
s.sendto(data, ('127.0.0.3', 9999))
    #接收来自客户端的数据：
print(s.recvfrom(1024)[0].decode('utf-8'))
s.close()

