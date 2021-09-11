import numpy as np
import json
#导入socket库
import socket
#建立IPv4,UDP的socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#绑定端口：
s.bind(('127.0.0.3', 9999))
#不需要开启listen，直接接收所有的数据
print('Bind UDP on 9999')
while True:
    #接收来自客户端的数据,使用recvfrom
    data, addr = s.recvfrom(1024)
    print('Received from %s:%s.' % addr)
    data=np.fromstring(data)#,np.uint8)
    #data=data.decode('utf-8')
    #data_json=json.loads(data)
    data.reshape(-1,5)
    print(data)
    #s.sendto(b'hello, %s!' % data, addr)
    s.sendto(data,addr)
