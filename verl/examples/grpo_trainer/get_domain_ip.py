import os
import sys
import socket
import time
# 需要解析的域名
domain = sys.argv[1]
import subprocess

# 要执行的命令
domain = sys.argv[1]
while True:
    try:
        # 获取域名的IP地址
        ip_address = socket.gethostbyname(domain)
        print(ip_address)
        break
    except socket.error as e:
        sys.stderr.write("Error: %s\n" % e)
        time.sleep(1)
