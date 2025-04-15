
import socket 
from threading import Thread 
from socketserver import ThreadingMixIn
import os
import pickle
import cv2

running = True

def startEdgeServer():
    class EdgeThread(Thread): 
 
        def __init__(self,ip,port, count): 
            Thread.__init__(self) 
            self.ip = ip 
            self.port = port
            self.count = count
            print('Request received from Client IP : '+ip+' with port no : '+str(port)+"\n") 
 
        def run(self): 
            data = conn.recv(1000000000)
            dataset = pickle.loads(data)
            request = dataset[0]
            if request == "sensordata":
                label = dataset[1]
                img = dataset[2]
                print("Received sensor data as : "+label)
                cv2.imwrite("files/"+str(count)+"_"+label+".png",img)
                        
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    server.bind(('localhost', 2222))
    threads = []
    print("Edge Cloud Server Started\n\n")
    count = 0
    while running:
        server.listen(4)
        (conn, (ip,port)) = server.accept()
        newthread = EdgeThread(ip,port,count) 
        newthread.start() 
        threads.append(newthread)
        count = count + 1
    for t in threads:
        t.join()

def startCloud():
    Thread(target=startEdgeServer).start()


startCloud()


