import zmq
import random
import sys
import time
import multiprocessing


port = "5556"
context = zmq.Context()

def server(port):
    socket = context.socket(zmq.PAIR)
    socket.bind("tcp://*:%s" % port)
    int_list = [5, 3, 23]
    while True:
        #socket.send_string("Server message to client3")
        socket.send_pyobj(int_list)
        #gui_command = socket.recv()
        #print(gui_command)
        time.sleep(1)

def client(port):
    socket = context.socket(zmq.PAIR)
    socket.connect("tcp://localhost:%s" % port)

    while True:
        msg = socket.recv_pyobj()
        print(msg[0])
        #socket.send_string("client message to server1")
        #socket.send_string("client message to server2")
        time.sleep(1)



multiprocessing.Process(target=server,args=[port]).start()
multiprocessing.Process(target=client,args=[port]).start()