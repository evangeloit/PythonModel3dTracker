import os, time


pipe_name = 'pipe_test.fifo'

print('FifoReader pipe:', pipe_name)
fifo = open(pipe_name, 'r')
counter = 0
while counter < 50:
    time.sleep(1.5)
    print('FifoReader reading', counter)
    line = fifo.read()
    print('FifoReader %d got "%s" at %s' % (os.getpid(), line, time.time()))

    counter += 1
