import os, time
import multiprocessing



def fifo_writer(pipe_name ):
    print('FifoWriter pipe:',pipe_name)
    fifo = open(pipe_name, 'w')
    counter = 0
    while counter < 50:
        print('FifoWriter writing {} \n'.format(counter))
        time.sleep(1)
        fifo.write('da')
        counter += 1


def fifo_reader(pipe_name ):
    print('FifoReader pipe:', pipe_name)
    fifo = open(pipe_name, 'r')
    counter = 0
    while counter < 50:
        time.sleep(1.5)
        print('FifoReader reading', counter)
        line = fifo.read()
        print('FifoReader %d got "%s" at %s' % (os.getpid(), line, time.time( )))

        counter +=1


pipe_name = 'pipe_test.fifo'
os.mkfifo(pipe_name)

fifo_writer(pipe_name)


# multiprocessing.Process(target=child,args=[pipe_name]).start()
# multiprocessing.Process(target=parent,args=[pipe_name]).start()
