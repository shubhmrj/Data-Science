import threading
import time 

def letter():
    for i in "abcd":
        time.sleep(2)
        print(f"letter:{i}")
       
def number():
    for i in range(5):
        time.sleep(2)
        print(f"number:{i}")
        
t1=threading.Thread(target=letter)   
t2=threading.Thread(target=number)

current_time=time.time()
# start the thread
t1.start()
t2.start()  
# wait for thread to complete
t1.join()
t2.join() 
 
used_time=time.time()-current_time 
print(used_time)   