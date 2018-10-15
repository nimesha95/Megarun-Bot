import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
from models import inception_v3 as googlenet
from getkeys import key_check
from collections import deque, Counter
import random
from statistics import mode,mean
import numpy as np

GAME_WIDTH = 300
GAME_HEIGHT = 560

WIDTH = 30
HEIGHT = 56
LR = 1e-3
EPOCHS = 10

choices = deque([], maxlen=5)
hl_hist = 250
choice_hist = deque([], maxlen=hl_hist)

w = [1,0,0,0,0]
s = [0,1,0,0,0]
a = [0,0,1,0,0]
d = [0,0,0,1,0]
nk= [0,0,0,0,1]

def straight():
    print("w") 
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    PressKey(W)

def left():
    print("a") 
    ReleaseKey(S)
    ReleaseKey(D)
    ReleaseKey(w)
    PressKey(A)


def right():
    print("d") 
    ReleaseKey(A)
    ReleaseKey(w)
    ReleaseKey(S)
    PressKey(D)

    
def reverse():
    print("s") 
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def no_keys():
    print("nk") 
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    

print('going to load the previous model!!!!')

model = googlenet(WIDTH, HEIGHT, 3, LR, output=5)
MODEL_NAME = 'VM_Model_30_epoch'
model.load(MODEL_NAME)

print('We have loaded a previous model!!!!')

def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    mode_choice = 0

    screen = grab_screen(region=(0,40,GAME_WIDTH,GAME_HEIGHT+40))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    prev = cv2.resize(screen, (30,56))
    screen =prev

    t_minus = prev
    t_now = prev
    t_plus = prev

    while(True):
        
        if not paused:
            screen = grab_screen(region=(0,40,GAME_WIDTH,GAME_HEIGHT+40))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            
            last_time = time.time()
            screen = cv2.resize(screen, (WIDTH,HEIGHT))
            
            prediction = model.predict([screen.reshape(WIDTH,HEIGHT,3)])[0]
            prediction = np.array(prediction) * np.array([0.5, 1, 1, 1, 1.8])
            print(prediction) 

            mode_choice = np.argmax(prediction)

            if mode_choice == 0:
                straight()
                choice_picked = 'straight'
                
            elif mode_choice == 1:
                reverse()
                choice_picked = 'reverse'
                
            elif mode_choice == 2:
                left()
                choice_picked = 'left'
            elif mode_choice == 3:
                right()
                choice_picked = 'right'
         
            elif mode_choice == 4:
                no_keys()
                choice_picked = 'nokeys'

        keys = key_check()

        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

main()       
