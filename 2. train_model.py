import os, cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import deque
from models import inception_v3 as googlenet
from random import shuffle


FILE_I_END = 1860

WIDTH = 30	#image width after resizing
HEIGHT = 56	#image height after resizing
LR = 1e-3
EPOCHS = 30

MODEL_NAME = 'new_model_30epochs'
PREV_MODEL = ''
LOAD_MODEL = False

model = googlenet(WIDTH, HEIGHT, 3, LR, output=5, model_name=MODEL_NAME)

if LOAD_MODEL:
    model.load(PREV_MODEL)
    print('We have loaded a previous model!!!!')

 #iterates through the training files
for e in range(EPOCHS):
	file_name = 'The_training_data.npy'
	train_data = np.load(file_name)
	print('training_data.npy', len(train_data))

	train = train_data[:-50]
	test = train_data[-50:]

	X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 3)
	Y = [i[1] for i in train]

	test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 3)
	test_y = [i[1] for i in test]

	model.fit({'input': X}, {'targets': Y}, n_epoch = 1, validation_set = ({'input': test_x}, {'targets': test_y}),
		snapshot_step = 2500, show_metric = True, run_id = MODEL_NAME)

	print('SAVING MODEL!')
	model.save(MODEL_NAME)

#tensorboard --logdir=foo:J:/phase10-code/log
