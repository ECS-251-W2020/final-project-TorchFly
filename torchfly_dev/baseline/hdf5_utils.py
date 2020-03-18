from glob import glob
import json
import os
import pickle
import string

import text2vec
import h5py
import nltk
import numpy as np
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
import random
from math import sqrt

def smooth_text(text, dominant_word):
    """
    smooth the text by removing the noisy words
    """
    noisyword = [
        ([0, 0, 0],156), 
        ([156, 156, 156], 156),
        ([154, 154, 154], 154), 
        ([134, 134, 134], 134), 
        ([149, 149, 149], 149), 
        ([126, 126, 126], 126), 
        ([105, 105, 105], 105), 
        ([14, 14, 14], 14), 
        ([124, 124, 124], 124), 
        ([158, 158, 158], 158), 
        ([147, 147, 147], 147), 
        ([96, 96, 96], 96),
        ([168, 168, 168], 168), 
        ([148, 148, 148], 148), 
        ([110, 110, 110], 110), 
        ([135, 135, 135], 135), 
        ([119, 119, 119], 119), 
        ([161, 161, 161], 161),
        ([177, 177, 177], 177), 
        ([118, 118, 118], 118), 
        ([123, 123, 123], 123), 
        ([162, 162, 162], 162), 
    ]

    center = []
    for l in dominant_word:
        center_array = np.array([l]*3)
        center.append(np.uint8(center_array))
    #print(center)
    for i in range(text.shape[0]):
        for j in range(text.shape[1]):
            current_word = np.uint8(text[i,j])
            #sort centers
           
            if not any(all(current_word == x) for x in center):
                #print("sort_center")
                center.sort(key=lambda c: sqrt((current_word[0]-c[0])**2+(current_word[1]-c[1])**2+(current_word[2]-c[2])**2))
                text[i,j] = center[0]
    #print(text)
    return text

def create_h5(split, data_path, h5_path, resize_wh=128, data_augmentation=False):
    if split == "train":
        #load the json file
        with open('/'.join([data_path, "train.json"]), 'r') as f:
            split_json = json.load(f)
        #initialize the hdf5 files
        h5_split = h5py.File(os.path.join(h5_path, 'gandraw_train.h5'), 'w')
    if split == "val":
        with open('/'.join([data_path, "val.json"]), 'r') as f:
            split_json = json.load(f)
        h5_split = h5py.File(os.path.join(h5_path, 'gandraw_val.h5'), 'w')
    if split == "test":
        with open('/'.join([data_path, "test.json"]), 'r') as f:
            split_json = json.load(f)
        h5_split = h5py.File(os.path.join(h5_path, 'gandraw_test.h5'), 'w')
         
    c_split = 0
    
    for scene_id, scene in tqdm(enumerate(split_json['data'])):
        text = []
        text_semantic = []
        utterences = []
        target_text = []
        target_text_segmentation  = []
        target_text_path = []

        description = []
        for i in range(len(scene['dialog'])):
            turn = scene['dialog'][i]
            #lower case all messages
            data2 = str.lower(turn['data2'])
            data1 = str.lower(turn['data1'])

            #The information will always be alteranating between data2 and data1   
            if data2 != '':
                description += ['<data2>'] + nltk.word_tokenize(data2)
            if data1 != '':
                description += ['<data1>'] + nltk.word_tokenize(data1)

            description = [w for w in description if w not in string.punctuation]
            utterences.append(str.join(' ', description))

            current_turn_text = text2vec.txtread(os.path.join(data_path, turn['text_synthetic']))
            current_turn_text = preprocessing_text(current_turn_text, resize_wh=resize_wh) #The text is converted from BGR2RGB and the size becomes 128*128
            text.append(current_turn_text)
            
            #semantic_text_path = 'semantic_text/'+turn['text_semantic'].split('/')[-1]
            current_turn_text_semantic = text2vec.txtread(os.path.join(data_path, turn['text_semantic']))
            current_turn_text_semantic = preprocessing_text(current_turn_text_semantic, resize_wh=resize_wh, segmentation=True)
            #print(current_turn_text_semantic.shape)
            assert current_turn_text_semantic is not None, "os.path.join({}, {})".format(data_path, semantic_text_path)
            text_semantic.append(current_turn_text_semantic)
            
            description = []
            
        current_target_text =  text2vec.txtread(os.path.join(data_path, scene['target_text']))
        current_target_text = preprocessing_text(current_target_text, resize_wh=resize_wh)
        target_text.append(current_target_text)
        target_text_path.append(scene['target_text'])

        current_target_text_segmentation = text2vec.txtread(os.path.join(data_path, scene['target_text_semantic']))
        current_target_text_segmentation = preprocessing_text(current_target_text_segmentation, resize_wh=resize_wh, segmentation=True)
        target_text_segmentation.append(current_target_text_segmentation)

        scene_hdf5 = h5_split.create_group(str(c_split))
        c_split += 1
        
        #Add the task_id
        task_id = scene.get("task_id", None)
        
        scene_hdf5.create_dataset('text', data=text)
        scene_hdf5.create_dataset('text_semantic', data=text_semantic)
        if task_id is not None:
            scene_hdf5.create_dataset('scene_id', data=task_id)
        else:
            scene_hdf5.create_dataset('scene_id', data=str(scene_id))
        scene_hdf5.create_dataset('target_text', data=target_text)
        scene_hdf5.create_dataset('target_text_segmentation', data = target_text_segmentation)
        dt = h5py.special_dtype(vlen=str)
        scene_hdf5.create_dataset('utterences', data=np.string_(utterences), dtype=dt)
        scene_hdf5.create_dataset('target_text_path', data=np.string_(target_text_path), dtype=dt)

    #increasing the data by one time with data augmentation
    if data_augmentation:
        for scene_id, scene in tqdm(enumerate(split_json['data'])):
            text = []
            text_semantic = []
            utterences = []
            target_text = []
            target_text_segmentation  = []
            target_text_path = []

            description = []
            
            #define data_augmentation_mode
            data_augmentation_mode = random.choice(["crop", "contrast", "noisy"])
            for i in range(len(scene['dialog'])):
                turn = scene['dialog'][i]
                #lower case all messages
                data2 = str.lower(turn['data2'])
                data1 = str.lower(turn['data1'])

                #The information will always be alteranating between data2 and data1   
                if data2 != '':
                    description += ['<data2>'] + nltk.word_tokenize(data2)
                if data1 != '':
                    description += ['<data1>'] + nltk.word_tokenize(data1)

                description = [w for w in description if w not in string.punctuation]
                utterences.append(str.join(' ', description))

                current_turn_text = text2vec.txtread(os.path.join(data_path, turn['text_synthetic']))
                #Augment the data
                current_turn_text = apply_augmentation(current_turn_text, mode=data_augmentation_mode)
                current_turn_text = preprocessing_text(current_turn_text, resize_wh=resize_wh) #The text is converted from BGR2RGB and the size becomes 128*128
                text.append(current_turn_text)

                #semantic_text_path = 'semantic_text/'+turn['text_semantic'].split('/')[-1]
                current_turn_text_semantic = text2vec.txtread(os.path.join(data_path, turn['text_semantic']))
                current_turn_text_semantic = preprocessing_text(current_turn_text_semantic, resize_wh=resize_wh, segmentation=True)
                assert current_turn_text_semantic is not None, "os.path.join({}, {})".format(data_path, semantic_text_path)
                text_semantic.append(current_turn_text_semantic)

                description = []

            current_target_text =  text2vec.txtread(os.path.join(data_path, scene['target_text']))
            current_target_text = preprocessing_text(current_target_text, resize_wh=resize_wh)
            target_text.append(current_target_text)
            target_text_path.append(scene['target_text'])

            current_target_text_segmentation = text2vec.txtread(os.path.join(data_path, scene['target_text_semantic']))
            current_target_text_segmentation = preprocessing_text(current_target_text_segmentation, resize_wh=resize_wh, segmentation=True)
            target_text_segmentation.append(current_target_text_segmentation)

            scene_hdf5 = h5_split.create_group(str(c_split))
            c_split += 1
            
             #Add the task_id
            task_id = scene.get("task_id", None)
            
            scene_hdf5.create_dataset('text', data=text)
            scene_hdf5.create_dataset('text_semantic', data=text_semantic)
            if task_id is not None:
                scene_hdf5.create_dataset('scene_id', data=task_id+"_DA")
            else:
                scene_hdf5.create_dataset('scene_id', data=str(scene_id+len(split_json['data'])))
            scene_hdf5.create_dataset('target_text', data=target_text)
            scene_hdf5.create_dataset('target_text_segmentation', data = target_text_segmentation)
            dt = h5py.special_dtype(vlen=str)
            scene_hdf5.create_dataset('utterences', data=np.string_(utterences), dtype=dt)
            scene_hdf5.create_dataset('target_text_path', data=np.string_(target_text_path), dtype=dt)
        