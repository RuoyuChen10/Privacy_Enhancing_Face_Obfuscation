import cv2
import torch
import argparse
import os
import numpy as np
import json
import sys
sys.path.append("../")

from models import BranchedTinyAttr
from utils import load_yaml, read_img
from tqdm import tqdm
import random

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--attribute', type=str, 
                        default='Female',
                        choices=[   
                            "Male","Female","Young","Middle Aged","Senior","Asian","White","Black","Bald","Wavy Hair",
                            "Receding Hairline","Bangs","Sideburns","Black Hair","Blond Hair","Brown Hair","Gray Hair","no beard",
                            "Mustache","5 o Clock Shadow","Goatee","Oval Face","Square Face",
                            "Round Face","Double Chin","High Cheekbones","Chubby","Obstructed Forehead",
                            "Fully Visible Forehead","Brown Eyes","Bags Under Eyes","Bushy Eyebrows","Arched Eyebrows",
                            "Mouth Closed","Smiling","Big Lips","Big Nose","Pointy Nose"
                        ])
    parser.add_argument('--image-dir', type=str, default='/home/lsf/桌面/UTK-Face/RISE/real')  
    parser.add_argument('--save-dir', type=str, default='/RISE/')  
    parser.add_argument('--device', type=str, default='cuda:0')      # json path

    args = parser.parse_args()
    return args

VGG2CelebA = {
    "Sideburns": 'sideburns',
    "Bald": 'bald',
    "Goatee": 'goatee', 
    "Mustache": 'mustache',
    "5 o Clock Shadow": '5_o_clock_shadow', 
    "Arched Eyebrows": 'arched_eyebrows', 
    "no beard": 'no_beard', 
    "Male": 'male',
    "Black Hair": 'black_hair', 
    "High Cheekbones": 'high_cheekbones', 
    "Smiling": 'smiling',
    "Oval Face": 'oval_face', 
    "Bushy Eyebrows": 'bushy_eyebrows',
    "Young": 'young', 
    "Gray Hair": 'gray_hair', 
    "Brown Hair": 'brown_hair', 
    "Blond Hair": 'blond_hair', 
    "Chubby": 'chubby',
    "Double Chin": 'double_chin', 
    "Big Nose": 'big_nose', 
    "Bags Under Eyes": 'bags_under_eyes', 
    "Bangs": 'bangs', 
    "Wavy Hair": 'wavy_hair', 
    "Big Lips": 'big_lips',
    "Pointy Nose": 'pointy_nose', 
    "Receding Hairline": 'receding_hairline',
}

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

def main(args):
    device = args.device
    
    mkdir(args.save_dir)
    
    cfg = load_yaml("/home/lsf/桌面/MaskFaceGAN/config.yml")
    model = BranchedTinyAttr(cfg.MODELS.CLASSIFIER)
    
    model.set_idx_list(attributes=['male', "young"])
    
    model.to(device)
    
    image_names = os.listdir(args.image_dir)
    
    for image_name in tqdm(image_names):
        image_path = os.path.join(args.image_dir, image_name)
        
        input_data = read_img(image_path)
        
        predicted = model(input_data.to(device))
        predicted = torch.sigmoid(predicted)
        
        predicted = (predicted > 0.5).int()

        with open(os.path.join(args.save_dir, "celeba-attribute-model-label.txt"),"a") as file:
            file.write("{} {} {}\n".format(image_path ,predicted[0][0], predicted[0][1]))
    
if __name__ == '__main__':

    args = parse_args()  # n, gpu
   
    main(args)