import cv2
import torch
import argparse
import os
import numpy as np
import json
import sys
sys.path.append("../")

from models import AttributeNet
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

def Path_Image_Preprocessing(path):
    '''
    Precessing the input images
        image_dir: single image input path, such as "/home/xxx/10.jpg"
    '''
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])
    image = cv2.imread(path)
    if image is None:
        return None
    image = cv2.resize(image,(224,224))
    image = image.astype(np.float32)
    image -= mean_bgr
    # H * W * C   -->   C * H * W
    image = image.transpose(2,0,1)
    image = torch.tensor(image)
    image = image.unsqueeze(0)
    return image

def main(args):
    device = args.device
    
    mkdir(args.save_dir)
    
    model = AttributeNet() 
    
    model.set_idx_list(attribute=["Male","Female","Young","Middle Aged","Senior","Asian","White","Black"])
    
    model.to(device)
    
    image_names = os.listdir(args.image_dir)
    
    for image_name in tqdm(image_names):
        image_path = os.path.join(args.image_dir, image_name)
        
        input_data = Path_Image_Preprocessing(image_path)
        
        predicted = model(input_data.to(device))
        # predicted = torch.sigmoid(predicted)
        
        predicted = (predicted > 0.5).int()

        with open(os.path.join(args.save_dir, "vgg-attribute-model-label.txt"),"a") as file:
            file.write("{} {} {} {} {} {} {} {} {}\n".format(image_path ,predicted[0][0], predicted[0][1],predicted[0][2],predicted[0][3],predicted[0][4],predicted[0][5],predicted[0][6],predicted[0][7]))
    
if __name__ == '__main__':

    args = parse_args()  # n, gpu
   
    main(args)