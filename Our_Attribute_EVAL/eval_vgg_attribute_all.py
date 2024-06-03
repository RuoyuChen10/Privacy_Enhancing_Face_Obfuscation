# -*- coding: UTF-8 -*-


import cv2
import torch
import argparse
import os
import numpy as np
import json
import sys
sys.path.append("../")

from models import AttributeNet, BranchedTinyAttr
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
    parser.add_argument('--attribute-net', type=str, default="VGGAttributeNet", choices=["VGGAttributeNet", "BranchedTinyAttr"])     
    parser.add_argument('--test-set', type=str, default='vgg-attribute-model-label.txt')     
    parser.add_argument('--save-dir', type=str, default='/RISE/')  
    parser.add_argument('--device', type=str, default='cuda:0')      # json path

    args = parser.parse_args()
    return args

StasticData = [
    ["Male"],
    ["Female"],
    ["Young"],
    ["Middle Aged"],
    ["Senior"],
    ["White"],
    ["Asian"],
    ["Black"],
    ["Male", "Young"],          # Gender-Age
    ["Male", "Middle Aged"],
    ["Male", "Senior"],
    ["Female", "Young"],
    ["Female", "Middle Aged"],
    ["Female", "Senior"],
    ["Male", "White"],          # Gender-Race
    ["Male", "Asian"],
    ["Male", "Black"],
    ["Female", "White"],          
    ["Female", "Asian"],
    ["Female", "Black"],
    ["Young", "White"],         # Age-Race
    ["Middle Aged", "White"],
    ["Senior", "White"],
    ["Young", "Asian"],
    ["Middle Aged", "Asian"],
    ["Senior", "Asian"],
    ["Young", "Black"],
    ["Middle Aged", "Black"],
    ["Senior", "Black"],
    ["Male", "Young", "White"], # Original
    ["Male", "Middle Aged", "White"],
    ["Male", "Senior", "White"],
    ["Male", "Young", "Asian"], 
    ["Male", "Middle Aged", "Asian"],
    ["Male", "Senior", "Asian"],
    ["Male", "Young", "Black"], 
    ["Male", "Middle Aged", "Black"],
    ["Male", "Senior", "Black"],
    ["Female", "Young", "White"],
    ["Female", "Middle Aged", "White"],
    ["Female", "Senior", "White"],
    ["Female", "Young", "Asian"],
    ["Female", "Middle Aged", "Asian"],
    ["Female", "Senior", "Asian"],
    ["Female", "Young", "Black"],
    ["Female", "Middle Aged", "Black"],
    ["Female", "Senior", "Black"]
]

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

desired_attribute = [
    "Male","Female","Young","Middle Aged","Senior","Asian","White","Black","Bald","Wavy Hair",
    "Receding Hairline","Bangs","Sideburns","Black Hair","Blond Hair","Brown Hair","Gray Hair","no beard",
    "Mustache","5 o Clock Shadow","Goatee","Oval Face","Square Face",
    "Round Face","Double Chin","High Cheekbones","Chubby","Obstructed Forehead",
    "Fully Visible Forehead","Brown Eyes","Bags Under Eyes","Bushy Eyebrows","Arched Eyebrows",
    "Mouth Closed","Smiling","Big Lips","Big Nose","Pointy Nose"
]

Face_attributes_name = [
    "Gender","Age","Race","Bald","Wavy Hair",
    "Receding Hairline","Bangs","Sideburns","Hair color","no beard",
    "Mustache","5 o Clock Shadow","Goatee","Oval Face","Square Face",
    "Round Face","Double Chin","High Cheekbones","Chubby","Obstructed Forehead",
    "Fully Visible Forehead","Brown Eyes","Bags Under Eyes","Bushy Eyebrows","Arched Eyebrows",
    "Mouth Closed","Smiling","Big Lips","Big Nose","Pointy Nose"
]

Gender = ["Male","Female"]
Age = ["Young","Middle Aged","Senior"]
Race = ["Asian","White","Black"]
Hair_color = ["Black Hair","Blond Hair","Brown Hair","Gray Hair","Unknown Hair"]

def get_idx(attribute_name):

    if attribute_name == "Male":
        return 0
    elif attribute_name == "Female":
        return 1
    elif attribute_name == "Young":
        return 2
    elif attribute_name == "Middle Aged":
        return 3
    elif attribute_name == "Senior":
        return 4
    elif attribute_name == "Asian":
        return 5
    elif attribute_name == "White":
        return 6
    elif attribute_name == "Black":
        return 7

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

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

def main(args):

    device = args.device#"cpu"
    
    mkdir(args.save_dir)
    
    model = AttributeNet() 
    
    model.to(device)
    
    for stasticdata in StasticData:
        model.set_idx_list(stasticdata)
        
        f=open(args.test_set)
        datas = f.readlines()  
        f.close()  

        acc = 0
        number = 0

        TP = 0
        TP_num = 0

        for data in tqdm(datas):
            data = data.strip() 

            image_path = data.split(" ")[0].replace("_orig", "")

            input_data = Path_Image_Preprocessing(image_path)
            
            if input_data is None:
                continue
            
            input_data = input_data.to(device)

            predicted = model(input_data)

            predicted_truths = predicted[0]>0.5

            TP_J = 1
            ACC_J = 1
            
            for i, attribute in enumerate(stasticdata):

                attr_idx = get_idx(attribute_name = attribute)
                attr_label = int(data.split(" ")[attr_idx + 1])
                predicted_truth = int(predicted_truths[i].item())
                
                if predicted_truth == False:
                    TP_J = 0
                    
                if predicted_truth != attr_label:
                    ACC_J = 0
                    
            number += 1
            if TP_J:
      
                TP_num += 1

            if ACC_J:
                acc += 1
                if TP_J:
            
                    TP +=1

        attr_names = stasticdata[0]
        for an in stasticdata[1:]:
            attr_names += "-"
            attr_names += an

        if TP_num == 0:
            TP_num = 0.000000001
        
        print("Done!The number of images are {}. The attributes are {}. The True positive is {}. The accuracy is {}. Instance number {}.".format(number, attr_names, TP/(TP_num), acc/number, TP_num))

        with open(os.path.join(args.save_dir, "VGG-ATTR-RESULT.txt"),"a") as file:
            file.write("{}, Instance number: {}, TP: {}, ACC: {}. \n".format(attr_names, TP_num, TP/(TP_num), acc/number))
                
    # with open("CelebA-ATTR-RESULT.txt","a") as file:
    #     file.write("Male, TP: {}, ACC: {}.\n".format(male_TP/male_T_num, male_acc/number))
    #     file.write("Female, TP: {}, ACC: {}.\n".format(female_TP/female_T_num, female_acc/number))
    #     file.write("Young, TP: {}, ACC: {}.\n".format(young_TP/young_T_num, young_acc/number))
    #     file.write("Male-Young, TP: {}, ACC: {}.\n".format(male_young_TP/male_young_T_num, male_young_acc/number))
    #     file.write("Female-Young, TP: {}, ACC: {}.\n".format(female_young_TP/female_young_T_num, female_young_acc/number))
    #     file.write("Male-Old, TP: {}, ACC: {}.\n".format(male_old_TP/male_old_T_num, male_old_acc/number))
    #     file.write("Female-Old, TP: {}, ACC: {}.\n".format(female_old_TP/female_old_T_num, female_old_acc/number))
        
if __name__ == '__main__':

    args = parse_args()  # n, gpu
   
    main(args)