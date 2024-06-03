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
    parser.add_argument('--image-dir', type=str, default='/exdata2/RuoyuChen/Datasets/VGGface2_None_norm_512_true_bygfpgan/')  
    parser.add_argument('--attribute-net', type=str, default="VGGAttributeNet", choices=["VGGAttributeNet", "BranchedTinyAttr"])     
    parser.add_argument('--attribute-set', type=str, default='/exdata2/RuoyuChen/Datasets/VGGFace2/attribute')    
    parser.add_argument('--test-set', type=str, default='/home/lsf/桌面/MaskFaceGAN/utils/attribute2.txt')     
    parser.add_argument('--json-path', type=str, default='/home/lsf/桌面/MaskFaceGAN/json_per_json')      # json path
    parser.add_argument('--convert-data', type=bool, default=False)     
    parser.add_argument('--test-number', type=int, default=-1)   

    parser.add_argument('--device', type=str, default='cuda:0')      # json path

    args = parser.parse_args()
    return args

StasticData = [
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

    if attribute_name in Gender:
        return 0
    elif attribute_name in Age:
        return 1
    elif attribute_name in Race:
        return 2
    elif attribute_name in Hair_color:
        return 8
    else:
        return Face_attributes_name.index(attribute_name)

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

def convert_(args, image_names, type_="all.txt"):
    f=open(os.path.join(args.attribute_set, type_))
    datas = f.readlines()  
    f.close()

    for i in tqdm(range(len(datas))):
        people_name = datas[i].split(" ")[0]
        
        if people_name in image_names:
            image_names.remove(people_name)
            with open(args.test_set, "a") as file:
                file.write(datas[i])


    for image_name in tqdm(image_names):
        for i in range(len(datas)):
            people_name = datas[i].split(" ")[0]
            if people_name.split("/")[0] == image_name.split("/")[0]:
                with open(args.test_set, "a") as file:
                    file.write(datas[i].replace(people_name, image_name))
                break

def convert(args):
    image_names = []


    peoples = os.listdir(args.json_path)

    for people in peoples:
        people_path = os.path.join(args.json_path, people)
        json_files = os.listdir(
            people_path
        )

        for json_file in json_files:
            image_name_path = os.path.join(people, json_file.split("-")[0]+".jpg")
            
            image_names.append(image_name_path)
    

    convert_(args, image_names, type_="all.txt")

def main(args):

    device = args.device#"cpu"
    if args.attribute_net == "VGGAttributeNet":
        model = AttributeNet() 

    # elif args.attribute_net == "BranchedTinyAttr":
    #     cfg = load_yaml("/exdata2/RuoyuChen/Demo/MaskFaceGAN/config.yml")
    #     model = BranchedTinyAttr(cfg.MODELS.CLASSIFIER)
    #     if args.attribute == "Female":
    #         model.set_idx_list(attributes=[VGG2CelebA["Male"]])
    #     else:
    #         model.set_idx_list(attributes=[VGG2CelebA[args.attribute]])
    
    model.to(device)

    for stasticdata in StasticData:
        model.set_idx_list(stasticdata)
        # for attribute in stasticdata:

      
        f=open(args.test_set)
        datas = f.readlines()  
        f.close()  
        # random.shuffle(datas)

        acc = 0
        number = 0

        TP = 0
        TP_num = 0

        peoples = os.listdir(args.json_path)
        '''
        for people in peoples:
            people_path = os.path.join(args.json_path, people)
            json_files = os.listdir(people_path)

            for json_file in json_files:
                json_file_path = os.path.join(people_path, json_file)
                new_dict = json.load(open(json_file_path, 'r', encoding='utf-8'))
            

                ### new ####
                image_name = new_dict["ima"]
                #print(image_name)
                
                image_name = image_name.split("/")[1]

                image_path=os.path.join(args.image_dir, image_name)
        '''          

        for data in tqdm(datas):
            data = data.strip() 
            #print("data",data.split(".")[0])
            image_path = os.path.join(args.image_dir, data.split(" ")[0])
            
            #im_name = data.split("/")[0]+"_"+data.split("/")[1]
            im_name = data.split(" ")[0]
            im_json = im_name.split(".")[0]
            json_path = args.json_path+'/'+im_json+'.json'
            #print(json_path)
            if not os.path.exists(json_path):
               continue  
            new_dict = json.load(open(args.json_path+'/'+im_json+'.json', 'r', encoding='utf-8'))
            image_name = new_dict["ima"]
            image_name = image_name.split("/")[1]
            image_path = os.path.join(args.image_dir,image_name)
            #print(image_path)
            #print(im_namg)        
            if not os.path.exists(image_path):
                continue        

            if args.attribute_net == "VGGAttributeNet":
                input_data = Path_Image_Preprocessing(image_path)
            # elif args.attribute_net == "BranchedTinyAttr":
            #     input_data = read_img(image_path)
            
            if input_data is None:
              
                continue
            input_data = input_data.to(device)

            predicted = model(input_data)

            # if args.attribute_net == "BranchedTinyAttr":
            #     predicted = torch.sigmoid(predicted)
            #     if args.attribute == "Female":
            #         predicted = 1-predicted

            predicted_truths = predicted[0]>0.8

            TP_J = 1
            ACC_J = 1

            for i, attribute in enumerate(stasticdata):

                attr_idx = get_idx(attribute_name = attribute)
                attr_label = int(data.split(" ")[attr_idx + 1])
                predicted_truth = predicted_truths[i].item()

                if predicted_truth == False:
                    TP_J = 0

                if attribute in Gender:
                    gt_label = Gender.index(attribute)
                    gt_judge = (gt_label == attr_label)

                    if predicted_truth != gt_judge:
                        # print(predicted[0][0].item(), predicted_truth, gt_label, attr_label, gt_judge, gt_label == attr_label)
                        # 0.9971669316291809 True 0 0 True True
                        # 0.0008440228411927819 False 0 1 False False
               
                        ACC_J = 0
                        break

                elif attribute in Age:
                    gt_label = Age.index(attribute)
                    gt_judge = (gt_label == attr_label)
                    if predicted_truth != gt_judge:
                        ACC_J = 0
                        break

                elif attribute in Race:
                    gt_label = Race.index(attribute)
                    gt_judge = (gt_label == attr_label)
                    if predicted_truth != gt_judge:
                        ACC_J = 0
                        break


                elif attribute in Hair_color:
                    gt_label = Hair_color.index(attribute)
                    gt_judge = (gt_label == attr_label)
                    if predicted_truth != gt_judge:
                        ACC_J = 0
                        break

                else:
                    if attr_label == 2:
               
                        continue
                    if attr_label == 0:
                        gt_judge = True
                    elif attr_label == 1:
                        gt_judge = False
                    if predicted_truth != gt_judge:
                        ACC_J = 0
                        break
                  
            number += 1
            if TP_J:
      
                TP_num += 1

            if ACC_J:
                acc += 1
                if TP_J:
            
                    TP +=1

            if number == args.test_number:
                break

        attr_names = stasticdata[0]
        for an in stasticdata[1:]:
            attr_names += ", "
            attr_names += an

        if TP_num == 0:
            TP_num = 0.000000001
        
        print("Done!The number of images are {}. The attributes are {}. The True positive is {}. The accuracy is {}.".format(number, attr_names, TP/(TP_num), acc/number))

        with open("all_dp.log","a") as file:
            file.write("Done!The number of images are {}. The attributes are {}. The True positive is {}. The accuracy is {}. \n".format(number, attr_names, TP/(TP_num), acc/number))
        
if __name__ == '__main__':

    args = parse_args()  # n, gpu
  
    if not os.path.exists(args.test_set) or args.convert_data:
        if os.path.exists(args.test_set):
            os.remove(args.test_set) 
        convert(args)
    
   
    main(args)