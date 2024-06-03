"""
陈若愚
2022.06.22

测试属性分类器的性能。
"""

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
    """
    程序分为两个步骤，转换与测试：
    
    第一个步骤，首先从json-path中获取我们需要测试的图像名称，并且从
    属性数据集attribute-set中查找是否由这样的标签，如果有，将数据
    写入到文件test-set中。
    转换步骤可以选择是否转换，如果先前已经转换完成了，可将参数
    convert-data设置为False，这样如果已经存在test-set文件，程序
    将不会进一步convert数据。当然如果想强制转换，可以设置convert-data
    为True

    第二个步骤，测试属性准确率。
    在给定test-set文件后（可以指定，也可以通过步骤一生成），可以通过
    第二部程序生成准确率。首先需要指定测试图像的文件夹image-dir，例如
    原始的VGGFace2还是VGGFace2Hq需要指定。然后需要指定测试的属性attribute
    之后就可以进行属性的测试了，给出一个准确率。目前只支持单个属性。
    
    如果测试集太大可以设置test-number来指定有效的测试数据，如果全部测试这个
    参数默认置1就行。

    指定运算设备，device。可以指定"cpu"，或者想要的gpu: "cuda:3"。目前版本不支持多卡运算
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--attribute', type=str, 
                        default='Male',
                        choices=[   # 属性名字请从这里选，选1个
                            "Male","Female","Young","Middle Aged","Senior","Asian","White","Black","Bald","Wavy Hair",
                            "Receding Hairline","Bangs","Sideburns","Black Hair","Blond Hair","Brown Hair","Gray Hair","no beard",
                            "Mustache","5 o Clock Shadow","Goatee","Oval Face","Square Face",
                            "Round Face","Double Chin","High Cheekbones","Chubby","Obstructed Forehead",
                            "Fully Visible Forehead","Brown Eyes","Bags Under Eyes","Bushy Eyebrows","Arched Eyebrows",
                            "Mouth Closed","Smiling","Big Lips","Big Nose","Pointy Nose"
                        ])
    parser.add_argument('--image-dir', type=str, default='/exdata2/RuoyuChen/Datasets/VGGface2_None_norm_512_true_bygfpgan/')    # VGGFace2数据集的文件夹，这个放置固定的
    parser.add_argument('--attribute-net', type=str, default="VGGAttributeNet", choices=["VGGAttributeNet", "BranchedTinyAttr"])      # 属性网络
    parser.add_argument('--attribute-set', type=str, default='/home/lsf/桌面/MaskFaceGAN/utils')      # 属性标签
    parser.add_argument('--test-set', type=str, default='/home/lsf/桌面/MaskFaceGAN/utils/VGGFace2HQ_attribute2.txt')      # 属性测试集的list
    parser.add_argument('--json-path', type=str, default='/home/lsf/桌面/MaskFaceGAN/json_per_json')      # json path
    parser.add_argument('--convert-data', type=bool, default=False)      # 是否需要转换数据，如果数据已经转换并且系统检索到了，就不需要花这个时间继续转了
    parser.add_argument('--test-number', type=int, default=-1)    # 整个测试集非常大，可以选择其中的部分，测试全部的话置-1

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
    """
    帮助我们知道属性名对应的数据集中索引号
    """
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
    datas = f.readlines()  # 直接将文件中按行读到list里，效果与方法2一样
    f.close()  # 关

    for i in tqdm(range(len(datas))):
        people_name = datas[i].split(" ")[0]
        
        if people_name in image_names:
            image_names.remove(people_name)
            with open(args.test_set, "a") as file:
                file.write(datas[i])

    # 补全
    for image_name in tqdm(image_names):
        for i in range(len(datas)):
            people_name = datas[i].split(" ")[0]
            if people_name.split("/")[0] == image_name.split("/")[0]:
                with open(args.test_set, "a") as file:
                    file.write(datas[i].replace(people_name, image_name))
                break

def convert(args):
    image_names = []

    # 读取json文件
    peoples = os.listdir(args.json_path)

    for people in peoples:
        people_path = os.path.join(args.json_path, people)
        json_files = os.listdir(
            people_path
        )

        for json_file in json_files:
            image_name_path = os.path.join(people, json_file.split("-")[0]+".jpg")
            
            image_names.append(image_name_path)
    
    # 读取数据集
    convert_(args, image_names, type_="all.txt")

def main(args):
    """
    计算某个属性的预测准确率。
    """
    device = args.device#"cpu"

    if args.attribute_net == "VGGAttributeNet":
        model = AttributeNet()  # 内部已经把参数冻结了，不需要手动eval()
        model.set_idx_list(attribute=[args.attribute])

    elif args.attribute_net == "BranchedTinyAttr":
        cfg = load_yaml("/home/lsf/桌面/MaskFaceGAN/config.yml")
        model = BranchedTinyAttr(cfg.MODELS.CLASSIFIER)
        model.set_idx_list(attributes=[VGG2CelebA[args.attribute]])
    
    model.to(device)

    # 开始读测试集
    f=open(args.test_set)
    datas = f.readlines()  # 直接将文件中按行读到list里，效果与方法2一样
    f.close()  # 关
    # random.shuffle(datas)

    attr_idx = get_idx(attribute_name = args.attribute)

    acc = 0
    number = 0

    TP = 0
    TP_num = 0

    peoples = os.listdir(args.json_path)
    Id_file = {}

    for person in peoples:
        im_p_json = os.listdir(os.path.join(args.json_path,person))
        
        new_dict = json.load(open(os.path.join(args.json_path,person,im_p_json[0]), 'r', encoding='utf-8'))
        Id_name = new_dict["ID"]
        Id_file[Id_name] = person
        print(Id_file)

    #print(abc)









    # 循环数据
    for data in tqdm(datas):
        data = data.strip() # 正则，去除空格
        image_path = os.path.join(args.image_dir, data.split(" ")[0])
        
        print(image_path)     
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
        if not os.path.exists(image_path):
            continue

        attr_label = int(data.split(" ")[attr_idx + 1])

        if args.attribute_net == "VGGAttributeNet":
            input_data = Path_Image_Preprocessing(image_path)
        elif args.attribute_net == "BranchedTinyAttr":
            input_data = read_img(image_path)
        
        if input_data is None:
            "部分在vgg test数据无需读取"
            continue
        input_data = input_data.to(device)

        predicted = model(input_data)

        if args.attribute_net == "BranchedTinyAttr":
            predicted = torch.sigmoid(predicted)

        predicted_truth = predicted[0][0].item()>0.5

        if args.attribute in Gender:
            gt_label = Gender.index(args.attribute)
            gt_judge = (gt_label == attr_label)

            if predicted_truth == gt_judge:
                # print(predicted[0][0].item(), predicted_truth, gt_label, attr_label, gt_judge, gt_label == attr_label)
                # 0.9971669316291809 True 0 0 True True
                # 0.0008440228411927819 False 0 1 False False
                acc += 1
                if gt_judge == True:
                    TP+=1
            number += 1

        elif args.attribute in Age:
            gt_label = Age.index(args.attribute)
            gt_judge = (gt_label == attr_label)
            if predicted_truth == gt_judge:
                acc += 1
                if gt_judge == True:
                    TP+=1
            number += 1

        elif args.attribute in Race:
            gt_label = Race.index(args.attribute)
            gt_judge = (gt_label == attr_label)
            if predicted_truth == gt_judge:
                acc += 1
                if gt_judge == True:
                    TP+=1
            number += 1

        elif args.attribute in Hair_color:
            gt_label = Hair_color.index(args.attribute)
            gt_judge = (gt_label == attr_label)
            if predicted_truth == gt_judge:
                acc += 1
                if gt_judge == True:
                    TP+=1
            number += 1

        else:
            if attr_label == 2:
                # 不确定的预测无意义
                continue
            if attr_label == 0:
                gt_judge = True
            elif attr_label == 1:
                gt_judge = False
            if predicted_truth == gt_judge:
                acc += 1
                if gt_judge == True:
                    TP+=1
            number += 1

        if gt_judge == True:
            # 记录判断为positive的
            TP_num += 1


        if number == args.test_number:
            break

    print("测试结束，有效测试数据数量为{}张，属性{}的预测结果的TP为{}，ACC为{}".format(number, args.attribute, TP/TP_num, acc/number))
        
if __name__ == '__main__':

    args = parse_args()  # n, gpu
    # 先转换数据
    if not os.path.exists(args.test_set) or args.convert_data:
        if os.path.exists(args.test_set):
            os.remove(args.test_set) 
        convert(args)
    
    # 测试准确率
    main(args)