
import os
from pathlib import Path
import torch
import argparse
import cv2
import numpy as np

from utils import save_image, read_img, load_config
from trainer import Trainer, NewTrainer
from model_module import ModelsModule, NewModelsModule
import time

import json

# attributes=['black_hair','brown_hair','blond_hair','gray_hair','wavy_hair','straight_hair']

# attributes=['wearing_lipstick','mouth_slightly_open','smiling','bushy_eyebrows','arched_eyebrows','narrow_eyes','pointy_nose','big_nose','black_hair','brown_hair','blond_hair','gray_hair','wavy_hair','straight_hair']

attributes=["Male", "Female", "Young", "Middle Aged", "Senior", "Asian", "White", "Black"]
#attributes=["Male", "Female", "Young", "Middle Aged", "Senior"]

def Path_Image_Preprocessing(path):
    '''
    Precessing the input images
        image_dir: single image input path, such as "/home/xxx/10.jpg"
    '''
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])
    image = cv2.imread(path)
    assert image is not None
    image = cv2.resize(image,(224,224))
    image = image.astype(np.float32)
    image -= mean_bgr
    # H * W * C   -->   C * H * W
    image = image.transpose(2,0,1)
    image = torch.tensor(image)
    image = image.unsqueeze(0)
    return image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=bool, default=True)
    parser.add_argument('--attribute', type=str, default='Young')
    parser.add_argument('--outdir', type=str, default='output')
    # parser.add_argument('--image', type=str, default='input/woman2.jpg')
    parser.add_argument('--image', type=str, default='input/man1.jpg')
    parser.add_argument('--target', type=int, default=1)
    parser.add_argument('--smoothing', type=float, default=0.05)
    parser.add_argument('--size', type=float, default=0)
    parser.add_argument('--use_e4e', action='store_true',default=True)
    parser.add_argument('--image_dir',type=str,default='/exdata2/lijia/MaskFaceGAN/input/vgg_align')
    parser.add_argument('--json-path',type=str,default='./json_per_ima')
    parser.add_argument('--image-root-path',type=str,default='/exdata2/RuoyuChen/Datasets/FaceV/VGGFace2/Img')
    parser.add_argument('--device',type=str,default='2')
    parser.add_argument('--threshold', type=float, default=0.9)
    args = parser.parse_args()
    return args

def load_data(image_file,target, smoothing=0.05):
    image = read_img(image_file)
    image_name = Path(image_file).name
    target = torch.tensor(target).float().unsqueeze(0)
    target = torch.abs(target - smoothing)
    return image, image_name, target

def save_results(img_result, image_name, outdir='.'):
    os.makedirs(outdir, exist_ok=True)
    save_image(img_result, os.path.join(outdir, image_name))

# ["Male", "Female", "Young", "Middle Aged", "Senior", "Asian", "White", "Black"]
#VGG_Mapping = np.array([
#    [           # Male
#        [       # Young
#            [0.02, 0.98, 0.01, 0.01, 0.98, 0.01, 0.98, 0.01], 
#            [0.02, 0.98, 0.01, 0.01, 0.98, 0.98, 0.01, 0.01],
#            [0.02, 0.98, 0.01, 0.01, 0.98, 0.01, 0.98, 0.01]
#        ],
#        [       # Middle
#            [0.02, 0.98, 0.98, 0.01, 0.01, 0.01, 0.98, 0.01], 
#            [0.02, 0.98, 0.98, 0.01, 0.01, 0.98, 0.01, 0.01],
#            [0.02, 0.98, 0.98, 0.01, 0.01, 0.01, 0.98, 0.01]
#        ],
#        [       # Senior
#            [0.02, 0.98, 0.01, 0.98, 0.01, 0.01, 0.98, 0.01],
#            [0.02, 0.98, 0.01, 0.98, 0.01, 0.98, 0.01, 0.01],
#            [0.02, 0.98, 0.01, 0.98, 0.01, 0.01, 0.98, 0.01]
#        ]
#    ],     
#    [       # Female
#        [   # Young
#            [0.98, 0.02, 0.01, 0.01, 0.98, 0.01, 0.98, 0.01],
#            [0.98, 0.02, 0.01, 0.01, 0.98, 0.98, 0.01, 0.01],
#            [0.98, 0.02, 0.01, 0.01, 0.98, 0.01, 0.98, 0.01]
#        ],     
#        [   # Middle
#            [0.98, 0.02, 0.01, 0.01, 0.98, 0.01, 0.98, 0.01],
#            [0.98, 0.02, 0.98, 0.01, 0.01, 0.98, 0.01, 0.01],
#            [0.98, 0.02, 0.01, 0.01, 0.98, 0.01, 0.98, 0.01]
#        ],
#        [   # Senior
#            [0.98, 0.02, 0.98, 0.01, 0.01, 0.01, 0.98, 0.01],
#            [0.98, 0.02, 0.98, 0.01, 0.01, 0.98, 0.01, 0.01],
#            [0.98, 0.02, 0.98, 0.01, 0.01, 0.01, 0.98, 0.01]
#        ]]      
#    ])

VGG_Mapping = np.array([
    [           # Male
        [       # Young
            [0.02, 0.98, 0.01, 0.01, 0.98], 
            [0.02, 0.98, 0.01, 0.01, 0.98],
            [0.02, 0.98, 0.01, 0.01, 0.98]
        ],
        [       # Middle
            [0.02, 0.98, 0.98, 0.01, 0.01], 
            [0.02, 0.98, 0.98, 0.01, 0.01],
            [0.02, 0.98, 0.98, 0.01, 0.01]
        ],
        [       # Senior
            [0.02, 0.98, 0.01, 0.98, 0.01],
            [0.02, 0.98, 0.01, 0.98, 0.01],
            [0.02, 0.98, 0.01, 0.98, 0.01]
        ]
    ],     
    [       # Female
        [   # Young
            [0.98, 0.02, 0.01, 0.01, 0.98],
            [0.98, 0.02, 0.01, 0.01, 0.98],
            [0.98, 0.02, 0.01, 0.01, 0.98]
        ],     
        [   # Middle
            [0.98, 0.02, 0.01, 0.01, 0.98],
            [0.98, 0.02, 0.98, 0.01, 0.01],
            [0.98, 0.02, 0.01, 0.01, 0.98]
        ],
        [   # Senior
            [0.98, 0.02, 0.98, 0.01, 0.01],
            [0.98, 0.02, 0.98, 0.01, 0.01],
            [0.98, 0.02, 0.98, 0.01, 0.01]
        ]]      
    ])



def global_attribute_target(predicted_label, attributes):
    """
    predicted: (batch, 8)
    """
    assert attributes == ["Male", "Female", "Young", "Middle Aged", "Senior", "Asian", "White", "Black"]
    #assert attributes == ["Male", "Female", "Young", "Middle Aged", "Senior"]    
    Gender = torch.argmax(predicted_label[:,:2], dim = 1).cpu().numpy()
    Age = torch.argmax(predicted_label[:,2:5], dim = 1).cpu().numpy()
    Race = torch.argmax(predicted_label[:,5:8], dim = 1).cpu().numpy()

    target = VGG_Mapping[Gender, Age, Race]
    target = torch.tensor(target).float().unsqueeze(0)

    return target

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    args = parse_args()  # n, gpu
    cfg = load_config('config.yml', args)
    names=os.listdir(args.image_dir)

    #os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    models = NewModelsModule(cfg = cfg.MODELS, attribute_subset=attributes).to(cfg.DEVICE)
    #models = ModelsModule(cfg = cfg.MODELS, attribute_subset=attributes).to(cfg.DEVICE)

    #print(next(models.face_parser.parameters()).device)

    peoples = os.listdir(args.json_path)

    for people in peoples:
        people_path = os.path.join(args.json_path, people)
        json_files = os.listdir(
            people_path
        )

        for json_file in json_files:
            json_file_path = os.path.join(people_path, json_file)
            new_dict = json.load(open(json_file_path, 'r', encoding='utf-8'))
            

            image_path=os.path.join(args.image_root_path, new_dict["ima"])
            print(image_path)

            image, image_name, target = load_data(image_path, args.target, smoothing=args.smoothing)

           
            mse_part = []                     
            mse_part_score = []            
            mse_part2idx = []       
            total_score = 0

                       
            for part_i in range(7):
                total_score += new_dict["Score_softmax"][part_i][1]
                mse_part.append(new_dict["Score_softmax"][part_i][0])
                mse_part_score.append(new_dict["Score_softmax"][part_i][1])

                mse_part2idx.append(
                    models.face_parser.mask2idx[new_dict["Score_softmax"][part_i][0]]
                )

                # 闃堝€煎垽鏂?                
                if total_score > args.threshold:
                    break
            
            models.face_parser.mask_idxs['mse'] = mse_part2idx
            #print(mse_part2idx)
            # print(models.face_parser.mask_idxs)
            
                     
            attr_input = Path_Image_Preprocessing(image_path)
            predicted = models.classifier(attr_input.to(cfg.DEVICE))
            
                    
            #target = global_attribute_target(predicted, attributes)
            
            
            starttime=time.time()
            trainer = Trainer(image, models, target, cfg)
            trainer.train_latent()
            trainer.train_noise()
            img_result = trainer.generate_result()
            endtime=time.time()

            print('time:',endtime-starttime,'s')


            save_results(img_result, image_name.split('.')[0]+'-'+str(args.threshold)+'.jpg', outdir='./output/abresults-RISE')
            save_results(image, image_name.split('.')[0]+'-'+str(args.threshold)+'.jpg', outdir='./output/abresults_orig-RISE')
