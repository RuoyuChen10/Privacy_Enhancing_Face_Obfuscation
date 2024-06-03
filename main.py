
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


# attributes=['black_hair','brown_hair','blond_hair','gray_hair','wavy_hair','straight_hair']

# attributes=['wearing_lipstick','mouth_slightly_open','smiling','bushy_eyebrows','arched_eyebrows','narrow_eyes','pointy_nose','big_nose','black_hair','brown_hair','blond_hair','gray_hair','wavy_hair','straight_hair']

#attributes=["Male", "Female", "Young", "Middle Aged", "Senior", "Asian", "White", "Black"]
attributes=["Male", "Female", "Young", "Middle Aged", "Senior"]

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
    parser.add_argument('--attribute', type=str, default='black_hair')
    parser.add_argument('--outdir', type=str, default='output')
    # parser.add_argument('--image', type=str, default='input/woman2.jpg')
    parser.add_argument('--image', type=str, default='input/man1.jpg')
    parser.add_argument('--target', type=int, default=1)
    parser.add_argument('--smoothing', type=float, default=0.05)
    parser.add_argument('--size', type=float, default=0)
    parser.add_argument('--use_e4e', action='store_true',default=True)
    parser.add_argument('--image_dir',type=str,default='/exdata2/lijia/MaskFaceGAN/input/vgg_align')
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
    #assert attributes == ["Male", "Female", "Young", "Middle Aged", "Senior", "Asian", "White", "Black"]
    assert attributes == ["Male", "Female", "Young", "Middle Aged", "Senior"]    
    Gender = torch.argmax(predicted_label[:,:2], dim = 1).cpu().numpy()
    Age = torch.argmax(predicted_label[:,2:5], dim = 1).cpu().numpy()
    Race = torch.argmax(predicted_label[:,2:5], dim = 1).cpu().numpy()

    target = VGG_Mapping[Gender, Age, Race]
    target = torch.tensor(target).float().unsqueeze(0)

    return target

# if __name__ == '__main__':
#     args = parse_args()  # n, gpu
#     cfg = load_config('config.yml', args)
#     names=os.listdir(args.image_dir)

#     for i in range(0,2):
#         for j in range(i+1,3):
#             for m in range(j+1,4):

#                 for attr0 in attributes[i]:
#                     for attr1 in attributes[j]:
#                         for attr2 in attributes[m]:
#                             models = ModelsModule(cfg.MODELS, attribute_subset=[attr0,attr1,attr2]).to(cfg.DEVICE)
#                             for name in names:
#                                 image_path=os.path.join(args.image_dir,name)
#                                 image, image_name, target = load_data(image_path, args.target, smoothing=args.smoothing)

#                                 starttime=time.time()
#                                 trainer = Trainer(image, models, target, cfg)
#                                 trainer.train_latent()
#                                 trainer.train_noise()
#                                 img_result = trainer.generate_result()
#                                 endtime=time.time()

#                                 print('time:',endtime-starttime,'s')


#                                 save_results(img_result, image_name.split('.')[0]+'_'+attr0+'_'+attr1+'_'+attr2+'.jpg', outdir='./output/ryc-test')

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    args = parse_args()  # n, gpu
    cfg = load_config('config.yml', args)
    names=os.listdir(args.image_dir)

    models = NewModelsModule(cfg = cfg.MODELS, attribute_subset=attributes).to(cfg.DEVICE)
    for name in names:
        image_path=os.path.join(args.image_dir,name)

        image, image_name, target = load_data(image_path, args.target, smoothing=args.smoothing)
        
        ## add the json score
        #
        #
        #
        #
        #
        #
        #
        
        # 预测
        human_information = models.face_net(image.to(cfg.DEVICE))  # 这里输入不用转换，不能用images_attr_input等经过self.convert_attr_input的作为输入
        
        attr_input = Path_Image_Preprocessing(image_path)
        predicted = models.classifier(attr_input.to(cfg.DEVICE))
        print(predicted)
        # 判别反属性
        # 标签
        target = global_attribute_target(predicted, attributes)
        
        
        starttime=time.time()
        trainer = Trainer(image, models, target, cfg)
        trainer.train_latent()
        trainer.train_noise()
        img_result = trainer.generate_result()
        endtime=time.time()

        print('time:',endtime-starttime,'s')


        save_results(img_result, image_name.split('.')[0]+'.jpg', outdir='./output/ryc-test')
