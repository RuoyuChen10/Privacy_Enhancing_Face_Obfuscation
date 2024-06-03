
import os
from pathlib import Path
import torch
import argparse

from utils import save_image, read_img, load_config
from trainer import Trainer
from model_module import ModelsModule, NewModelsModule
import time


# attributes=['black_hair','brown_hair','blond_hair','gray_hair','wavy_hair','straight_hair']

# attributes=['wearing_lipstick','mouth_slightly_open','smiling','bushy_eyebrows','arched_eyebrows','narrow_eyes','pointy_nose','big_nose','black_hair','brown_hair','blond_hair','gray_hair','wavy_hair','straight_hair']

attributes=[['wearing_lipstick','mouth_slightly_open','smiling','Male', 'Female', 'Young', 'Middle Aged', 'Senior','Asian', 'White', 'Black'],['bushy_eyebrows','arched_eyebrows','narrow_eyes'],['pointy_nose','big_nose'],['black_hair','brown_hair','blond_hair','gray_hair','wavy_hair','straight_hair']]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=bool, default=None)
    parser.add_argument('--attribute', type=str, default='black_hair')
    parser.add_argument('--outdir', type=str, default='output')
    parser.add_argument('--image', type=str, default='input/vgg_align/n000659.jpg')
    parser.add_argument('--target', type=int, default=1)
    parser.add_argument('--smoothing', type=float, default=0.05)
    parser.add_argument('--size', type=float, default=0)
    parser.add_argument('--use_e4e', action='store_true',default=False)
    # parser.add_argument('--image_dir',type=str,default='/exdata2/lijia/MaskFaceGAN/input/vgg_align')
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



if __name__ == '__main__':
    args = parse_args()  # n, gpu
    cfg = load_config('config.yml', args)

    models = NewModelsModule(cfg.MODELS, attribute_subset=['Male', 'Female', 'Young', 'Middle Aged', 'Senior','Asian', 'White', 'Black','Black Hair']).to(cfg.DEVICE)
    image, image_name, target = load_data(args.image, args.target, smoothing=args.smoothing)

    starttime=time.time()
    trainer = Trainer(image, models, target, cfg)
    trainer.train_latent()
    trainer.train_noise()
    img_result = trainer.generate_result()
    endtime=time.time()

    print('用时',endtime-starttime,'s')


    save_results(img_result,  image_name,outdir='/exdata2/RuoyuChen/Demo/MaskFaceGAN/example')
