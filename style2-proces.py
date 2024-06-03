
from PIL import Image

import os 
#file_list = os.listdir('/home/lsf/桌面/MaskFaceGAN/json_per_json/')
img_path = '/home/lsf/桌面/MaskFaceGAN/expression-results/vggface2/'
img_name_list = os.listdir(img_path)
        
for im_path in img_name_list:
    img = Image.open(os.path.join(img_path, im_path))
    size_img = img.size
    x=0
    y=0
    x_num=10
    y_num= 1
    w = int(size_img[0]/x_num)
    h = int(size_img[1]/y_num)


    category_raw = im_path.split('.')[0]
    category = category_raw.split('_')[-1]
    if category=="gender":
        img_save_name = im_path.split('_')[1] + "_" + im_path.split('_')[2]+ '.jpg'
        print(img_save_name)
        save_name_m = './styleg2-results/gender/male/'+img_save_name
        save_name_F = './styleg2-results/gender/female/'+img_save_name
        sub_img_L = img.crop((x,y,x+w,y+h))
        sub_img_R = img.crop((x+9*w,y,x+10*w,y+h))
        sub_img_L.save(save_name_F)
        sub_img_R.save(save_name_m)
    if category=="age":
        img_save_name = im_path.split('_')[1] + "_" + im_path.split('_')[2]+ '.jpg'
        print(img_save_name)
        save_name_y = './styleg2-results/age/young/'+img_save_name
        save_name_s = './styleg2-results/age/senior/'+img_save_name
        sub_img_L = img.crop((x,y,x+w,y+h))
        sub_img_R = img.crop((x+9*w,y,x+10*w,y+h))
        sub_img_L.save(save_name_s)
        sub_img_R.save(save_name_y)
    if category=="white":
        img_save_name = im_path.split('_')[1] + "_" + im_path.split('_')[2]+ '.jpg'
        print(img_save_name)
        save_name_w = './styleg2-results/race/white/'+img_save_name
        #save_name_F = './gender/female/'+img_save_name
        #sub_img_L = img.crop((x,y,x+w,y+h))
        sub_img_R = img.crop((x+9*w,y,x+10*w,y+h))
        #sub_img_L.save(save_name_F)
        sub_img_R.save(save_name_w)
    if category=="black":
        img_save_name = im_path.split('_')[1] + "_" + im_path.split('_')[2]+ '.jpg'
        print(img_save_name)
        save_name_b = './styleg2-results/race/black/'+img_save_name
        sub_img_R = img.crop((x+9*w,y,x+10*w,y+h))
        sub_img_R.save(save_name_b)
    if category=="yellow":
        img_save_name = im_path.split('_')[1] + "_" + im_path.split('_')[2]+ '.jpg'
        print(img_save_name)
        save_name_y = './styleg2-results/race/yellow/'+img_save_name
        sub_img_R = img.crop((x+9*w,y,x+10*w,y+h))
        sub_img_R.save(save_name_y)
