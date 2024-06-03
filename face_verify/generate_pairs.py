import os
import random
import numpy as np
import json

#path = "/home/lsf/face_anymoize/deepprivacy/DeepPrivacy-master/results-vgg2/"
#path = "/home/lsf/face_anymoize/facemasoic/Face_mosaic-master/vggface2/"
#path = "/home/lsf/桌面/MaskFaceGAN/styleg2-results/race/white/"
#path = "/home/lsf/桌面/MaskFaceGAN-github/results-com/vggface_output/"
path = "/home/lsf/桌面/MaskFaceGAN-github/e4e_young/"

path_rep = "/home/lsf/face_anymoize/datasets/Img_VGG/"calculate_activation_statistics
positive_pairs = 1
negative_pairs = 10

images = os.listdir(path)

print(len(images))
datasets = []
# choose pairs different
json_path = '/home/lsf/桌面/MaskFaceGAN/json_per_json/'
peoples = os.listdir(json_path)
Id_file = {}

for person in peoples:
    im_p_json = os.listdir(os.path.join(json_path,person))
        
    new_dict = json.load(open(os.path.join(json_path,person,im_p_json[0]), 'r', encoding='utf-8'))
    Id_name = new_dict["ID"]
    Id_file[Id_name] = person
p_num = 0 
for image_name in images:
    data1 = image_name.split('_')[0]
    #print(image_name)
    #print(abc)
    #im_name_info = os.listdir(os.path.join(path,data1.split('_')[0]))
    #print(im_name_info)
    #if not im_name_info:
        #continue


    file_name = image_name
    #print(file_name)
    #print(abc)
    #print(file_name)
    #data1 = image_name.split("-")[0] + "-0.6.jpg" 
    #im_name = image_name.split('-')[0]+'.jpg'
    #str_flg = image_name.split("-")[1]
    #im_name = image_name.split('-')[0]+'-'+str_flg
    #if "detected" in str_flg:
       #continue
    #if int(str_flg.split(".")[0]) in Id_file.keys():
       #file_name = Id_file[int(str_flg.split(".")[0])]
    #else:
       #continue
    
    im_p = os.listdir(os.path.join(path_rep,data1))
    im_p_name = random.choice(im_p) 
    im_pc_name =  random.choice(im_p)     
    q_img = os.path.join(path,file_name)
    #q_img = os.path.join(path,data1,file_name)
    c_img = os.path.join(path_rep,data1,im_p_name)
    pc_img = os.path.join(path_rep,data1,im_pc_name)
    #print(q_img)
    #print(c_img)
    #print(pc_img)
    #print(abc)
    datasets.append([q_img,c_img,1])
    datasets.append([q_img,pc_img,1])

    i = 0


    while(True):
       #data1 = random.choice(images)
       sample_info = random.choice(images)
       data2 = sample_info.split('_')[0]
       im_file2 = os.listdir(os.path.join(path_rep,data2))
       im_c = random.choice(im_file2)

       #str_flag2 = data2.split("-")[1]
       #if "0.6" in str_flag2: 
       #if "detected" in str_flag2:
           #continue                   
       if data1 != data2:
          n_img = os.path.join(path,data1)
          nc_img = os.path.join(path_rep,data2,im_c)
          #print(nc_img)
          #print(abc)
          datasets.append([q_img,nc_img,0])
          i+=1
       if i == negative_pairs:
           break
    p_num+=1
    if p_num==1000:
       break
#i=0
#while(True):
#    data1 = random.choice(images)
#    data2 = random.choice(images)
#    if data1 != data2:
#        if data1.split('_')[0] == data2.split('_')[0]:
#            datasets.append([data1,data2,1])
#            i+=1
#    if i == positive_pairs:
#        break

np.savetxt('./text/tutorial_interfacegan_id.txt',
                   np.array(datasets),
                   fmt='%s %s %s',
                   delimiter='\t')