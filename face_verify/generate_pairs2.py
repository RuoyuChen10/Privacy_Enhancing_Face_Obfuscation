import os
import random
import numpy as np

path = "/home/lsf/face_anymoize/datasets/CelebAMask-HQ/CelebA-HQ-img/"
path2 = "/home/lsf/face_anymoize/facemasoic/Face_mosaic-master/celeb-a/"
positive_pairs = 2000
negative_pairs = 2000

id_i = os.listdir(path)
ids = os.listdir(path2)

datasets = []
# choose pairs different

i = 0
while(True):
    id_im = random.choice(id_i)
    
    #images = os.listdir(os.path.join(path, id_im))
    data1 = id_im
    
    #id = data1.split("_")[0]

    #id_c = random.choice(ids)
    #if id == id_c:
        #continue

    #data2s = os.listdir(os.path.join(path2, id_c))

    data2 = random.choice(id_i)
    datasets.append([
        os.path.join(path, data1),
        os.path.join(path, data2),0])
    i+=1
    if i == negative_pairs:
        break
'''
i =0
while(True):
    id_im = random.choice(id_i)
    
    images = os.listdir(os.path.join(path, id_im))
    data1 = random.choice(images)
    data2 = random.choice(images)
    if data1.split('_')[0] != data2.split('_')[0]:
        datasets.append([
            os.path.join(path,  id_im+'/'+data1),
            os.path.join(path,  id_im+'/'+data2),
            0])
        i+=1
    else:
        continue
    if i == negative_pairs:
        break
'''
i=0
while(True):
    id_im = random.choice(id_i)
    
    #images = os.listdir(os.path.join(path, id_im))
    #data1 = random.choice(images)

    #data2s = os.listdir(os.path.join(path2, id_im))

    #data2 = random.choice(data2s)
    datasets.append([
        os.path.join(path, id_im),
        os.path.join(path2, id_im),
        1])
    i+=1
    if i == positive_pairs:
        break

np.savetxt('./text/tutorial-CELEBA-M.txt',
                   np.array(datasets),
                   fmt='%s %s %s',
                   delimiter='\t')