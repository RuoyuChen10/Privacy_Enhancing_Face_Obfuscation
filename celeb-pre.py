import os 
#file_list = os.listdir('/home/lsf/桌面/MaskFaceGAN/json_per_json/')
json_path = '/home/lsf/桌面/MaskFaceGAN-github/results-com/vggface_output/'
peoples = os.listdir(json_path)
        
for people in peoples:
    people_path = os.path.join(json_path, people)
    json_files = os.listdir(people_path)
    os.chdir(people_path)
    for json_file in json_files:
        json_file_name = json_file
        
         
        json_file_name_new = json_file_name.split('-')[0]+'.jpg'
        os.rename(json_file_name,json_file_name_new)
        print(json_file_name)

                
#with open("./test.txt_","a") as file:
    #doc = os.path.join(people,people_image)+" "+str(people_num+8631)+"\n"
    #file.write(doc)