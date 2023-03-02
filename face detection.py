#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries(MTCNN=multi_task cascaded convutional neural network)
import facenet_pytorch
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import os


# In[2]:


# initiating MTCNN and InceptionResnetV1
mtcnn0 = MTCNN(image_size=240, margin=0, keep_all= False, min_face_size=40)
mtcnn = MTCNN(image_size=240, margin=0, keep_all= True, min_face_size=40)
resnet =  InceptionResnetV1(pretrained = 'vggface2').eval()


# In[3]:


# read data from folder
dataset =datasets.ImageFolder(r"D:\lol\pics")
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()}

def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

face_list = []
name_list = []  #list of names correspondence to cropped photos
embedding_list = []    #list of embedded matrix after conversion from cropped faces to embedding matrix using resnet

for img, idx in loader:
    face, prob = mtcnn0(img, return_prob=True)
    if face is not None and prob>0.92:
        emb = resnet(face.unsqueeze(0))
        embedding_list.append(emb.detach())
        name_list.append(idx_to_class[idx])
        
# save data
data = [embedding_list, name_list]
torch.save(data, 'data.pt') # saving data.pt file


# In[ ]:


#using webcam to detect face
load_data=torch.load('data.pt')
embedding_list = load_data[0]
name_list = load_data[1]

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame, try again")
        break
        
    img = Image.fromarray(frame)
    img_cropped_list, prob_list = mtcnn(img, return_prob = True)
    
    if img_cropped_list is not None:
        boxes, _=mtcnn.detect(img)
        
        for i in prob in enumerate(prob_list):
            if prob>0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()
                
                dist_list = [] # list of amtched distances minimum distances is used to identify the person
                
                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)
                    
                min_dist = min(dist_list)#get min dist value
                min_dist_idx = dist_list.index(min_dist) # get min dist index
                name = name_list[min_dist_idx] # get name corresponding to minimum dist
                
                box = boxes[i]
                original_frame = frame.copy() #storing copy of frames before drawing on it
                
                if min_dist<0.90:
                    frame=cv2.putText(frame, name+' '+str(min_dist), (box[0],box[1])< cv2.FRONT_HERSHEY_SIMPLEX, 1)
                    
                frame = cv2.rectangle(frame, (box[0].box[1]),(box[2],box[3]),(255,0,0),2)
                
        cv2.imshow("IMG",frame)
        
        k = cv2.waitkey(1)
        if k%256==27: 
            print('closing...')
        
            break
            
            
        elif k%256==32:
            print("enter your name: ")
            name = input()
            
            #create directory if not exists 
            if not os.path.exists('D:\lol\pics/'+name):
                os.mkdir('D:\lol\pics/'+name)
                
            img_name = "D:\lol\pics/{}/{}.jpg".formate(name, int(time.time()))
            cv2.imwrite(img_name, original_frame)
            print(" saved: {}".formt(img_name))
            
cam.released()
cv2.destroyAllWindows()


# In[ ]:




