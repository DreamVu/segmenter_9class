#%%accuracy testing of enet 10 class exp15




import os
import cv2
import numpy as np

def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
        return 1
    return (2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin))

true_class_path = "/home/dreamvu/workspace/aman/segmenter_9class/all_test_data/Labels_test/"


pred_class_path = "/home/dreamvu/workspace/aman/segmenter_9class/results/labels_gray/"
#folders = os.listdir(pred_class_path)
#folders = ["Seg-640(544)-9-FP16-Concat_9_class","Seg-640(544)-9-FP16-Padding_9_class"]
folders = [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 63, 66, 68, 70, 71, 72, 73, 74, 75]

all_model_list = []

for folder in folders:
    folder = str(folder)
    pred_path = pred_class_path+folder+"/"
    true_path = true_class_path+folder+"/"
    fnames = os.listdir(true_path)
    fnames.sort(reverse=True)
    param_list_image = [] 
    param_list_image_array_folder =[]
    for fname in fnames:
        print(fname)
        pred_img = cv2.imread(pred_path + fname[:-4] + ".jpg", 0)
        #print(np.unique(pred_img))
        true_img = cv2.imread(true_path + fname, 0)
        #print(np.unique(true_img))
        # calculating for 3  class #floor, #wall and #ceiling
        pred_class = [0,1,2,3,4,5,6,7,8]
        true_class =  [1,2,3,4,5,6,7,8,9]
        
        param_list = []
        for cl in range(len(pred_class)):
            #print(pred_class[cl],true_class[cl])
            if pred_class[cl] in np.unique(pred_img) and true_class[cl] in np.unique(true_img):
                #print("True")
                pred_img_bin =  np.where(pred_img==pred_class[cl], 1, 0).astype(np.uint8)
                h,w = pred_img_bin.shape
                true_img_bin =  np.where(true_img==true_class[cl], 1, 0).astype(np.uint8)
                true_img_bin = cv2.resize(true_img_bin,(w,h),interpolation = cv2.INTER_NEAREST)
                #plt.imshow(pred_img_bin)
                #plt.imshow(true_img_bin)
                
                dc = single_dice_coef(pred_img_bin, true_img_bin)
                #for false positives and false negative
                fp_fn_bin = np.where(true_img_bin!= pred_img_bin,1, 0).astype(np.uint8)
                fn_bin = fp_fn_bin.copy()*(1-pred_img_bin.copy())
                fp_bin = fp_fn_bin.copy()*pred_img_bin.copy()
                
                #for true negatives and true positives
                tn_tp_bin = np.where(true_img_bin==pred_img_bin,1, 0).astype(np.uint8)
                tp_bin = tn_tp_bin.copy()*true_img_bin.copy()
                tn_bin = tn_tp_bin.copy()*(1-true_img_bin.copy())
                
                #fnr and tpr
                fnr = np.sum(fn_bin)/(np.sum(fn_bin)+np.sum(tp_bin))
                tpr = np.sum(tp_bin)/(np.sum(fp_bin)+np.sum(tp_bin))
                
                param_list.append([dc, np.sum(fp_bin),np.sum(fn_bin),np.sum(tp_bin),np.sum(tn_bin), fnr,tpr])
                #param_list.append([dc, fnr,tpr])
            else:
                #print("False")
                param_list.append([0,0,0,0,0,0,0])
                #param_list.append([0, 0,0])
        param_list_array = np.array(param_list).flatten()
        param_list_image.append(param_list_array)
    param_list_image_array = np.array(param_list_image)
    param_list_image_array_folder = [np.mean(param_list_image_array, axis=0), np.std(param_list_image_array, axis=0), np.var(param_list_image_array, axis=0)]
    all_model_list.append(param_list_image_array_folder)
#print(all_model_list)
all_model_list_array = np.array(all_model_list)
a_copy = all_model_list_array[0]
for i in range(1,len(all_model_list)):
    a = all_model_list_array[i]
    a_copy=np.vstack((a_copy,a))

#copy the a_copy variable to sheet

np.savetxt("accuracy_test.txt", a_copy)
