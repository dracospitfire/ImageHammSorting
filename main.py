#=====================================================#
#               ImageHammingSorting.py                #
# =       Created by Austin Flores on 8/14/22       = #
#       All rights reservied by dracospitfire         #
#=====================================================#
#!/usr/bin/env python3
import hashlib, os, shutil
from hashlib import md5
from pathlib import Path
import itertools, scipy.spatial
import matplotlib.pyplot as plt
import numpy as np
import cv2, glob

jpgFolderDirectory = Path('images')

### Main function for iteration over images in folder ###
def main():
    os.chdir('images')
    files_list = glob.glob('*.jpg')
    #print(len(files_list))

    duplicates = []
    hash_keys = dict()
    for index, filename in  enumerate(files_list):
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                filehash = hashlib.md5(f.read()).hexdigest()
            if filehash not in hash_keys:
                hash_keys[filehash] = index
            else:
                duplicates.append((index,hash_keys[filehash]))
    
    ###Removes all duplicate images###
    for index in duplicates:
        os.remove(files_list[index[0]])
    
    image_files = filter_images()
    HAMM_duplicates, ds_dict = difference_score_dict_HAMM(image_files)
    
    i = 0
    for k1,k2 in itertools.combinations(ds_dict, 2):
        if hamming_distance(ds_dict[k1], ds_dict[k2]) < .10:
            i += 1
            HAMM_duplicates.append((k1,k2))
    print("HAMMING Duplicates: ", len(HAMM_duplicates))
    print(HAMM_duplicates)
    
    '''
    i = 0
    for images in HAMM_duplicates:
        i += 1
        similar = "HAMM similar " + str(i)
        FolderDirectory = Path('')
        NewFolderDirectory = FolderDirectory/similar
        if not NewFolderDirectory.exists():
            NewFolderDirectory.mkdir()
            #os.makedirs(similar)
        for image in images:
            OldFolderDirectory = FolderDirectory/image
            imageFolderDirectory = NewFolderDirectory/image
            shutil.copyfile(OldFolderDirectory, imageFolderDirectory)
    '''

###
def filter_images():
    image_list = []
    for image in glob.glob('*.jpg'):
        try:
            assert cv2.imread(image).shape[2] == 3
            image_list.append(image)
        except  AssertionError as e:
            print(e)
    return image_list

#Hamming
def difference_score_dict_HAMM(image_list):
    ds_dict = {}
    HAMM_duplicates = []
    for image in image_list:
        ds = difference_score(image)
        if image not in ds_dict:
            ds_dict[image] = ds
        else:
            HAMM_duplicates.append((image, ds_dict[image]) )
    return  HAMM_duplicates, ds_dict


def difference_score(image, height = 30, width = 30):
    gray = img_gray(image)
    row_res, col_res = resize(gray, height, width)
    difference = intensity_diff(row_res, col_res)
    return difference

###First turn the image into a gray scale image
def img_gray(image):
    image = cv2.imread(image)
    return np.average(image, weights=[0.299, 0.587, 0.114], axis=2)

###resize image and flatten
def resize(image, height=30, width=30):
    row_res = cv2.resize(image,(height, width), interpolation = cv2.INTER_AREA).flatten()
    col_res = cv2.resize(image,(height, width), interpolation = cv2.INTER_AREA).flatten('F')
    return row_res, col_res

###gradient direction based on intensity
def intensity_diff(row_res, col_res):
    difference_row = np.diff(row_res)
    difference_col = np.diff(col_res)
    difference_row = difference_row > 0
    difference_col = difference_col > 0
    return np.vstack((difference_row, difference_col)).flatten()

#Now Let's try Hamming distance
def hamming_distance(image, image2):
    score = scipy.spatial.distance.hamming(image, image2)
    return score

### Statement allows you to run file either as reusable module or standalone programs ###
if __name__ == "__main__":
    main()