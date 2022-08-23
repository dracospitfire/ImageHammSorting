#=====================================================#
#                 ImageHammSorting.py                 #
# =       Created by Austin Flores on 8/14/22       = #
#       All rights reservied by dracospitfire         #
#=====================================================#
#!/usr/bin/env python3
import hashlib, os, shutil, time
from hashlib import md5
from pathlib import Path
import itertools, scipy.spatial
import matplotlib.pyplot as plt
import numpy as np
import cv2, glob

### Global Path for file with images ###
jpgFolderDirectory = Path('FOLDER_DIRECTORY_WITH_ALL_YOUR_IMAGES')

### Main function for scoring and sorting images in directory ###
def main():
    ### Sets working directory ###
    os.chdir(jpgFolderDirectory)

    ### Write only images into list ###
    start = time.time()
    image_list = glob.glob('*.jpg')
    print("TOTAL IMAGES:", len(image_list))
    end = time.time()
    print("Building Image File Time Lap:",(end - start))
    
    ### This is optional if you don't expect to have duplicastes ###
    #start = time.time()
    #duplicates = find_duplicates(image_list)
    duplicates = []
    #end = time.time()
    #print("Finding Duplicates Time Lap:",(end - start))
    
    start = time.time()
    remove_duplicates(duplicates, image_list)
    end = time.time()
    print("Removing Duplicates Time Lap:",(end - start))
   
    start = time.time()
    HAMM_duplicates, ds_dict = difference_score_dict_HAMM(image_list)
    end = time.time()
    print("\nTotal Scoring Lap:",(end - start))
   
    start = time.time()
    HAMM_duplicates = build_image_library(ds_dict, HAMM_duplicates)
    end = time.time()
    print("Dictionary Build Time Lap:",(end - start))
    
    start = time.time()
    sort_image_library(HAMM_duplicates)
    end = time.time()
    print("Sorting Time Laps:",(end - start))
    
### Build list of duplicate images###
def find_duplicates(image_list):
    print("\nLooking for Duplicates")
    duplicates = []
    hash_keys = dict()
    for index, filename in  enumerate(image_list):
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                filehash = hashlib.md5(f.read()).hexdigest()
            if filehash not in hash_keys:
                hash_keys[filehash] = index
            else:
                duplicates.append((index,hash_keys[filehash]))
    print("DUPLICATE Images:", len(duplicates))
    return duplicates

### Removes any duplicate images ###
def remove_duplicates(duplicates, image_list):
    print("\nRemoving Duplicates Found")
    if len(duplicates) > 0:
        for index in duplicates:
            os.remove(image_list[index[0]])

### Hamming process starts here ###
def difference_score_dict_HAMM(image_list):
    print("\nBuilding Hamming Score Library")
    ds_dict = {}
    HAMM_duplicates = {}
    i = 0
    for image in image_list:
        i += 1
        print("\nIMAGE:", i)
        start = time.time()
        ds = difference_score(image)
        if image not in ds_dict:
            ds_dict[image] = ds
        else:
            HAMM_duplicates.append((image, ds_dict[image]))
        end = time.time()
        print("\n", "Image Processing Time:",(end - start))
    return  HAMM_duplicates, ds_dict

### Build difference score for each image ###
def difference_score(image, height = 30, width = 30):
    start = time.time()
    gray = img_gray(image)
    end = time.time()
    print("Grayscale Time:",(end - start))
    start = time.time()
    row_res, col_res = resize(gray, height, width)
    end = time.time()
    print("Resizing Time:",(end - start))
    start = time.time()
    difference = intensity_diff(row_res, col_res)
    end = time.time()
    print("Intesity Scale Time:",(end - start))
    return difference

### Open image as gray scale image ###
def img_gray(image):
    print("Convert to Gray Scale")
    image = cv2.imread(image, 0)
    return image

### Resize image and flatten into an array ###
def resize(image, height=102, width=153):
    print("Resizing Image")
    row_res = cv2.resize(image,(height, width), interpolation = cv2.INTER_AREA).flatten()
    col_res = cv2.resize(image,(height, width), interpolation = cv2.INTER_AREA).flatten('F')
    return row_res, col_res

### Set gradient direction based on intensity ###
def intensity_diff(row_res, col_res):
    print("Setting Intenisty")
    difference_row = np.diff(row_res)
    difference_col = np.diff(col_res)
    difference_row = difference_row > 0
    difference_col = difference_col > 0
    return np.vstack((difference_row, difference_col)).flatten()

### Determine Hamming distance score ###
def hamming_distance(image, image2):
    score = scipy.spatial.distance.hamming(image, image2)
    return score

### Build dictionary of similar images ###
def build_image_library(ds_dict, HAMM_duplicates):
    print("\nComparing Images form Library")
    for i1, i2 in itertools.combinations(ds_dict, 2):
        if hamming_distance(ds_dict[i1], ds_dict[i2]) < .15:
            if i1 in HAMM_duplicates.keys() and i2 not in HAMM_duplicates.keys():
                k = 0
                for image_list in HAMM_duplicates.values():
                    if i2 in image_list:
                        k += 1
                if k == 0:
                    HAMM_duplicates.get(i1).append(i2)
            elif i2 in HAMM_duplicates.keys() and i1 not in HAMM_duplicates.keys():
                k = 0
                for image_list in HAMM_duplicates.values():
                    if i1 in image_list:
                        k += 1
                if k == 0:
                    HAMM_duplicates.get(i2).append(i1)
            elif i1 not in HAMM_duplicates.keys() and i2 not in HAMM_duplicates.keys():
                k, K = 0, 0
                for keys, image_list in HAMM_duplicates.items():
                    if i1 in image_list:
                        k += 1
                        key1 = keys
                    if i2 in image_list:
                        K += 1
                        key2 = keys
                if k == 0 and K == 0:
                    HAMM_duplicates.update({i1:[i1,i2]})
                elif k > 0 and K == 0:
                    HAMM_duplicates.get(key1).append(i2)
                elif k == 0 and K > 0:
                    HAMM_duplicates.get(key2).append(i1)
    print("Similar Sets: ", len(HAMM_duplicates.keys()))
    return HAMM_duplicates

### Sort dictonary of similar images ###
def sort_image_library(HAMM_duplicates):
    print("\nSorting Similar Sets")
    i = 0
    for images in HAMM_duplicates.values():
        i += 1
        similar = "Similar Set " + str(i)
        FolderDirectory = Path('')
        NewFolderDirectory = FolderDirectory/similar
        if not NewFolderDirectory.exists():
            NewFolderDirectory.mkdir()
        for image in images:
            OldFolderDirectory = FolderDirectory/image
            imageFolderDirectory = NewFolderDirectory/image
            shutil.move(OldFolderDirectory, imageFolderDirectory)

### Statement allows you to run file either as reusable module or standalone program ###
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("\n", "Total Script Time:",(end - start))