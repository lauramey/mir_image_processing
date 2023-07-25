import cv2
from hand_crafted_features import hand_crafted_features
import numpy as np
import glob
import sys, os
from base import DATA_DIR
import csv
import pandas as pd
#Trennzeichen: import os os.sep oder from pathlib import PATH

#DATA_DIR = 'src/static/images/database'
def get_images_paths(image_directory, file_extensions):
    """
    Function to receive every path to a file with ending "file_extension" in directory "image_directory".
    Parameters
    ----------
    image_directory : string
        Image directory. For example: 'static/images/database/'
    file_extensions : tuple
        Tuple of strings with the possible file extensions. For example: '(".png", ".jpg")'
    Returns
    -------
    - image_paths : list
        List of image paths (strings).
    Tasks
    -------
        - Iterate over every file_extension
            - Create a string pattern 
            - Use glob to retrieve all possible file paths (https://docs.python.org/3.7/library/glob.html )
            - Add the paths to a list  (extend)
        - Return result
    """
    images = []
    for extension in file_extensions: 
        images.extend(glob.glob(image_directory + '/*' + extension))

    return images

    


def create_feature_list(image_paths):
    """
    Function to create features for every image in "image_paths".
    Parameters
    ----------
    image_paths : list
        Image paths. List of image paths (strings).
    Returns
    -------
    - result : list of lists.
        List of 'feature_list' for every image. Each image is summarized as list of several features.
    Tasks
    -------
        - Iterate over all image paths
        - Read in the image
        - Extract features with class "feature_extractor"
        - Add features to a list "result"
    """
    feature_extractor = hand_crafted_features()

    features = []
    for imagePath in image_paths: 
       
       features.append(feature_extractor.extract(imagePath))
    
    return features


def write_to_file(feature_list, image_paths, output_path):
    """
    Function to write features into a CSV file.
    Parameters
    ----------
    feature_list : list
        List with extracted features. Should come from 'create_feature_list':
    image_paths : list
        Image paths. List of image paths (strings). Should come from 'get_images_paths':
    output_path : string
        Path to the directory where the index file will be created.
    Tasks
    -------
        - Open file ("output_name")
        - Iterate over all features (image wise)
        - Create a string with all features concerning one image seperated by ","
        - Write the image paths and features in one line in the file [format: image_path,feature_1,feature_2, ..., feature_n]
        - Close file eventually

        - Information about files http://www.tutorialspoint.com/python/file_write.htm 
    """
    
    df = pd.DataFrame(feature_list)
    df['path'] = image_paths

    df.to_csv(output_path)
""" 
    file = open(output_path + '.csv', "w")
    writer = csv.writer(file, delimiter=";")

    for i, feature in enumerate(feature_list):       
        row = image_paths[i] + ';' + ';'.join(str(x) for x in feature) + '\n'
        writer.writerow([row])
        
    file.close() """


def preprocessing_main(image_directory, output_path, file_extensions = (".png", ".jpg")):
    """
    Function which calls 'get_images_paths', 'create_feature_list' and 'write_to_file'
    """

    image_paths = get_images_paths(image_directory, file_extensions)

    feature_list  = create_feature_list(image_paths)
    
    write_to_file(feature_list, image_paths, output_path)


if __name__ == '__main__':
    #TODO: relativer Pfad hier :(
    preprocessing_main(image_directory=os.path.join(DATA_DIR, 'test_images'), output_path=os.path.join(DATA_DIR, 'output'))