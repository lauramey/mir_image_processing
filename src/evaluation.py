from preprocessing import get_images_paths
from query import Query
from irma import IRMA
import glob

import cv2
import sys
import numpy as np
from pathlib import Path
import csv
import os
from tqdm import tqdm
from base import IRMA_DIR, OUTPUT_DIR, IMAGE_DIR
import pandas as pd


def get_image_paths(image_directory, file_extensions):
    
    images = []
    for extension in file_extensions: 
        images.extend(glob.glob(image_directory + '/*' + extension))

    return images

def count_codes(code_path = os.path.abspath(IRMA_DIR) + "/image_codes.csv"): 
    """
    Counts the occurrence of each code in the given "CSV" file.

    Parameters
    ----------
    code_path : string
        Path to the csv file. Default= irma_data/image_codes.csv"
    Returns
    -------
    results : dict
        Occurrences of each code. Key is the code, and value the amount of occurrences.
    Task
    -------
        - If there is no code file -> Print error and return False [Hint: Path(*String*).exists()]
        - Open the code file
        - Read in CSV file [Hint: csv.reader()]
        - Iterate over every row of the CSV file
            - Make an entry in a dict
        - Close file
        - Return results
    """
    code_dict = {}
    codes = pd.read_csv(code_path, delimiter=";", header=None, dtype=str, keep_default_na=False).values
    for code in codes: 
        if code[1] not in code_dict:
            code_dict[code[1]] = 1
        else: 
            code_dict[code[1]] += 1
    
    return code_dict

def precision_at_k(correct_prediction_list, k = None):
    """
    Function to calculate the precision@k.

    Parameters
    ----------
    correct_prediction_list : list
        List with True/False values for the retrieved images.
    k : int
        K value.
    Returns
    -------
    precision at k : float
        The P@K for the given list.
    Task
    -------
        - If k is not defined -> k should be length of list
        - If k > length -> Error
        - If k < length -> cut off correct_prediction_list at k
        - Calculate precision for list
    Examples
    -------

        print("P@K: ", precision_at_k([True, True, True, False]))
        >>> P@K:  0.75

        print("P@K: ", precision_at_k([True, True, True, False], 2))
        >>> P@K:  1.0
    """
    if k is None:
        k = len(correct_prediction_list)
    elif k > len(correct_prediction_list):
        print("k > correct_prediction")
        exit()

    k_predictions = correct_prediction_list[:k]

    return sum(k_predictions)/k       


def average_precision(correct_prediction_list, amount_relevant= None):
    """
    Function to calculate the average precision.

    Parameters
    ----------
    correct_prediction_list : list
        List with True/False values for the retrieved images.
    amount_relevant : int
        Number of relevant documents for this query. Default is None.
    Returns
    -------
    average precision : float
        The average precision for the given list.
    Tasks
    -------
        - If amount_relevant is None -> amount_relvant should be the length of 'correct_prediction_list'
        - Iterate over 'correct_prediction_list'
            - Calculate p@k at each position
        - sum up values and divide by 'amount_relevant'
    Examples
    -------

        print("AveP: ", average_precision([True, True, True, False]))
        >>> P@AveP:  0.75

        print("AveP: ", average_precision([True, True, False, True], 3))
        >>> AveP:  0.9166666666666666
    """
    if amount_relevant is None:
        amount_relevant = len(correct_prediction_list)

    p_at_k_total = 0
    for idx, pred in enumerate(correct_prediction_list):
        p_at_k_total += precision_at_k(correct_prediction_list, idx+1) * pred

    return p_at_k_total / amount_relevant

def mean_average_precision(limit = 10000):
    """
    Function to calcualte the mean average precision of the database.

    Parameters
    ----------
    limit : int
        Limit of the query. Default is None.
    Returns
    -------
    mean average precision : float
        The meanaverage precision of the selected approach on the database.
    Tasks
    -------
        - Create irma object and count codes.
        - Iterate over every image path (you can use 'tqdm' to check the run time of your for loop)
            - Create and run a query for each image
            - Compute a correct_prediction_list
            - Remove the first element (its the query image)
            - Compute AP (function) and save the value
        - Compute mean of APs
    """
    irma = IRMA()
    query = Query(path_to_index= os.path.abspath(OUTPUT_DIR))
    code_count = count_codes()
    
    ap_sum = 0
    image_paths = get_image_paths(os.path.abspath(IMAGE_DIR), file_extensions = (".png", ".jpg"))

    for image_path in tqdm(image_paths):
        query.set_image_name(query_image_name=image_path)
        query_result = query.run()[1:]

        source_irma = irma.get_irma([image_path])[0]
        preds_irma = irma.get_irma(list(zip(*query_result))[0])
        
        correct_prediction_list = []
        for pred in preds_irma:
            correct_prediction_list.append(pred == source_irma)
            
            ap_sum += average_precision(correct_prediction_list, code_count[source_irma]) 

        
    return ap_sum/len(image_paths) 

if __name__ == "__main__":

    test = [True, False, True, False]
    print("Examples with query results: ", str(test)) 
    print("P@K: ", precision_at_k(test, 3))
    print("AveP: ", average_precision(test, 3))

    result = mean_average_precision()
    print("\n\n")
    print("-"*50)
    print("Evaluation of the database")
    print("MAP: ", result)