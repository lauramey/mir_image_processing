from hand_crafted_features import hand_crafted_features
#from ae import auto_encoder
from searcher import Searcher
import cv2
import os, sys
import numpy as np
from base import OUTPUT_DIR, IMAGE_DIR

class Query:
    
    def __init__(self, path_to_index):
        """
        Init function of the Query class. Sets 'path_to_index' to the class variable 'path_to_index'. 
        Class variables 'query_image_name' and 'results' are set to None.
        Parameters
        ----------
        path_to_index : string
            Path to the index file.
        """
        self.path_to_index = path_to_index
        self.query_image_name = None
        self.results = None


    def set_image_name(self, query_image_name):
        """
        Function to set the image name if it does not match the current one. Afterwards the image is loaded and 
        features are retrieved.
        Parameters
        ----------
        query_image_name : string
            Image name of the query. For example: 'static/images/query/test.png'
        Tasks
        ---------
            - Check if 'query_image_name' is different to 'self.query_image_name'
            - If yes:
                - Set 'self.results' to None
                - Overwrite 'query_image_name'
                - Read in the image and save it under 'self.query_image'
                - Calculate features
        """
        if(self.query_image_name is not query_image_name):
            self.results = None
            self.query_image_name = query_image_name
            self.query_image = cv2.imread(self.query_image_name, cv2.IMREAD_GRAYSCALE)
            self.calculate_features()


    def calculate_features(self):
        """
        Function to calculate features for the query image.
        Tasks
        ---------
            - Check if "self.query_image" is None -> exit()
            - Extract features wit "FeatureExtractor" and set to "self.features"
        """
        if self.query_image is None: 
            exit()
        feature_extractor = hand_crafted_features()
        self.features = feature_extractor.extract(self.query_image)


    def run(self, counter=0, quantity = 10):
        """
        Function to start a query if results have not been computed before.
        Parameters
        ----------
        limit : int
            Amount of results that will be retrieved. Default: 10.
        Returns
        -------
        - results : list
            List with the 'limit' first elements of the 'results' list. 

        Tasks
        ---------
            - Check if 'self.results' is None
            - If yes:
                - Create a searcher and search with features
                - Set the results to 'self.results'
            - Return the 'limit' first elements of the 'results' list.
        """
        if(self.results is None):
            searcher = Searcher(self.path_to_index)
            self.results = searcher.search(self.features)
            #print(self.results)
        return self.results[:quantity]
    
    def rocchio(original_query, relevant, non_relevant, a = 1, b = 0.8, c = 0.1):
        """
        Function to adapt features with rocchio approach.

        Parameters
        ----------
        original_query : list
            Features of the original query.
        relevant : list
            Features of the relevant images.
        non_relevant : list
            Features of the non relevant images.
        a : int
            Rocchio parameter.
        b : int
            Rocchio parameter.
        c : int
            Rocchio parameter.
        Returns
        -------
        - features : list
            List with of features.
        """

    # TODO:
    pass

if __name__ == "__main__":
    query = Query(path_to_index= os.path.abspath(OUTPUT_DIR))
    query.set_image_name(query_image_name=os.path.abspath(IMAGE_DIR + "/3319.png"))
    query_result = query.run()
    print("Retrieved images: ", query_result)
    cv2.imshow('input image', cv2.imread(os.path.abspath(IMAGE_DIR + "/3319.png")))
    cv2.waitKey(0)

    cv2.imshow('First result', cv2.imread(os.path.abspath(query_result[0][0])))
    cv2.waitKey(0)
    cv2.imshow('Second result', cv2.imread(os.path.abspath(query_result[1][0])))
    # add wait key. window waits until user presses a key
    cv2.waitKey(0)
    cv2.destroyAllWindows()
