import numpy as np
import cv2
import os
from base import IMAGE_SRC_DIR
from sklearn.manifold import SpectralEmbedding
 
class hand_crafted_features:
    """
    Class to extract features from a given image.

    Example
    --------
    feature_extractor = hand_crafted_features()
    features = feature_extractor.extract(example_image)
    print("Features: ", features)
    """

    def extract(self, imagePath):
        """
        Function to extract features for an image.
        Parameters
        ----------
        - image : (x,y) array_like
            Input image.

        Returns
        -------
        - features : list
            The calculated features for the image.
        """
        full_path = os.path.abspath(IMAGE_SRC_DIR + os.path.basename(imagePath))
        image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        assert image is not None
        #features = self.histogram(image)
        features = self.laplacian_eigenmaps(image)
        #print(features)
        # TODO: You can even extend features with another method:
        #features.extend(self.thumbnail_features(image))

        return features
    
    def laplacian_eigenmaps(self, image):
        embedding = SpectralEmbedding(n_components=2)
        image_map = embedding.fit_transform(image[:100]).flatten()
        return image_map

    def histogram(self, image):
        """
        Function to extract histogram features for an image.
        Parameters
        ----------
        - image : (x,y) array_like
            Input image.
        
        Returns
        -------
        - features : list
            The calculated features for the image.

        Tipps
        -----
            - Use 'cv2.calcHist'
            - Create a list out of the histogram (hist.tolist())
            - Return a list (flatten)
        """
        assert image is not None

        hist = cv2.calcHist([image], [0], None, [256], [0,256]).tolist()
        for index, item in enumerate(hist):
            hist[index] = int(item[0])

        return hist

    def thumbnail_features(self, image):
        """
        Function to create a thumbnail of an image and return the image values (features).
        Parameters
        ----------
        - image : (x,y) array_like
            Input image.
        
        Returns
        -------
        - features : list
            The calculated features for the image.

        Tipps
        -----
            - Resize image (dim(30,30))
            - Create a list from the image (np array)
            - Return a list (flatten)
        """
        resized_image = np.array(cv2.resize(src=image, dsize=(30,30), interpolation=cv2.INTER_AREA))
        return resized_image.flatten()


    def partitionbased_histograms(self, image, factor = 10):
        """
        Function to create partition based histograms.
        Parameters
        ----------
        - image : (x,y) array_like
            Input image.
        - factor : int
            Partitioning factor.
        
        Returns
        -------
        - features : list
            The calculated features for the image.

        Tipps
        -----
            - Resize image (dim(200, 200))
            - Observe (factor * factor) image parts
            - Calculate a histogramm for each part and add to feature list
        """
        #TODO:
        pass


    def extract_features_spatial(self, image, factor = 10):
        """
        Function to extract spatial features.
        Parameters
        ----------
        - image : (x,y) array_like
            Input image.
        - factor : int
            Partitioning factor.
        
        Returns
        -------
        - features : list
            The calculated features for the image.

        Tipps
        -----
            - Resize image (dim(200, 200))
            - Observe (factor * factor) image parts
            - Calculate max, min and mean for each part and add to feature list
        """
        #TODO:
        pass


    def image_pixels(self, image):
        """
        Function to return the image pixels as features.
        Example of a bad implementation. The use of pixels as features is highly inefficient!

        Parameters
        ----------
        - image : (x,y) array_like
            Input image.
        
        Returns
        -------
        - features : list
            The calculated features for the image.
        """
        # cast image to list of lists
        features =  image.tolist()
        # flatten the list of lists
        features = [item for sublist in features for item in sublist]
        # return 
        return features

if __name__ == '__main__':
    # Read the test image
    # TODO: You can change the image path here:
    example_image = cv2.imread("../../solution/static/images/database/1880.png", cv2.IMREAD_GRAYSCALE)

    # Assert image read was successful
    assert example_image is not None

    # create extractor
    feature_extractor = hand_crafted_features()

    # describe the image
    features = feature_extractor.extract(example_image)

    # print the features
    print("Features: ", features)
    print("Length: ", len(features))
