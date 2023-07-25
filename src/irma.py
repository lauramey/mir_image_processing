import csv
import os
import pandas as pd
from base import IRMA_DIR
from pathlib import Path

def csv_to_dict(file_path, delimiter=";"):
    """
    Function to read in a csv file and create a dict based on the first two columns.

    Parameters
    ----------
    - file_path : string
        The filepath to the CSV file.
    
    Returns
    -------
    - csv_dict : dict
        The dict created from the CSV file.

    Tipps
    -------
    - Read in the CSV file from the path
    - For each row, add an entry to a dict (first column is key, second column is value)
    - Return the dict
    """
    return dict(zip(*pd.read_csv(file_path, delimiter=delimiter, header=None, dtype=str, keep_default_na=False).values.T))

    
class IRMA:
    """
    Class to retrieve the IRMA code and information for a given file.
    """
    labels_long = ["Technical code for imaging modality", "Directional code for imaging orientation", "Anatomical code for body region examined", "Biological code for system examined"]
    labels_short = ["Imaging modality", "Imaging orientation", "Body region", "System"]

    def __init__(self, dir_path=os.path.abspath(IRMA_DIR)):
        """
        Constructor of an IRMA element.

        Parameters
        ----------
        - dir_path : string
            The path where the irma data is. There should be a "A.csv", "B.csv", "C.csv", "D.csv" and "image_codes.csv" file in the directory.

        Tipps
        -------
        - Create a dict for part A, B, C, and D of the IRMA code (user csv_to_dict(file_path))
        - Save the dicts (list) as class variable
        - Save "image_codes.csv" as dict in a class variable
        """
        self.a = csv_to_dict(os.path.join(dir_path, 'A.csv'))
        self.b = csv_to_dict(os.path.join(dir_path, 'B.csv'))
        self.c = csv_to_dict(os.path.join(dir_path, 'C.csv'))
        self.d = csv_to_dict(os.path.join(dir_path, 'D.csv'))
        self.codes_list = [self.a, self.b, self.c, self.d]
        self.image_codes = csv_to_dict(os.path.join(dir_path, 'image_codes.txt'), " ")

    def get_irma(self, image_names):
        """
        Function to retrieve irma codes for given image names.

        Parameters
        ----------
        - image_names : list
            List of image names.

        Returns
        -------
        - irma codes : list
            Retrieved irma code for each image in 'image_list'

        Tipps
        -------
        - Remove file extension and path from all names in image_names. Names should be in format like first column of 'image_codes.csv'
        - Use self.image_dict to convert names to codes. ('None' if no associated code can be found)
        - Return the list of codes
        """
        codes = []
        for image_name in image_names:
            image_name_cleaned = Path(image_name).stem 

        
            #print(type(list(self.image_codes.keys())[0]))
            try:
                #print(str(image_name_cleaned))
                image_codes = self.image_codes[image_name_cleaned]
                codes.append(image_codes)
            except Exception as e:
                print("code not found", e.with_traceback)

        return codes

    def decode_as_dict(self, code):
        """
        Function to decode an irma code to a dict.

        Parameters
        ----------
        - code : str
            String to decode.

        Returns
        -------
        - decoded : dict

        Tipps
        -------
        - Make use of 'labels_short'
        - Possible solution: {'Imaging modality': ['x-ray', 'plain radiography', 'analog', 'overview image'], ...}
        - Solution can look different
        """
        codes = code.split("-")
        decoded_dict = {self.labels_short[0]:[], self.labels_short[1]:[],self.labels_short[2]:[],self.labels_short[3]:[]}
        
        for idx, image_code in enumerate(codes):
            temp_list = []
            for index, _ in enumerate(image_code):
                try:
                    temp_list.append(self.codes_list[idx][image_code[0:index+1]])
                except Exception as e:
                    # temp_list.append('not in the files')
                    pass
            decoded_dict[self.labels_short[idx]] = temp_list

        return decoded_dict

    def decode_as_str(self, code):
        """
        Function to decode an irma code to a str.

        Parameters
        ----------
        - code : str
            String to decode.

        Returns
        -------
        - decoded : str
            List of decoded strings.

        Tipps
        -------
        - Make use of 'decode_as_dict'
        - Possible solution: ['Imaging modality: x-ray, plain radiography, analog, overview image', 'Imaging orientation: coronal, anteroposterior (AP, coronal), supine', 'Body region: abdomen, unspecified', 'System: uropoietic system, unspecified']
        - Solution can look different -> FLASK will use this representation to visualize the information on the webpage.
        """
        codes = code.split("-")
        #decoded_dict = {self.labels_short[0]:[], self.labels_short[1]:[],self.labels_short[2]:[],self.labels_short[3]:[]}
        decoded_list = []
        for idx, image_code in enumerate(codes):
            temp_str = self.labels_short[idx] + ": "
            for index, _ in enumerate(image_code):
                try:
                    temp_str = temp_str + self.codes_list[idx][image_code[0:index+1]] + ", "
                except Exception as e:
                    # temp_list.append('not in the files')
                    pass
            decoded_list.append(temp_str)

        return decoded_list

if __name__ == '__main__':
    csv_to_dict(os.path.abspath(os.path.join(IRMA_DIR, 'A.csv')))
    image_names = ["3145.png"]

    irma = IRMA()

    codes = irma.get_irma(image_names)
    print("Codes: ", codes)

    if codes is not None:
        code = codes[0]
        print("Dict: \n{}\n\n".format(irma.decode_as_dict(code)))
        #print("String: \n{}\n\n".format(irma.decode_as_str(code)))

    '''
    Result could look like this:


    Codes:  ['1121-127-700-500']
    Dict:
    {'Imaging modality': ['x-ray', 'plain radiography', 'analog', 'overview image'], 'Imaging orientation': ['coronal', 'anteroposterior (AP, coronal)', 'supine'], 'Body region': ['abdomen', 'unspecified'], 'System': ['uropoietic system', 'unspecified']}


    String:
    ['Imaging modality: x-ray, plain radiography, analog, overview image', 'Imaging orientation: coronal, anteroposterior (AP, coronal), supine', 'Body region: abdomen, unspecified', 'System: uropoietic system, unspecified']
    '''