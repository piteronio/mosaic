import os
from pathlib import Path
import glob

import math

import cv2

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


IMAGES_FOLDER = Path("images/")
LIBRARY_FOLDER = Path("library/")
MAS_DATA_FOLDER = Path("library/mas_data/")
MASTER_FOLDER = Path("master/")
OUTPUT_FOLDER = Path("output/")









class Mosaic_project:
    """
    Class for mosaic projects.

    ...

    Attributes
    ----------
    height : integer
        target height of tiles in mosaic
    width : integer
        target width of tiles in mosaic

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    """
    def __init__(self, height=100, width=120):
        #Test whether old project files already exist.
        if len(os.listdir(LIBRARY_FOLDER)) > 2:
            print("Would you like to:")
            print("(1) continue the old project, or")
            print("(2) start a new one and clear all old files?")
            choice = "0"
            while choice not in ["1", "2"]:
                choice = input("Please answer with 1 or 2: ")
            if choice == "1":
                try:
                    library_log = open(LIBRARY_FOLDER / "libary_log.txt", "r")
                    values_list=library_log.readlines()
                    library_log.close()
                except:
                    raise FileNotFoundError("Old project is corrupted, please start a new one.")
                else:
                    height = int(values_list[0])
                    width = int(values_list[1])
            if choice == "2":
                clear_library()
                clear_mas_data()
        self.height = height
        self.width = width

            

    def get_height(self):
        '''height getter'''
        return self.height
    def get_width(self):
        '''width getter'''
        return self.width

    def set_height(self,height):
        '''height setter'''
        self.height=height
        return None
    def set_width(self,width):
        '''height setter'''
        self.width=width
        return None

    def build_library(self):
        '''build an image library by cropping and resizing each
        image found in the images folder and its subfolders, in 
        accordance with the specified height and width values.
        
        Save all the processed images in the library folder as well as
        a file averages.csv containing average colour values of all
        the processed images.
        '''
        #check whether a library already exists
        if len(os.listdir(LIBRARY_FOLDER)) > 2:
            print("Would you like to rebuild the library (y/n)?")
            print("WARNING: this will delete all intermediate files.")
            choice = input("Input: ")
            while choice not in ["y", "n"]:
                print("Please respond with y or n.")
                choice = input("Input: ")
            if choice == "n":
                return None
            else:
                clear_library()
                clear_mas_data()
                print("library cleared")
        #retrieve target height and width of image tiles
        height = self.get_height()
        width = self.get_width()
        #create list of all files in images folder and its subfolders
        file_list = images_files()
        total_files = len(file_list)    #total number of files
        image_counter = 0               #tracks number of images among files
        percent = 0                     #tracks completion percentage
        file_counter = 0                #tracks number of files processed
        #We save average colour values of images during processing
        #in the following list.
        average_colours_list = []
        print("building image library initiated")
        print("images processed:   0%")
        for file in file_list:
            image = cv2.imread(file)
            file_counter += 1
            percent_new = int((file_counter * 100) // total_files)
            if percent_new > percent:
                percent = percent_new
                print("images processed: "+((3 - len(str(percent))) * " ")+str(percent)+"%")
            #Check whether the file is an image or not.
            if image is None:   #If not, then skip to the next iteration.
                continue
            #Crop and resize image.
            image = image_processor(image, height, width)
            #Save processed image into the library folder.
            cv2.imwrite(str(LIBRARY_FOLDER / ("proc_im_" + str(image_counter) + ".jpg")), image)
            #Compute the average colour values of the processed image.
            average_colours = image.mean(axis=1).mean(axis=0)
            average_colours_list.append(average_colours)
            image_counter += 1
        #Save the average colour values of the images in the library
        #as averages.csv in the library folder.
        average_colours_list = pd.DataFrame(average_colours_list)
        average_colours_list.to_csv(str(LIBRARY_FOLDER / "averages.csv"))
        #Create a log in the library folder with the height, width
        #and number of images in the library.
        log_file = open(LIBRARY_FOLDER / "libary_log.txt", "w")
        log_file.writelines([str(height)+"\n", str(width)+"\n", str(image_counter)])
        log_file.close()
        print("building image library completed")
        return None
    
    def master_processor(self, max_im=False):
        '''process master image'''
        #check whether a master image has already been processed.
        if len(os.listdir(MAS_DATA_FOLDER)) > 1:
            print("Would you like to reprocess a master image?")
            choice = "0"
            while choice not in ["y", "n"]:
                choice = input("Please respond with y or n: ")
            if choice == "n":
                return None
            else:
                clear_mas_data()
        files = os.listdir(MASTER_FOLDER)
        try:
            files.remove("master folder readme.md")
        except ValueError:
            pass
        if len(files) == 0:
            raise FileNotFoundError("No image found in master folder.")
        elif len(files) == 1:
            file = files[0]
        else:
            print("Several files found in master folder:")
            for index in range(len(files)):
                print("("+str(index+1)+") "+files[index])
            choice = "0"
            options = [str(k+1) for k in range(len(files))]
            while choice not in options:
                choice = input("Which image would you like to use (1 or 2 or...)? ")
            file = files[int(choice) - 1]
        image = cv2.imread(file)
        return image
            
        
        
        
        
        
        
        
        
        
        

def image_processor(image, height, width):
    '''Croppes and resizes image in accordance with specified height
    and width, whilst preventing stretching or shrinking.
    --------
    Keyword arguments:
    image  -- array-like image to be processed
    height -- target height of output image
    width  -- target width of output image
    '''
    he_im = image.shape[0]
    wi_im = image.shape[1]
    #Compare aspect ratio of input image and target aspect ratio.
    if height / width < he_im / wi_im:     #In this case crop vertically.
        top_lim = math.ceil((he_im-1) / 2 - wi_im * height / (2 * width))
        bot_lim = math.floor((he_im-1) / 2 + wi_im * height / (2 * width))+1
        cropped_im = image[top_lim : bot_lim, 0 : wi_im]
    else:                                   #Otherwise crop horizontally.
        lft_lim = math.ceil((wi_im-1) / 2 - he_im * width / (2 * height))
        rgt_lim = math.floor((wi_im-1) / 2 + he_im * width / (2 * height))+1
        cropped_im = image[0 : he_im, lft_lim : rgt_lim]
    #Resize cropped image.
    output = cv2.resize(cropped_im, (width, height), interpolation=cv2.INTER_AREA)
    return output

def clear_library():
    '''Clear library folder.'''
    files = glob.glob("library/*")
    for file in files:
        if file not in [str(LIBRARY_FOLDER / "library_readme.md"),
                        str(MAS_DATA_FOLDER)]:
            os.remove(file)
    return None

def clear_mas_data():
    '''Clear mas_data subfolder of library.'''
    files = glob.glob("library/*/*")
    for file in files:
        if not file == str(MAS_DATA_FOLDER / "mas_data_readme.md"):
            os.remove(file)
    return None

def images_files():
    '''Return list of all file-paths in images folder.'''
    file_list = []
    for root, _, files in os.walk(IMAGES_FOLDER):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def optimal(nr_im, asp_tile, asp_mast):
    '''Compute optimal number of rows and columns of image tiles in mosaic
    with nr_im images and aspect ratios asp_tile and asp_mast for the
    image tiles and master image respectively.
    --------
    Keyword arguments:
    nr_im    --- number of image tiles in library
    asp_tile --- aspect ratio of image tiles
    asp_mast --- aspect ratio of master tiles
    --------
    returns tuple with optimal number of rows and columns.
    '''
    def deviation(nr_row, nr_col):
        '''compute squared difference between aspect ratios of master image
        and mosaic with number of rows and columns as specified.
        '''
        return ((nr_row / nr_col) * asp_tile - asp_mast)**2
    #construct list of possible row and column numbers
    options = [(nr_row, nr_im // nr_row) for nr_row in range(1,nr_im+1)]
    #find option with minimal deviation.
    option_best = min(options, key = deviation)
    return option_best
        