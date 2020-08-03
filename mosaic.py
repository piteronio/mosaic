#created by Pieter Roffelsen
"""This module allows one to build a mosaic of a given master image,
using images from a given collection of images. For details, please see the README.
"""


import os
from pathlib import Path
import glob
import json

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



class _Parameters:
    """
    Class for parameters of mosaic projects.
    ...

    Attributes
    ----------
    dic : dictionary with keys (parameters):
        "height" - height of tiles in mosaic
        "width"  - width of tiles in mosaic
        "nr_im"  - number of images in library
        "nr_row" - number of rows of images in mosaic
        "nr_col" - number of columns of images in mosaic
    """
    def __init__(self, height="", width=""):
        self.dic = {"height" : height, "width" : width,
                    "nr_im" : "", "nr_row" : "",
                    "nr_col" : ""}

    def get(self, parameter):
        '''Get value of parameter.'''
        return self.dic[parameter]

    def set(self, parameter, value):
        '''Set value of parameter.'''
        self.dic[parameter] = value
        return None

    def save(self):
        '''Save parameters to file in library folder.'''
        log = open(LIBRARY_FOLDER / "log.txt", "w")
        json_dump = json.dumps(self.dic)
        log.write(json_dump)
        log.close()
        return None

    def load(self):
        '''Load parameters from file.'''
        json_file = open(LIBRARY_FOLDER / "log.txt", "r")
        dic = json.load(json_file)
        self.dic = dic
        return None


class MosaicProject:
    """
    Class for mosaic projects.
    ...

    Attributes
    ----------
    parameters : instance of the _Parameter class.
    Contains values of relevant parameters of a mosaic project.

    Methods
    -------
    build_library(self)
    Build library from images in image folder.

    process_master(self, max_im=0)
    Find optimal assignment of library images to tiles in mosaic,
    using no more than max_im images if max_im > 0.

    print_mosaic(self, tuning=None)
    Print a mosaic with optional tuning

    build_mosaic(self, max_im=0, tuning=None)
    Call build_library, process_master and print_mosaic.
    """
    def __init__(self, height=100, width=120):
        #initialise parameters of mosaic.
        self.parameters = _Parameters(height=height, width=width)
        #Test whether old project files already exist.
        if len(os.listdir(LIBRARY_FOLDER)) > 2:
            print("Would you like to:")
            print("(1) continue the old project, or")
            print("(2) start a new one and clear the old one?")
            choice = "0"
            while choice not in ["1", "2"]:
                choice = input("Please answer with 1 or 2: ")
            if choice == "1":
                try:
                    self.parameters.load()
                except:
                    raise FileNotFoundError("Old project is corrupted, please start a new one.")
            if choice == "2":
                _clear_library()
                _clear_mas_data()

    def get_parameters(self):
        '''parameters getter'''
        return self.parameters

    def build_library(self):
        '''build an image library by cropping and resizing each
        image found in the images folder and its subfolders, in
        accordance with the specified height and width values.
        ---------
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
                _clear_library()
                _clear_mas_data()
                print("library cleared")
        #retrieve mosaic parameters
        param = self.get_parameters()
        #create list of all files in images folder and its subfolders
        file_list = _images_files()
        total_files = len(file_list)    #total number of files
        image_counter = 0               #tracks number of images among files
        percent = 0                     #tracks completion percentage
        file_counter = 0                #tracks number of files processed
        #We save average colour values of images during processing
        #in the following list.
        average_colours_list = []
        print("Building of image library initiated.")
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
            image = _image_processor(image, param.get("height"), param.get("width"))
            #Save processed image into the library folder.
            cv2.imwrite(str(LIBRARY_FOLDER / ("lib_im_" + str(image_counter) + ".jpg")), image)
            #Compute the average colour values of the processed image.
            average_colours = image.mean(axis=1).mean(axis=0)
            average_colours_list.append(average_colours)
            image_counter += 1
        #Save the average colour values of the images in the library
        #as averages.csv in the library folder.
        average_colours_list = pd.DataFrame(average_colours_list)
        average_colours_list.to_csv(str(LIBRARY_FOLDER / "averages.csv"))
        #Set parameter "nr_im" to image_counter.
        param.set("nr_im", image_counter)
        #Save parameters on file.
        param.save()
        print("Building of image library completed.")
        return None

    def process_master(self, max_im=0):
        '''Process master image:
        -load master image from the master folder,
        -determine optimal number of rows and columns of tiles
         to use for mosaic, with the additional constraint that the total
         number of tiles is less than max_im, in case max_im > 0.
        -compute optimal assignment of images in library to mosaic tiles.
        -save optimal assignment as assignment.csv in mas_data folder.
        --------
        Keyword arguments:
        max_im -- maximum number of tiles to use in mosaic.
        '''
        #check whether a master image has already been processed.
        if len(os.listdir(MAS_DATA_FOLDER)) > 1:
            print("Would you like to reprocess a master image?")
            choice = "0"
            while choice not in ["y", "n"]:
                choice = input("Please respond with y or n: ")
            if choice == "n":
                return None
            else:
                _clear_mas_data()
        #load master image
        mas_im = _load_mas_im()
        print("Master image processing initiated.")
        #get aspect ratio of master image
        mas_dim = mas_im.shape
        asp_mast = mas_dim[0] / mas_dim[1]
        #get aspect ratio of tile images
        param = self.get_parameters()
        asp_tile = param.get("height") / param.get("width")
        #set maximum number of image tiles in mosaic
        if max_im > 0:
            nr_im_used = min(param.get("nr_im"), max_im)
        else:
            nr_im_used = param.get("nr_im")
        #find optimal numbers of rows and columns in mosaic
        (nr_row, nr_col) = _optimal(nr_im_used, asp_tile, asp_mast)
        print("Your mosaic will consist of "+str(nr_row)+\
              " rows and "+str(nr_col)+" columns of images.")
        #set numbers of rows and columns parameters
        param.set("nr_row", nr_row)
        param.set("nr_col", nr_col)
        param.save()
        #resize master image to shape of to be built mosaic
        mas_im_resized = _image_resizer(mas_im, nr_row * param.get("height"),\
                                        nr_col * param.get("width"))
        #save resized master image in mas_data subfolder of library
        cv2.imwrite(str(MAS_DATA_FOLDER / "mas_res_1.jpg"), mas_im_resized)
        #extract average colour values of sub_images of master_image
        mas_im_data = _extract_data(mas_im_resized, nr_row, nr_col)
        del mas_im_resized
        #compute optimal assignment of library images to mosaic tiles
        assignment = _optimal_assignment(mas_im_data)
        #save optimal assignment as csv file in mas data subfolder
        pd.DataFrame(assignment).to_csv(str(MAS_DATA_FOLDER / "assignment.csv"))
        print("Master image processing completed.")
        return None

    def print_mosaic(self, tuning=None):
        '''Build mosaic and save in output folder.
        -------
        There are four allowed input forms for tuning:
        1) tuning=None, then no tuning is performed.
        -
        2) tuning=[w_0,w_1,...,w_n], where elements are integers
        and their sum equals 100. Then mosaic is tuned, namely
        the following weighted average is taken
        mosaic = + w_0/100 * mosaic_basic
                 + w_1/100 * unfiltered resized master image
                 + w_2/100 * 1 time filtered master image
                 |
                 + w_n/100 * (n-1) times filtered master image,
        where mosaic_basic is the untuned mosaic,
        and the filtered master images are obtained by repeated filtering
        with uniform kernel of shape the same as a tile.
        -
        3) tuning = "tuning_1", which yields same as tuning = [80, 13, 0, 0, 7] .
        -
        4) tuning = "tuning_2", which yields same as tuning = [70, 15, 0, 0, 15] .
        '''
        #load optimal assignment from file
        assignment_df = pd.read_csv(str(MAS_DATA_FOLDER / "assignment.csv"), index_col=0)
        assign = {}
        for index in range(len(assignment_df)):
            assign[index] = assignment_df.iloc[index][1]
        #retrieve parameter values
        param = self.get_parameters()
        height = param.get("height")
        width = param.get("width")
        nr_col = param.get("nr_col")
        nr_row = param.get("nr_row")
        #if tuning is None, average will not involve any filtered master image.
        if tuning is None:
            weights = [100]
        #set pre-defined weights
        elif tuning == "tuning_1":
            weights = [80, 13, 0, 0, 7]
        elif tuning == "tuning_2":
            weights = [70, 15, 0, 0, 15]
        else:
            #check whether parameter tuning is a list of weights
            if _weights_check(tuning):
                weights = tuning
            #if not, raise ValueError
            else:
                raise ValueError("Parameter tuning does not have correct form.")
        #compute support of weights
        support = [k for k, weight in enumerate(weights) if weight]
        #Construct missing filtered versions of master image necessary
        #for building mosaic, if any are missing.
        self._filter_master(max(support))
        #load all relevant filtered versions of master image in dictionary
        mas_res = {}
        for index in support[1 : ]:
            mas_res[index] = cv2.imread(str(MAS_DATA_FOLDER / ("mas_res_"+str(index)+".jpg")))
        #initiate mosaic
        print("Building of mosaic initiated.")
        mosaic = np.zeros((nr_row * height, nr_col * width, 3), dtype=np.uint8)
        #build mosaic tile by tile, looping over rows and columns
        for row_ind in range(0, nr_row):
            for col_ind in range(0, nr_col):
                #load image in library used for tile
                lib_im_nr = assign[row_ind * nr_col + col_ind]
                image_loc = LIBRARY_FOLDER / ("lib_im_"+str(lib_im_nr)+".jpg")
                lib_im = cv2.imread(str(image_loc))
                #start computing weighted average
                tile = weights[0] * np.array(lib_im, dtype=np.uint16)
                #define limits of tile within mosaic
                top_lim = row_ind * height
                bot_lim = (row_ind + 1) * height
                lef_lim = col_ind * width
                rig_lim = (col_ind + 1) * width
                #take weighted average with filtered versions of master image
                for index in support[1 : ]:
                    mas_res_part = mas_res[index][top_lim : bot_lim, lef_lim : rig_lim]
                    tile += weights[index] * np.array(mas_res_part, dtype=np.uint16)
                tile = tile // 100
                #save tile to mosaic
                mosaic[top_lim : bot_lim, lef_lim : rig_lim] = tile
        print("Building of mosaic completed.")
        #use mosaic_namer() to determine name of mosaic
        mosaic_name = _mosaic_namer(weights)
        #save mosaic to file in output folder
        cv2.imwrite(str(OUTPUT_FOLDER / mosaic_name), mosaic)
        print("Mosaic has been written to output folder.")
        return None

    def _filter_master(self, level):
        '''build filtered versions of resized master image through repeated
        filtering with kernel the shape of a tile and save them in mas_data.
        --------
        Keyword arguments:
        level -- number of repetitions of filtering
        '''
        #determine existing filtered versions of master image
        existent = glob.glob("library/*/mas_res_*.jpg")
        level_cur = len(existent)
        #if current level is higher or equal to target level, nothing left to do.
        if level <= level_cur:
            return None
        #load image with highest existing level of filtering
        mas_res = cv2.imread(str(MAS_DATA_FOLDER / ("mas_res_"+str(level_cur)+".jpg")))
        #retrieve parameter values
        param = self.get_parameters()
        height = param.get("height")
        width = param.get("width")
        #define kernel with the shape of a tile
        kernel = np.ones((height, width), np.float32) / (height * width)
        #apply repeating filtering and save results in mas_data folder
        for index in range(level_cur, level):
            mas_res = cv2.filter2D(mas_res, -1, kernel)
            cv2.imwrite(str(MAS_DATA_FOLDER / ("mas_res_"+str(index+1)+".jpg")), mas_res)
        return None

    def build_mosaic(self, max_im=0, tuning=None):
        '''Build mosaic and save it in output folder.'''
        #build library
        self.build_library()
        #process master image
        self.process_master(max_im=max_im)
        #build mosaic
        self.print_mosaic(tuning=tuning)

#BUILD_LIBRARY HELPER FUNCTIONS
def _clear_library():
    '''Clear library folder.'''
    files = glob.glob("library/*")
    for file in files:
        if file not in [str(LIBRARY_FOLDER / "README.md"),
                        str(MAS_DATA_FOLDER)]:
            os.remove(file)
    return None

def _clear_mas_data():
    '''Clear mas_data subfolder of library.'''
    files = glob.glob("library/*/*")
    for file in files:
        if not file == str(MAS_DATA_FOLDER / "README.md"):
            os.remove(file)
    return None

def _images_files():
    '''Return list of all file-paths in images folder.'''
    file_list = []
    for root, _, files in os.walk(IMAGES_FOLDER):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def _image_processor(image, height, width):
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

#PROCESS_MASTER HELPER FUNCTIONS
def _load_mas_im():
    '''Load master image from master folder and return it.
    ---------
    If there are several files in master folder, ask user to choose one.
    If there is no image file in master folder, raise FileNotFoundError.
    '''
    #get all files in master folder.
    files = os.listdir(MASTER_FOLDER)
    #remove master folder readme from list
    try:
        files.remove("README.md")
    except ValueError:
        pass
    #if no files left, raise FileNotFoundError
    if not files:
        raise FileNotFoundError("No image found in master folder.")
    #if there is precisely one file left, get this file
    elif len(files) == 1:
        file = files[0]
        mas_im = cv2.imread(str(MASTER_FOLDER / file))
        #if file is not really an image, raise FileNotFoundError
        if mas_im is None:
            raise FileNotFoundError("No image found in master folder.")
    #if there are several files left, ask user to choose file
    else:
        print("Several files found in master folder:")
        for index, file in enumerate(files):
            print("("+str(index+1)+") "+file)
        choice = "0"
        options = [str(k+1) for k in range(len(files))]
        while choice not in options:
            choice = input("Which image would you like to use (1 or 2 or...)? ")
        file = files[int(choice) - 1]
        #open file as image
        mas_im = cv2.imread(str(MASTER_FOLDER / file))
        #if file is not really an image, raise TypeError
        if mas_im is None:
            raise TypeError("Chosen file isn't recognised as an image.")
    return mas_im

def _optimal(nr_im, asp_tile, asp_mast):
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
    #construct list of possible row and column numbers
    options = [(nr_row, nr_im // nr_row) for nr_row in range(1, nr_im+1)]
    def deviation(option):
        '''compute squared difference between aspect ratios of master image
        and mosaic with number of rows and columns as specified.
        '''
        (nr_row, nr_col) = option
        return ((nr_row / nr_col) * asp_tile - asp_mast)**2
    #find option with minimal deviation.
    option_best = min(options, key=deviation)
    return option_best

def _image_resizer(image, target_height, target_width):
    '''Resizes image to specified height and width.
    --------
    Keyword arguments:
    image         --- array-like image
    target_height --- desired height
    target_width  --- desired width
    --------
    returns resized image
    '''
    #resize image
    output = cv2.resize(image, (target_width, target_height))
    #design kernel for smoothening resized image
    im_width = image.shape[1]
    fac = target_width // (2 * im_width)
    kern_size = 2 * fac + 1
    kernel = np.ones((kern_size, kern_size), np.float32) / kern_size**2
    #smoothen image by filtering
    output = cv2.filter2D(output, -1, kernel)
    return output

def _extract_data(image, nr_row, nr_col):
    '''Partition image into nr_row rows and nr_col columns of tiles and return
    list with average colour values of each part, where
    output[row_index * nr_col + col_index] = average in tile in
    in (row,column) = (row_index, col_index).
    --------
    Keyword arguments:
    image  --- array-like image
    nr_row --- number of rows
    nr_col --- number of columns
    --------
    returns list with average colour values of parts.
    '''
    #get image height and width
    im_height = image.shape[0]
    im_width = image.shape[1]
    #built output recursively
    output = []
    for row_ind in range(nr_row):          #row_ind loops over rows
        for col_ind in range(nr_col):      #col_ind loops over columns
            #define limits of part
            top_lim = int(row_ind * im_height / nr_row)
            bot_lim = int((row_ind + 1) * im_height / nr_row)
            lef_lim = int(col_ind * im_width / nr_col)
            rig_lim = int((col_ind+1) * im_width / nr_col)
            #get part
            sub_im = image[top_lim : bot_lim, lef_lim : rig_lim]
            #compute average colour value of part
            sub_im_average = sub_im.mean(axis=1).mean(axis=0)
            #add average to output
            output.append(sub_im_average)
    return output

def _optimal_assignment(mas_im_data):
    '''Given data of master image, return optimal assignment of library
    images to tiles in mosaic.
    '''
    #try loading library data
    try:
        lib_data = pd.read_csv(str(LIBRARY_FOLDER / "averages.csv"), index_col=0)
    except:
        raise FileNotFoundError("Library missing or corrupt, please rebuild it.")
    #save library and master image data as numpy arrays
    lib_data = np.array(lib_data)
    mas_data = np.array(mas_im_data)
    nr_mas_entries = len(mas_im_data)   #number of tiles in mosaic
    nr_lib_entries = len(lib_data)      #number of images in library
    print("Construction of cost matrix initialised.")
    def point_dist(mas_index, lib_index):
        '''return l^2 difference between average colour values of
        part mas_index of master image and image lib_index of library.
        '''
        dif = mas_data[int(mas_index)]-lib_data[int(lib_index)]
        return np.linalg.norm(dif/256)
    #initialise cost matrix
    cost_matrix = np.zeros((nr_mas_entries, nr_lib_entries))
    #compute cost matrix entry-wise
    for mas_index in range(nr_mas_entries):
        for lib_index in range(nr_lib_entries):
            cost_matrix[mas_index][lib_index] = point_dist(mas_index, lib_index)
    print("Construction of cost matrix completed.")
    print("Finding optimal assignment initalised.")
    if nr_mas_entries > 700:
        print("This can take a little while.")
    #apply Hungarian method to obtain optimal assignment
    _, col_ind = linear_sum_assignment(cost_matrix)
    print("Finding optimal assignment completed.")
    #compute total cost of assignment
    total = 0
    for k in range(nr_mas_entries):
        total += point_dist(k, col_ind[k])**2
    cost = (total / nr_mas_entries)**(0.5)
    print("Total assignment cost: ", cost)
    assignment = [[k, col_ind[k]] for k in range(nr_mas_entries)]
    return assignment

#PRINT_MOSAIC HELPER FUNCTIONS
def _weights_check(weights):
    '''Check whether list or tuple consists only of integers
    with sum equal to 100 and first entry is nonzero.
    '''
    if not all(isinstance(n, int) for n in weights):
        return False
    return sum(weights) == 100 and weights[0]

def _mosaic_namer(weights):
    '''Return name which uniquely encodes weights used for building mosaic.'''
    if weights[0] == 100:
        output = "mosaic.jpg"
    else:
        output = "mosaic"
        for weight in weights:
            output += "_" + str(weight)
        output += ".jpg"
    return output

#IF RUNNING SCRIPT
if __name__ == "__main__":
    MOS = MosaicProject()
    #build library
    MOS.build_library()
    #process master image
    MOS.process_master(max_im=0)
    #build mosaic
    MOS.print_mosaic(tuning=None)
