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









class MosaicProject:
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
            print("(2) start a new one and clear the old one?")
            choice = "0"
            while choice not in ["1", "2"]:
                choice = input("Please answer with 1 or 2: ")
            if choice == "1":
                try:
                    library_log = open(LIBRARY_FOLDER / "log.txt", "r")
                    values_list = library_log.readlines()
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

    def set_height(self, height):
        '''height setter'''
        self.height = height
        return None

    def set_width(self, width):
        '''height setter'''
        self.width = width
        return None

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
        log = open(LIBRARY_FOLDER / "log.txt", "w")
        log.writelines([str(height) + "\n", str(width) + "\n", str(image_counter) + "\n"])
        log.close()
        print("Building of image library completed.")
        return None

    def master_processor(self, max_im=0):
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
                clear_mas_data()
        #load master image
        mas_im = load_mas_im()
        print("Master image processing initiated.")
        #get aspect ratio of master image
        mas_dim = mas_im.shape
        asp_mast = mas_dim[0] / mas_dim[1]
        #get aspect ratio of tile images
        height = self.get_height()
        width = self.get_width()
        asp_tile = height / width
        #get number of images in library from log file
        log = open(LIBRARY_FOLDER / "log.txt", "r")
        nr_lib = int(log.readlines()[2])
        log.close()
        #set maximum number of image tiles in mosaic
        if max_im > 0:
            nr_im = max(nr_lib, max_im)
        else:
            nr_im = nr_lib
        #find optimal numbers of rows and columns in mosaic
        (nr_row, nr_col) = optimal(nr_im, asp_tile, asp_mast)
        print("Your mosaic will consist of "+str(nr_row)+\
              " rows and "+str(nr_col)+" columns of images.")
        #save optimal numbers of rows and columns in log file
        log = open(LIBRARY_FOLDER / "log.txt", "w")
        log.writelines([str(height) + "\n", str(width) + "\n",
                        str(nr_lib) + "\n", str(nr_row) + "\n",
                        str(nr_col) + "\n"])
        log.close()
        #resize master image to shape of to be built mosaic
        mas_im_resized = image_resizer(mas_im, nr_row * height, nr_col * width)
        #save resized master image in mas_data subfolder of library
        cv2.imwrite(str(MAS_DATA_FOLDER / "mas_res.jpg"), mas_im_resized)
        #extract average colour values of sub_images of master_image
        mas_im_data = extract_data(mas_im_resized, nr_row, nr_col)
        del mas_im_resized
        #compute optimal assignment of library images to mosaic tiles
        assignment = optimal_assignment(mas_im_data)
        #save optimal assignment as csv file in mas data subfolder
        pd.DataFrame(assignment).to_csv(str(MAS_DATA_FOLDER / "assignment.csv"))
        print("Master image processing completed.")
        return None

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



def load_mas_im():
    '''Load master image from master folder and return it.
    ---------
    If there are several files in master folder, ask user to choose one.
    If there is no image file in master folder, raise FileNotFoundError.
    '''
    #get all files in master folder.
    files = os.listdir(MASTER_FOLDER)
    #remove master folder readme from list
    try:
        files.remove("master folder readme.md")
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
        for index in range(len(files)):
            print("("+str(index+1)+") "+files[index])
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

def image_resizer(image, target_height, target_width):
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

def extract_data(image, nr_row, nr_col):
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

def optimal_assignment(mas_im_data):
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
