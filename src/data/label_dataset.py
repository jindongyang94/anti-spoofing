"""
The aim of this script is to be able to detect anti spoofing using an existing training model to help set up the initial dataset,
and then curate from there again.
"""
import os
import shutil
from glob import glob
from os.path import abspath, dirname, join

import cv2
import fire
from tqdm import tqdm

from src.modules.aws_helper import logger
from src.modules.config import (external_data_location, models_location,
                                processed_data_location, working_directory)


def show_images(location):
    """
    Show images in a certain folder location one by one. 
    Yup that's it.
    """
    files = glob("%s/*.jpg" % location)
    logger.info("There are %s files in location: %s" % (len(files), location))

    for filename in files:
        img = cv2.imread(filename)
        cv2.imshow("Profile Pic", img)
        key = cv2.waitKey()
        cv2.destroyAllWindows()
        # if ESC is pressed, exit loop
        if key == 27:
            break
    return


def label_images(old_location, cat1_location, cat2_location):
    """
    Go into unsorted, and move images into either real or spoof
    """
    files = glob("%s/*.jpg" % old_location)
    while files:
        logger.info("There are %s files in location: %s" %
                    (len(files), old_location))

        bar = tqdm(files, dynamic_ncols=True, desc='Bar desc', leave=True)
        for filename in files:
            img = cv2.imread(filename)
            cv2.imshow("Profile Pic", img)
            key = cv2.waitKey()
            cv2.destroyAllWindows()

            # if ESC is pressed, exit loop
            if key == 27:
                return "Labelling Terminated."

            fname = str(filename.split('/')[-1])
            # If 1 is pressed, it goes into category 1 --> real images
            if key == 49:
                new_filename = "%s/%s" % (cat1_location, fname)
                shutil.move(filename, cat1_location)

                catname = str(cat1_location.split('/')[-1])
                bar.set_description("%s moved to %s Folder" % (fname, catname))

            # If 2 is pressed, it goes into category 2 --> spoof images
            if key == 50:
                new_filename = "%s/%s" % (cat2_location, fname)
                shutil.move(filename, cat2_location)

                catname = str(cat2_location.split('/')[-1])
                bar.set_description("%s moved to %s Folder" % (fname, catname))

            bar.update(1)
            bar.refresh()

        files = glob("%s/*.jpg" % old_location)

    return "All Images Moved."


def categorise_images_based_on_model(old_location, cat1_location, cat2_location):
    """
    This function is to speed up the labelling process as well as testing existing models on the internet.
    It might be better to do this and see if we can catch any positive examples from our database.
    """

    pass

## ------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
# Sub Functions
## ------------------------------------------------------------------------------------------------------------------------------------------------------------- ##


def find_datafolder():
    """
    Return parent folder, unsorted folder, real folder and spoof folder
    """
    original_folder = join(abspath('.'), 'data')

    unsorted_folder = join(original_folder, 'unsorted')
    category1_folder = join(original_folder, 'real')
    category2_folder = join(original_folder, 'spoof')

    return original_folder, unsorted_folder, category1_folder, category2_folder


if __name__ == "__main__":
    original_folder, unsorted_folder, category1_folder, category2_folder = find_datafolder()
    logger.info(label_images(unsorted_folder,
                             category1_folder, category2_folder))
