import os
from pprint import pformat, pprint

import boto3
import fire
from tqdm import tqdm, trange

from src.modules.aws_helper import DATALAKE_NAME, PROFILEIMG_FOLDER, S3Helper, logger


"""
The check in / check out pictures are placed in:
Company --> hubble/attendances/attendance/(check_in_photo or check_out_photo)/(numeric id)/(check_in or check_out_photo.jpg) 
or (thumb_check_in or out_photo.jpg

For now, we can ignore the thumbnails as we want to have higher resolution pictures. 
"""

## ------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
## AWS Download
## ------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
def migrate_pictures():
    """
    This function is the main function to migrate all buckets has check in and check out photos to a centralized location denoted 
    in the helper script: PROFILEIMG_FOLDER
    """
    s3 = S3Helper()
    
    buckets_dict = list_buckets()

    # Print Bucket List
    for line in pformat(buckets_dict).split('\n'):
        logger.info(line)

    for key, values in buckets_dict.items():
        logger.info('Migrating for %s pictures' % key)

        bar = tqdm(values, dynamic_ncols=True, desc='Bar desc', leave=True) 
        for keypaths in bar:

            keypath = list(filter(lambda x: str(x.split('/')[-1]) == key, keypaths))
            if keypath and len(keypath) == 1:
                bucketname = keypath[0]
            else:
                logger.error("Keypath Filter is wrong. Keypath: %s" % keypath)
                break

            oldkeypaths = s3.list_objects(bucketname, keypath)
            # logger.info(oldkeypaths)

            for oldkeypath in oldkeypaths:
                # Remove all thumb photos
                if 'thumb' in str(oldkeypath.split('/')[-1]):
                    continue
                full_oldkeypath = bucketname + '/' + oldkeypath
                bar.set_description(full_oldkeypath)
                # logger.info(full_oldkeypath)

                newimagename = bucketname + '_' + oldkeypath.split('/')[-2] + '_' + key + '.' + oldkeypath.split('.')[-1]
                full_newkeypath = DATALAKE_NAME + '/' + PROFILEIMG_FOLDER + '/' + newimagename

                success = s3.move_file(full_oldkeypath, full_newkeypath)
                if not success:
                    logger.info("Unsuccessful Transfer")
            bar.refresh()
    return

def list_buckets():
    """
    Simply a wrapper function to run it specifically for this case.
    """
    check_in_key = "hubble/attendances/attendance/check_in_photo"
    check_out_key = "hubble/attendances/attendance/check_out_photo"
    paths = [check_in_key, check_out_key]
    return __list_buckets(paths)

def download_images():
    """
    Simply a wrapper function to call what is already defined.
    """
    local_folderpath = '/Users/jindongyang/Documents/repos/hubble/hubble_projects/hubble_spoofing_detection/data/external/attendance_photos'
    logger.info(__download_images(DATALAKE_NAME, PROFILEIMG_FOLDER, local_folderpath, start_index=0, limit=500))

## ------------------------------------------------------------------------------------------------------------------------------------------------------
## Sub Functions
## ------------------------------------------------------------------------------------------------------------------------------------------------------
def __list_buckets(paths):
    """
    This function is to check every bucket if there is check_in / check_out folder present for us to consider
    so that we can use it to migrate the folders.
    """
    
    keys = list(map(lambda x: str(x.split('/')[-1]), paths))
    logger.info(keys)

    s3 = S3Helper()
    # List all the buckets in the system
    buckets = s3.client.list_buckets()['Buckets']
    bucketnames = list(map(lambda x: x['Name'], buckets))

    bucket_dict = {x : [] for x in keys}
    logger.info("Empty Dict: %s" % bucket_dict)

    bar = tqdm(bucketnames, dynamic_ncols=True, desc='Bar desc', leave=True)
    for bucketname in bar:
        bar.set_description('Accessing %s' % bucketname)
        
        for keypath in paths:
            if s3.check_folder_in_bucket(bucketname, keypath):
                # logger.info('%s has Check In Photos.' % bucketname)
                key = str(keypath.split('/')[-1])
                bucket_dict[key].append(bucketname)
        
        bar.refresh()

    return bucket_dict

def __download_images(bucketname, folderpath, location, start_index = 0, limit=100):
    """
    This function is simply gonna download the first 100 images (maybe next time done in random?)
    from the bucket to this repo under 'dataset/unsorted/'
    """
    s3 = S3Helper()
    keypaths = s3.list_objects(bucketname, folderpath)

    logger.info('Starting Downloading Images...')
    index = start_index
    bar = trange(limit, dynamic_ncols=True, desc='Bar desc', leave=True)
    for i in bar:
        keypath = keypaths[index]
        local_imagepath = location + '/' + str(keypath.split('/')[-1])
        bar.set_description(keypath)

        s3.download(keypath, bucketname, local_imagepath)
        bar.refresh()
        index += 1

    files = []
    for r, d, f in os.walk(folderpath):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))
    
    statement = "There are %s images in %s" % (len(files), location)

    return statement



if __name__ == "__main__":
    # migrate_pictures()
    # local_folderpath = '/Users/jindongyang/Documents/repos/hubble/hubble_projects/hubble_picinpic/dataset/unsorted'
    # logger.info(download_images(DATALAKE_NAME, PROFILEIMG_FOLDER, local_folderpath, start_index=100, limit=500))

    # Fire is a library for automatically generating command line interfaces (CLIs) from absolutely any Python object
    fire.Fire()
