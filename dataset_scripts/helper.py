import logging
import os

import boto3

DATALAKE_NAME = 'hubble-datalake1'

PROFILEIMG_FOLDER = 'images/profile_pics'

## Logger --------------------------------------------------------------------------------------
try:
    import colorlog
    HAVE_COLORLOG = True
except ImportError:
    HAVE_COLORLOG = False

def create_logger():
    """
        Setup the logging environment
    """
    log = logging.getLogger()  # root logger
    log.setLevel(logging.INFO)
    format_str = '%(asctime)s - %(levelname)-8s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    if HAVE_COLORLOG and os.isatty(2):
        cformat = '%(log_color)s' + format_str
        colors = {'DEBUG': 'reset',
                  'INFO': 'reset',
                  'WARNING': 'bold_yellow',
                  'ERROR': 'bold_red',
                  'CRITICAL': 'bold_red'}
        formatter = colorlog.ColoredFormatter(cformat, date_format,
                                              log_colors=colors)
    else:
        formatter = logging.Formatter(format_str, date_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    return logging.getLogger(__name__) 

logger = create_logger()

# Class Methods (Should encapsulate all s3 and rds methods to make the work easier to understand) ----------------------------------
class S3Helper:
    """
    Encapsulate all needed functions here
    """
    def __init__(self):
        self.client = boto3.client("s3")
        self.s3 = boto3.resource('s3')

    ## ==========================================================================================================
    ## General Functions

    def create_folder(self, path, location):
        """
        The idea of this function is to encapsulate all kinds of folder creation in s3
        1. Create bucket (if bucket does not exist)
        2. Create folders
        """
        path_arr = path.rstrip("/").split("/")
        # If the path given is only the bucket name.
        if len(path_arr) == 1:
            return self._check_bucket(location)
        parent = path_arr[0]
        self._check_bucket(parent)
        bucket = self.s3.Bucket(parent)
        status = bucket.put_object(Key="/".join(path_arr[1:]) + "/")
        return status

    def upload(self, filename, bucketname, s3_pathname):
        self.s3.meta.client.upload_file(filename, bucketname, s3_pathname)

    def download(self, filename, bucketname, local_pathname):
        self.client.download_file(bucketname, filename, local_pathname)

    def check_folder_in_bucket(self, bucketname, keypath):
        """
        Check with a bucket 
        """
        response = self.client.list_objects_v2(
            Bucket=bucketname,
            Prefix=keypath
            )
        # logger.info(response.get('Contents', []))
        if len(response.get('Contents', [])):
            return True
        else:
            return False

    def move_file(self, oldfullkeypath, newfullkeypath):
        """
        This function moves the same file into a different keypath in different/same buckets
        """
        oldbucketname = oldfullkeypath.split('/')[0]
        newbucketname = newfullkeypath.split('/')[0]

        oldkeypath = '/'.join(oldfullkeypath.split('/')[1:])
        newkeypath = '/'.join(newfullkeypath.split('/')[1:])

        copy_source = {
            'Bucket': oldbucketname,
            'Key': oldkeypath
        }
        self.s3.meta.client.copy(copy_source, newbucketname, newkeypath)

        if self.check_folder_in_bucket(newbucketname, newkeypath):
            return True
        else:
            return False
    
    def list_objects(self, bucketname, folderpath):
        """
        List all the keypaths to an object in a certain bucket of a certain folderpath
        """
        response = self.client.list_objects_v2(
            Bucket=bucketname,
            Prefix=folderpath
            )
        contents = response.get('Contents', [])
        keypaths = list(map(lambda x: x['Key'], contents))

        return keypaths

    ## ==========================================================================================================
    ## Sub Functions

    def _check_bucket(self, location):
        # Check if data lake exists
        bucketlist = self.client.list_buckets()['Buckets']
        print(bucketlist)
        bucketnames = list(map(lambda x: x['Name'], bucketlist))
        print(bucketnames)
        datalake = list(filter(lambda x: x.lower() ==
                               DATALAKE_NAME, bucketnames))
        print(datalake)

        # We can create a datalake for each region as well, but for now we don't need to do that yet.
        # datalake_name = DATALAKE_NAME + "-" + location
        if not datalake:
            # Create a bucket based on given region
            self.client.create_bucket(Bucket=DATALAKE_NAME)
        return True
