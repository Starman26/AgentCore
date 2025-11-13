import os
from supabase import create_client, Client, StorageException
# Code for testing with local env variables
# from dotenv import load_dotenv
# load_dotenv()

SB: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

def printBuckets():
    try:
        print(SB.storage.list_buckets())
    except Exception as e:
        print(e)

def uploadFile(localPath, pathInBucket,bucketName):
    try:
        with open(localPath,"rb") as f:
            response = SB.storage.from_(bucketName).upload(file=f,path=pathInBucket,file_options={"cache-control": "3600", "upsert": "false"})
            print(response)
    except StorageException as e:
        print(e)
        
def deleteFile(pathInBucket,bucketName):
    try:
        SB.storage.from_(bucketName).remove([pathInBucket])
    except StorageException as e:
        print(e)
