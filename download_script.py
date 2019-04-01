# importing google_images_download module 
from google_images_download import google_images_download  
  
# creating object 
response = google_images_download.googleimagesdownload()

query="right eye"

def downloadimages(query):
    arguments={"keywords":query,
                "format":"jpg",
                "limit":100,
                "print_urls":True,
                "size":"medium",
                "aspect_ratio":"panoramic"}
    try:
        response.download(arguments)

    except FileNotFoundError:
        arguments = {"keywords": query,
                     "format": "jpg",
                     "limit":100,
                     "print_urls":True,
                     "size": "medium"}
     # Providing arguments for the searched query 
        try: 
            # Downloading the photos based 
            # on the given arguments 
            response.download(arguments)  
        except: 
            pass
downloadimages(query)
