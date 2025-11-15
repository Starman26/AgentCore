from requests_toolbelt import MultipartEncoder
import requests
import json

#TODO: Save images in the ouput dir
#pdf workflow: upload file(user), upload it to storage?, fetch it from supabase storage and parse the images

extracted_images_endpoint_url = 'https://api.pdfrest.com/extracted-images'

mp_encoder_extractedImages = MultipartEncoder(
    fields={
        'file': ('manual1.pdf', open('resources/manual1.pdf', 'rb'), 'output/pdf'),
        'pages': '1-100',
    }
)

headers = {
    'Accept': 'application/json',
    'Content-Type': mp_encoder_extractedImages.content_type,
    'Api-Key': 'ac6873ae-6fd8-43ab-ae8d-f53477aa58e3' # place your api key here
}

response = requests.post(
    extracted_images_endpoint_url, 
    data=mp_encoder_extractedImages, 
    headers=headers
)

print("Response status code: " + str(response.status_code))

if response.ok:
    response_json = response.json()
    print(json.dumps(response_json, indent=2))
else:
    print(response.text)