from requests_toolbelt import MultipartEncoder
import requests
import json
from pathlib import Path

#pdf workflow: upload file(user), upload it to storage?, fetch it from supabase storage and parse the images
def image_parser(pdf_url : str, img_extractor_url: str, img_extractor_api_key: str) -> dict:

    if not img_extractor_api_key or not img_extractor_url:
        raise ValueError("PDF REST API KEY OR URL IS NOT BEING PROVIDED")
    
    file_name = Path(pdf_url).stem
    
    mp_encoder_extractedImages = MultipartEncoder(
        fields={
            'file': (f'{file_name}.pdf', open(pdf_url, 'rb'), 'output/pdf'),
            'pages': '1-40',
            'output': f'{file_name}_extracted_images'
        }
    )

    headers = {
        'Accept': 'application/json',
        'Content-Type': mp_encoder_extractedImages.content_type,
        'Api-Key': img_extractor_api_key 
    }

    response = requests.post(
        img_extractor_url, 
        data=mp_encoder_extractedImages, 
        headers=headers
    )

    print("Response status code: " + str(response.status_code))

    if response.ok:
        response_json = response.json()
        print(json.dumps(response_json, indent=2))
        
        extracted_images = response_json.get('outputUrl', [])
        if (isinstance(extracted_images, str)):
            extracted_images = [extracted_images]
        return {
            "success": True,
            "images": extracted_images,
            "api_response": response_json
        }
    else:
        error_msg = f"API Error {response.status_code}: {response.text}"
        print(response.text)
        return {
            "success": False,
            "images": error_msg,
            "status_code": response.status_code
        }