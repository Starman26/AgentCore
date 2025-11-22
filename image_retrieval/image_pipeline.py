from dotenv import load_dotenv
from image_parser import image_parser
from sbFileUploadAssistant import uploadFile    
from pathlib import Path
import requests
import os
import base64
from openai import OpenAI
from supabase import Client, create_client

load_dotenv()

class ImagesPipeline:
    """Pipeline to process pdf documents and manage image extraction"""
    _general_api_url = "https://api.pdfrest.com"
    _openAi_client = OpenAI()
    _DB_client : Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    
    def __init__(self, output_dir: str = "output/extracted_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = os.environ.get("PDF_REST_API_KEY")
        
        if not self.api_key:
            raise ValueError("IMAGE HANDLER APIKEY WAS NOT PROVIDED")
        
    def process_pdf(self, pdf_url: str):
        pdf_name = Path(pdf_url).stem
        self.image_extractor_url = f'{self._general_api_url}/extracted-images'        
        try:
            #upload complete file/manual to supabase storage 
            uploadFile(localPath=pdf_url,
                    pathInBucket=f"{pdf_name}.pdf",
                    bucketName="manuals"
            )
            
            #perform image extraction from file
            extraction_res = image_parser(
                pdf_url=pdf_url,
                img_extractor_url=self.image_extractor_url,
                img_extractor_api_key=self.api_key
            )
            #get img base 64 info, for each img_id in extraction res
            img_ids = extraction_res.get("api_response")["outputId"]
            img_urls = extraction_res.get("api_response")["outputUrl"]
            for i, (img_id, img_url) in enumerate(zip(img_ids, img_urls)):
                #Get base 64 format for each img file
                base_64_img = self.get_img_file_from_img_id(img_id, "file")

                #use base64 to get img context with llm (gemini pro vision with gemini API could be used here)
                #you could send requests with multiple images at once, maybe use a dictionary to 
                # localize each image_id and it's base64
                img_content_text = self.get_image_context(base_64_img)
                
                image_data = {
                        "img_id": i + 1, 
                        "pdf_name": pdf_name,
                        "page_number": i + 1,
                        "img_path": f"{pdf_name}/img{i + 1}",
                        "img_context": img_content_text,         
                        "img_url": img_url,         
                }
                
                #upload img supabase storage link and context to DB
                new_img = self.storage_img_(image_data)
                print(f"SALVADO DE IMG {i + 1}: {new_img}")
            
            #use the same base64 to let the model show the imgs4
            #u save the img link and the context of it, when the model needs it, u use a function to get the actual base_64 format of the img from the link
            #saving the hole base_64 text is completely inefficient
                            
            result = {
                "success": True,
                "pdf_url": pdf_url,
                "imgs_ids": extraction_res.get("api_response")["outputId"],
                "metadata": {
                    "total_images": len(extraction_res.get("images", [])),
                    "pdf_filename": pdf_name
                }
            }
            return result
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "pdf_url": pdf_url
            }
     
    def get_img_file_from_img_id(self, img_id : str, format : str) -> str:
        """Gets the actual resource from resource id, the id's come from image extraction

        Args:
            img_id (str): image id from image extraction 
            format (str): desired format to download the resource as (file || json)
        Returns:
            response.content || response.text (str):  with base64 file format
        """
        resource_url = f"{self._general_api_url}/resource/{img_id}?format={format}"
        print(f"Sending GET request to {self._general_api_url}/resource/{img_id}?format={format} endpoint... ")
        response = requests.get(resource_url)
            
        if response.ok and format == 'file':
            binary_byte_response_file = response.content
            base_64_response_file = base64.b64encode(binary_byte_response_file).decode('utf-8')
            print(base_64_response_file[:50])
            return base_64_response_file
        else:
            msg = response.text
            print("MENSAJE: ", msg)
            return 
    
    def get_image_context(self, img_base64 : str) :
        """Function to get an image context from image base64 format
        Reference: https://platform.openai.com/docs/guides/images-vision?api-mode=responses&format=base64-encoded
        Args:
            img_base64 (str): base64 img content

        Returns:
            _type_: img description from llm
        """
        response = self._openAi_client.responses.create(
                model="gpt-4.1",
                input=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "input_text", "text": "what's in this image?, keep the answer short and concise, ignore the pdfRest watermark" },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{img_base64}",
                            },
                        ],
                    }
                ],
            )
        
        print("LLM RESPONSE -> ", response.output_text)
        return response.output_text
    
    def storage_img_(self, image_data : dict):
        """Store image data in Supabase data base

        Args:
            image_data (dict): objetct containing image matedata
        """
        try:
            result = self._DB_client.table("manual_images").insert({
                "pdf_id": image_data["pdf_name"],
                "page": image_data["page_number"],
                "path": image_data["img_path"],
                "caption": image_data["img_context"],
                "ocr_text": "",
                "tags": ["manual", "robotic", "technical"],
                "phash" : image_data["img_url"]
            }).execute()
            
            return result.data
            
        except Exception as e:
            print(f"Could not store image context: {e}")
            return f"upload_error_{image_data["img_id"]}"

    
        

           
if __name__ == "__main__":   
    imageTest = ImagesPipeline()
    res = imageTest.process_pdf("C:/Users/Daniela Herrera/Desktop/FrED/AgentCore/image_retrieval/resources/manual1.pdf")

    if res["success"]:
        images = res["imgs_ids"]
        print("RESULTSSSS->", images)

    else:
        print("ERROR: ", res["error"])
        

        
        