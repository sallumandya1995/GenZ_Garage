import os
import json
import pathlib
from typing import Union
import base64
from io import BytesIO
import subprocess 
import time
import functools

import pandas as pd
import vertexai
from vertexai.language_models import TextGenerationModel
import torch
import transformers
from PIL import Image
import requests
from IPython.display import display
from serpapi import GoogleSearch
import requests
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,StableDiffusionInpaintPipeline
import gradio as gr

import os
SPECS = {
    'GrandVitara': {
        "ARAI Mileage": "27.97 kmpl",
        "Fuel Type": "Petrol",
        "Engine Displacement(cc)": 1490,
        "Number of Cylinders": 3,
        "Max Power bhp@rpm": "91.18bhp@5500rpm",
        "Seating Capacity": 5,
        "Transmission Type": "Automatic",
        "Fuel Tank Capacity": 45.0,
        "Body Type": "SUV"
    },
    'Nexon':{
        'ARAI Mileage': '24.07 kmpl',
        'Fuel Type': 'Diesel',
        'Engine Displacement(cc)': 1497,
        'Number of Cylinders': 4,
        'Max Power bhp@rpm': '113.42bhp@3750rpm',
        'Seating Capacity': 5,
        "Transmission Type": "Automatic",
        "Fuel Tank Capacity": 45.0,
        "Body Type": "SUV"
    },
    'Porsche911':{
        'WLTP Mileage': '10.64 kmpl',
        'Fuel Type': 'Petrol',
        'Engine Displacement(cc)': 3996,
        'Number of Cylinders': 6,
        'Max Power bhp@rpm': '379.50bhp@6500rpm',
        'Seating Capacity': 4,
        "Transmission Type": "Automatic",
        "Fuel Tank Capacity": 64.0,
        "Body Type": "Coupe"
    },
    'Creta':{
        'City Mileage': '18.0 kmpl',
        'Fuel Type': 'Diesel',
        'Engine Displacement(cc)': 1493,
        'Number of Cylinders': 4,
        'Max Power bhp@rpm': '113.45bhp@4000rpm',
        'Seating Capacity': 5,
         "Transmission Type": "Automatic",
        "Fuel Tank Capacity": 50.0,
        "Body Type": "SUV"
    },
    'Fronx':{
        'ARAI Mileage': '20.01 kmpl',
        'Fuel Type': 'Petrol',
        'Engine Displacement(cc)': 998,
        'Number of Cylinders': 3,
        'Max Power bhp@rpm': '98.69bhp@5500rpm',
        'Seating Capacity': 5,
         "Transmission Type": "Automatic",
        "Fuel Tank Capacity": 37.0,
        "Body Type": "SUV"
    }

}

GALLERY_IMAGES = [
              'cars/Creta.png',
             'cars/Fronx.jpg',
              'cars/GrandVitara.jpg',
              'cars/Nexon.png',
              'cars/Porsche911.jpg'
          ]

text2img_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                               torch_dtype=torch.float16)
text2img_pipe = text2img_pipe.to("cuda")

print('Text To Image')

inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
)
inpaint_pipe = inpaint_pipe.to("cuda")

vertexai.init(project="gen-team-phoeniks", location="us-central1")
model = TextGenerationModel.from_pretrained("text-bison@001")



def timer(func: callable) -> callable:
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        t1 = time.perf_counter()
        res = func(*args,**kwargs)
        t2 = time.perf_counter() - t1
        print(f'{func.__name__} took {t2:.2f} seconds(s)')
        return res
    return wrapper

@timer
def get_auth_token():
    op = subprocess.run("gcloud auth print-access-token".split(), capture_output = True)
    return op.stdout.decode().strip()

@timer
def gpt(user_query: str) -> str:
    parameters = {
    "temperature": 0.5,
    "max_output_tokens": 1024,
    "top_p": 0.8,
    "top_k": 40
    }
    response = model.predict(
    user_query,
    **parameters
    )
    return response.text

@timer
def customize_image(evt:gr.SelectData)-> tuple[Image, pd.core.frame.DataFrame]:
  image = GALLERY_IMAGES[evt.index]
  print(image)
  car_specs = SPECS[pathlib.Path(image).stem]

  df = pd.DataFrame({'Key Specifications': car_specs.keys(),' ':car_specs.values()})
  image = Image.open(image).resize((768, 768))
  display(image)
  return image, df

@timer
def  customize_image_for_search (evt:gr.SelectData) -> Image:
  image, _ = customize_image(evt)
  return image

@timer
def sd_txt_to_image(prompt: str) -> Image:
  return text2img_pipe(prompt).images[0]

@timer
def base64_to_image(image_str):
    image = Image.open(BytesIO(base64.b64decode(image_str)))
    return image

@timer
def txt_to_image(prompt: str) -> Image:
    print(prompt)
    url = "https://us-central1-aiplatform.googleapis.com/v1/projects/gen-team-phoeniks/locations/us-central1/publishers/google/models/imagegeneration:predict"
    data = {
      "instances": [
        {
          "prompt": prompt
        }
      ],
      "parameters": {
        "sampleCount": 1
      }
    }
    auth_token = get_auth_token()
    res = requests.post(url, headers = {'Authorization' : f'Bearer {auth_token}',
                                        'Content-Type': 'application/json; charset=utf-8'},
                       json = data)
    
    if res:
        print(res)
        img = base64_to_image(res.json()['predictions'][0]['bytesBase64Encoded'])
        print('Vision AI')
        display(img)
    else:
        print('txt_to_image')
        print(res.content)
        img = sd_txt_to_image(prompt)
    return img

    
    

@timer
def inpainting(prompt: str,
               images: dict[str,Image]) -> Image:
    init_image = images["image"]
    print("Init Image")
    display(init_image)
    print("Init Image")
    mask_image = images['mask']
    display(mask_image)
    mask_image.save('mask.png')
    return inpaint_pipe(prompt = prompt,
                      image = init_image,
                      mask_image = mask_image,
                      guidance_scale = 7.5).images[0]

@timer
def image_search(image: Image) -> Image:
  display(image)
  image.save('cropped.png')
  url = upload_img_url('cropped.png')

  for i in range(1,15):
    try:
      serpapi_op = use_serpapi(url,i)
      print(i)
      break
    except Exception as e:
      print(e)
  return serpapi_op


@timer
def use_serpapi(image_url:str,number:int=0) -> pd.core.frame.DataFrame:

  api_key = os.environ[f'SERPAPI_KEY{number}']
  print(f'SERPAPI_KEY{number}')
  params = {
    "engine": "google_reverse_image",
    "image_url": image_url,
    "api_key": os.environ[f'SERPAPI_KEY{number}']
  }

  search = GoogleSearch(params)
  results = search.get_dict()
  print(f'Results SERPAPI:{results}')
  inline_images = results["inline_images"][:5]




  shopping_results = []
  shopping_results+=[result_dct for result in inline_images for result_dct in shopping_url(result['title'],api_key)]
  car_df = pd.DataFrame(shopping_results)
  car_df.to_csv('cars.csv',index=False)
  return car_df

@timer
def shopping_url(product_title:str,api_key:str) -> list[dict[str,str]]:
  params = {
    "engine": "google_shopping",
    "q": product_title,
    "api_key": api_key
  }

  search = GoogleSearch(params)
  results = search.get_dict()
  shopping_results = results["shopping_results"]
  return [{'title': result_dict['title'],
                      'price':result_dict['price'], 'link': result_dict['link'],
                      }
                     for result_dict in shopping_results ]

@timer
def upload_img_url(image_path:str) -> str:
    API_ENDPOINT = "https://api.imgbb.com/1/upload"
    API_KEY = "6a93e6e238d91262a739c1b1d1af64d8"

    with open(image_path, "rb") as image:
        image_data_ = base64.b64encode(image.read())
        image_data = image_data_.decode('utf-8')

    payload = {
        "key": API_KEY,
        "image": image_data
    }
    # Send the API request
    response = requests.post(API_ENDPOINT, payload)
    print(response)
    # Get the generated link from the API response
    response_json = response.json() #
    print("Response json:", response_json)
    image_url = response_json["data"]["url"]
    print("Generated link:", image_url)
    return image_url

@timer
def get_car_prompt(body_type:str,car_color:str,
                    roof_type:str, use_car_custom_description:bool,car_description:str):
    
    if not use_car_custom_description:
        
        car_description = gpt(f"""Generate description of automobile from the following desciptors in less than 77 tokens. 
                   Take inspiration from inspiration from "fast and furious", "Need for speed" movies, do not use the movie name.:

                    Request: silver sedan with sunroof roof.
                    Response: A photograph of taken from front of a silver sedan with a sunroof in futuristic car garage.

                    ============================================================

                    Request: cyan suv with a solid roof. 
                    Response:  A photograph taken from front of a cyan SUV with a solid roof in futuristic city at night time, neon lights.

                    ============================================================

                   Request: black coupe with a solid roof.
                   Response: A photograph taken from front of a  black coupe with a solid roof in futuristic city at night time, neon lights.

                   ============================================================

                  Request: purple sports car with a solid roof.
                  Response: A photograph taken from front of a  purple sports car with a solid roof surrounded by mountains, snowing.

                  ============================================================ 
                  Request: {car_color} {body_type} with a {roof_type} roof.
                  Response: 
                  """).strip()
        
        print(f'{car_description = }')
    img = txt_to_image(car_description) 
    return car_description,img.resize((512,512))

@timer
def get_prompts(evt:gr.SelectData,search_text:str = 'Car logo') -> list[str]:
    print('Get Prompt')
    print(search_text)
    response = requests.get('https://lexica.art/api/v1/search?',params={'q':f'{search_text} logo'})
    print(f'{response = }')
    images = response.json()["images"]
    return pd.DataFrame([dct["prompt"] for dct in images
          if len(dct["prompt"].strip().split()) > 5][:50],columns = ["Logo Prompts"])

@timer
def get_example_tag_lines_and_names(car_description: str) -> tuple[pd.core.frame.DataFrame]:
    response = gpt("""
    For following car description generate 10 taglines and 10 car names take inspiration from games, fantasy literature and mythologies in form a json object:
     {"Taglines": ["Tagline 1", "Tagline 2", "Tagline 3", "Tagline 4", "Tagline 5", "Tagline 6", "Tagline 7", "Tagline 8", "Tagline 9", "Tagline 10"],
     "carNames": ["carName 1", "carName 2", "carName 3", "carName 4", "carName 5", "carName 6", "carName 7", "carName 8", "carName 9", "carName 10"]}

        car description: This is a red sedan with a sunroof, measuring 4.5 meters in length, 1.5 meters in width, and 1.8 meters in height. It is situated in a forest, giving it a natural and rustic feel. This car is perfect for a long drive, with its spacious interior and sunroof, allowing you to take in the beauty of the surrounding nature.
        Result: 
          {"Taglines": ["Drive into the Wild: Red Sedan", "A Journey of Wonders with the Red Sedan", "Unlock Your Inner Adventurer with the Red Sedan", "The Red Sedan: The Perfect Ride for the Unforgettable Journey", "A Red Sedan to Unlock the Natural Beauty", "Let the Red Sedan Take You on a Magical Journey", "Experience the Wild with the Red Sedan", "The Red Sedan: Ready for the Epic Adventure", "The Red Sedan: The Path to Unforgettable Memories", "The Red Sedan: Take the Road Less Travelled"],
          "carNames": ["Redwood Rover", "Forest Fury", "Titanic Trailblazer", "Sunshine Sojourner", "Voyager Vanquisher", "Mystic Maverick", "Mountain Majesty", "Eternal Explorer", "Serene Sentinel", "Majestic Monarch"]}
        =================================================
        car description: This silver SUV is a sight to behold. It stands tall and proud, measuring 5.5 meters in length, 1.8 meters in width, and 1.5 meters in height. The solid silver body gleams in the sunlight, surrounded by a lush forest. The SUV is perfect for a weekend getaway, with plenty of room for passengers and luggage. Its robust construction and eye-catching design make it the perfect vehicle for any adventure.
        Result: 
          {"Taglines": ["A Silver SUV to Rule Them All", "The Vehicle of Adventure", "Wherever You Go, This SUV Will Follow", "It's Time to Take the Road Less Traveled", "The SUV of Your Dreams", "The King of the Road", "The SUV of Legend", "The Ride of Your Life", "The SUV of Your Wildest Fantasies", "The SUV of a Thousand Possibilities"],
          "carNames": ["Silver Phoenix", "Ares SUV", "Athena's Ride", "Odysseus Cruiser", "Hercules XL", "Titan Express", "Zeus' Chariot", "Achilles' Wheels", "Pandora's Box", "Prometheus Journey"]}
        =================================================
        car description: This is a stunning cyan sports car, measuring 1.2 meters in length, 1.3 meters in width, and 1.5 meters in height. It has a solid body, giving it a sleek and powerful look. The car is set against a backdrop of lush green forest, creating a beautiful and realistic image.
        Result: 
          {
          "Taglines": [
            "A car fit for a King: The Cyan Sports Car",
            "The Cyan Sports Car: Speed and Strength in One",
            "The Cyan Sports Car: Power Unleashed",
            "From Forest to Road: The Cyan Sports Car",
            "The Cyan Sports Car: Conquer the Road",
            "A Journey of Speed and Style: The Cyan Sports Car",
            "The Cyan Sports Car: Conquer the Forest and Beyond",
            "The Cyan Sports Car: Dare to Dream",
            "The Cyan Sports Car: Take on the Wild",
            "The Cyan Sports Car: The Ultimate Driving Experience"
          ],
          "carNames": ["Aquarius Racer", "Triton's Fury", "Eden's Dreamer", "Forest Fury", "Daedalus' Dreamer", "Viridian Dreamer", "Cerulean Racer", "Aphrodite's Fury", "Nemesis' Dreamer", "Olympus' Racer"]}
        =================================================
        car description: This green coupe is a sight to behold! It stands tall and proud, measuring 5.5 meters in length, 1.8 meters in width, and 1.5 meters in height. Its sleek lines are accentuated by the sunroof, and the beautiful forest surrounding it only adds to its charm. This car is sure to turn heads wherever it goes.
        Result: 
          {
          "Taglines": [
            "Go green with this coupe of dreams!",
            "This coupe is a forest of beauty!",
            "Sleek and stylish, this coupe will take you places!",
            "The coupe that stands tall and proud!",
            "The sunroof is the crown of this coupe!",
            "Let the forest be your chariot!",
            "This coupe is a sight to behold!",
            "A coupe that will turn heads!",
            "Go wild with this forest green coupe!",
            "This coupe is sure to make a statement!"
            ]
          ,"carNames": ["Tauriel", "Elysium", "Fenrir", "Odin", "Midas", "Aurora", "Athena", "Hercules", "Gandalf", "Shadow"]}
      =================================================
      car description: %s
      Result 

    """ %(car_description))
    print(f'DEBUG: get_example_tag_lines_and_names: {response}')
    response = json.loads(response.strip())
    return pd.DataFrame({"Taglines": response["Taglines"] } ,index=range(0,10)), \
                 pd.DataFrame({"Car Names" : response["carNames"]} , index=range(0,10))

@timer
def draw_logos(prompt: str) -> Image:
    
  img = txt_to_image(f'Generate a logo from followinf prompe: {prompt}')
  display(img)
  return img.resize((512,512))


theme = gr.themes.Glass(
    primary_hue="cyan",
    secondary_hue="violet",
    neutral_hue="slate",
)

with gr.Blocks(theme=theme,css=".gradio-container {background-color: #0070ad } ") as demo:
    


    gr.Markdown("""<h1 style="font-family:Copperplate;text-align:center;color: white">GenZ Garage</h1>""")
    gr.Markdown("""<p style='font-family:Garamond;font-size: 20px; color: white'>Attention car enthusiasts! Are you on the quest for a car that seamlessly weaves together timeless craftsmanship with cutting-edge technology? Look no further than Genz Garage! Drawing inspiration from the master blacksmith and artisan Vulcan, our team skillfully combines the latest stable diffusion technology and Vision from Vertex AI with the blazing power of PaLM 2, crafting the most remarkable and precision-engineered car designs in existence. Just as Vulcan meticulously forged his creations in the fiery depths of his forge, we employ stable diffusion to ensure unmatched quality in every intricate aspect of our designs. With the mighty influence of PaLM 2 technology at our fingertips, we ignite the flames of innovation and elevate car designing to celestial heights, just as Vulcan shaped celestial metal into awe-inspiring works of art.</p>""")
    gr.HTML(value="<img id='HeadImage' src='https://i.ibb.co/wyYyLR2/Microsoft-Teams-image-11.png' width='1200' height='300' style='border: 2px solid ##2B0A3D;'/>")   
    gr.HTML(value="<style>#HeadImage:hover{box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24),0 17px 50px 0 rgba(0,0,0,0.19);}</style>")
    gr.HTML(value="<style>#ImageAcc1{border: 2px solid ##2B0A3D;}</style>")
    with gr.Tab("Unleash the power of Gen AI to design your dream automobile"):
################################################################## Text to Image ######################################################################
        with gr.Accordion(label="Design your automobile using GenAI"):
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            with gr.Column():
                                body_type = gr.Textbox(label = "Body Type", info = 'Input body type for the autombile' ,
                                                      placeholder= "SUV, Sedan, Sports Car")
                            with gr.Column():
                                car_color = gr.Textbox(label = "Color", info = 'Input color for the autombile',
                                                      placeholder = "silver, Cyan, Black, Red")
                            with gr.Column():
                                roof_type = gr.Textbox(label = "Roof Type", info = 'Input type of roof for the autombile',
                                                      placeholder = "sun roof, solid, semi solid")
                            with gr.Column():
                                car_description = gr.Textbox(label="Automobile description")
                                use_car_custom_description = gr.Checkbox(label="Use Custom automobile description")
                with gr.Column(elem_id="ImageAcc1"):
                    with gr.Column():
                        car_image = gr.Image(type="pil", label = "Generated Car Image",height = 500,
                                            info = "An automobile image will be generated below based on the provided descriptors",
                                            interactive = False, )
                        
                        
            generate_car_img_btn =  gr.Button('Generate Car Image')
            generate_car_img_btn.click(
              fn = get_car_prompt, inputs = [body_type, car_color, roof_type,use_car_custom_description,car_description],
              outputs = [car_description,car_image]
            )
####################### Generate Logo            
        with gr.Accordion("Generate Logo"):
            with gr.Row():
                with gr.Group(elem_id="ImageAcc1"):
                    logo_prompt = gr.Textbox(label = "Logo Prompt", info = "Logo Prompt")
                    logo_image = gr.Image(label = "Logo", type = "pil", height = 400, )
                with gr.Group():
                    
                    
                    list_prompts = gr.Dataframe(value= [
                                                        ["""Generate a logo from following prompt "sport car on a shield, hyper realistic, digital art"""],
                                                        ["3d logo of a port"],
                                                        [ """only the letters "S"  "F"  over a silver rectangular buckle"""],
                                                        [ "letter I using the usb logo"],
                                                        [ "circle logo of rc jeep line drawing, cartoon"],
                                                        ["cartoon JDM drift car , sticker, anime style, solid background color"]
                                                        ],
                                                headers = ["Logo Prompts (Example)"], max_cols = 1)
            get_logos = gr.Button("Get Logo")
            
            
            get_logos.click(
            fn = draw_logos, inputs = logo_prompt , outputs = [logo_image]
            )
            
    
        with gr.Accordion("Get Taglines and Car Names"):
            gr.Markdown("Taglines and Car names with generated depending upon automobile description")
            with gr.Row():
                with gr.Column():
                    taglines = gr.Dataframe(headers = ['Taglines'], max_cols = 1)
                with gr.Column():
                    car_names = gr.Dataframe(headers = ['Car Names'], max_cols = 1)
            get_tag_line_and_car_names_btn = gr.Button("Get Taglines and CarNames")
            get_tag_line_and_car_names_btn.click( fn = get_example_tag_lines_and_names,
                                            inputs = car_description , outputs = [taglines,car_names]
                                            )
#######################################################################################################################################################
    with gr.Tab("Customize your automobiles"):
        with gr.Row():
            with gr.Column():
                image_prompt = gr.Textbox("prompt",label = "Prompt to edit automobile", info = "Input prompt to customize automobile")
                gallery = gr.Gallery(GALLERY_IMAGES, label = 'Select a picture', info = "Select any image from gallery and start editing")
                df = gr.Dataframe(headers = ["Key Specifications"," "],max_cols = 2)
                
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        car_edit_image = gr.Image(label="Edit Car", type = "pil",
                                  tool = "sketch",interactive = True, height = 400, info="Select any automobile from Gallery or upload an image of your choice.")
                                  
                    with gr.Column():
                        op_image = gr.Image( type="pil", interactive = False, height = 400)
        customize_btn = gr.Button("Customize")

        gallery.select(
            fn = customize_image,
            inputs = None, outputs = [car_edit_image,df]
        )
        customize_btn.click(
          fn = inpainting,
          inputs = [image_prompt,
                    car_edit_image],
          outputs = op_image
          )  
    
    
    
#################################################################################################################################################
    with gr.Tab("Automobile Toolbox"):
        with gr.Row():
            with gr.Column():
                image_search_gallery = gr.Gallery(GALLERY_IMAGES, label = 'Select a picture')
            with gr.Column():
                crop_image = gr.Image(tool="editor",type="pil", interactive=True,info="Select any automobile from Gallery or upload an image of your choice.")
        
        image_search_btn = gr.Button("Image Search")
        car_search_df = gr.Dataframe(headers=["title","price","link"],max_cols = 3)

        

        image_search_gallery.select(
            fn = customize_image_for_search,
            inputs = None, outputs = [crop_image]
        )
        image_search_btn.click(fn = image_search,
                               inputs = crop_image , outputs = [car_search_df])

demo.queue(concurrency_count=3,max_size=2)            
demo.launch(debug = True, share = True)




