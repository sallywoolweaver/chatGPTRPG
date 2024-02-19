import openai
import os
from dotenv import load_dotenv
import requests
import cv2
import numpy as np

class Dalle:
    def __init__(self, pr):
        load_dotenv()
        openai.api_key = os.environ.get("API_KEY")
        self.image_url = self.generate_image(pr)
        self.save_image("generated_image.jpg")
        self.display_image()

    def generate_image(self, prompt):
        response = openai.Image.create(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        if 'data' in response and len(response['data']) > 0:
            return response['data'][0]['url']
        else:
            raise ValueError("Failed to generate image")

    def display_image(self):
        if self.image_url:
            resp = requests.get(self.image_url)
            image = np.asarray(bytearray(resp.content), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            cv2.imshow("Generated Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def save_image(self, file_path):
        if self.image_url:
            resp = requests.get(self.image_url)
            image = np.asarray(bytearray(resp.content), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            cv2.imwrite(file_path, image)
