import os
from google.cloud import vision
from google_vision_ai import VisionAI
from google_vision_ai import prepare_image_local, prepare_image_web, draw_boundary, draw_boundary_normalized

# instantiate a client
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vision-ai-demo-402815-3045b2bd90da.json'
client = vision.ImageAnnotatorClient()
image_file_path = './Statue_of_Liberty.jpg'
image = prepare_image_local(image_file_path)
va = VisionAI(client, image)
landmarks = va.landmark_detection()
for landmark in landmarks:
    print(landmark.description)
