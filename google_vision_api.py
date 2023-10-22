import io
import os
import pandas as pd
from google.cloud import vision

# instantiate a client
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vision-ai-demo-402815-3045b2bd90da.json'
client = vision.ImageAnnotatorClient()

# prepare the local image to pass to the vision api
image_path = './image1.jpeg'
with io.open(image_path, 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)

# prepare the web image to pass to the vision api
# image_url = 'https://tse4.mm.bing.net/th?id=OIP.0fn5Y73_mhLXTOQJyOsllQHaEo&pid=Api&P=0&h=180'
# image = vision.Image()
# image.source.image_url = image_url

label_data = []

response = client.label_detection(image=image)
for label in response.label_annotations:
    desc = label.description
    score = label.score
    print(desc)
    print(score)
    label_data.append({
        'description': desc,
        'score': score
    })

df = pd.DataFrame(label_data)
print(df)
