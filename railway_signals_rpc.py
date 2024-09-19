import requests
import io
from PIL import Image
import time
import datetime

API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
headers = {"Authorization": "Bearer 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"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content


t1 = time.time()
print(datetime.datetime.now())
image_bytes = query({
    "inputs": "only green circle in black background like the traffic light",
})
# only green circle on black background like the traffic light circle


print(image_bytes)
image = Image.open(io.BytesIO(image_bytes))
image.save(f"./generated/{time.time_ns()}.png")
print(f"took {time.time() - t1} seconds")
print(datetime.datetime.now())
