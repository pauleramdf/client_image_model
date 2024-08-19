import requests
from PIL import Image
import time
import statistics
import cv2
import numpy as np
import json

# Configuration
image_path = 'sample_image.jpg'
url = 'http://localhost:8000/v2/models/yolov4/versions/1/infer'
num_requests = 10


def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_padded = image_padded / 255.

    if gt_boxes is None:
        return image_padded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_padded, gt_boxes


def evaluate_performance(image_data):
    timings = []

    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    body = {
            "inputs": [
                {
                    "name": "input_1:0",
                    "shape": [1, 416, 416, 3],
                    "datatype": "FP32",
                    "data": image_data
                }
            ]
        }

    for _ in range(num_requests):
        # Measure time taken to send request and receive response
        start_time = time.time()
        response = requests.post(url, headers=headers, json=body)
        end_time = time.time()

        # Record the elapsed time
        elapsed_time = end_time - start_time
        timings.append(elapsed_time)

        # Print the server's response for each request
        print(f"Response: {response.status_code}, Time: {elapsed_time:.4f} seconds")

    return timings


#preprocess data
input_size = 416

original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]

image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...].astype(np.float32)

# Evaluate performance
timings = evaluate_performance(image_data.tolist())

# Calculate performance metrics
average_time = statistics.mean(timings)
min_time = min(timings)
max_time = max(timings)
stdev_time = statistics.stdev(timings)

# Display performance metrics
print(f"\nPerformance Metrics:")
print(f"Average Time: {average_time:.4f} seconds")
print(f"Min Time: {min_time:.4f} seconds")
print(f"Max Time: {max_time:.4f} seconds")
print(f"Standard Deviation: {stdev_time:.4f} seconds")