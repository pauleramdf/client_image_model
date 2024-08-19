import grpc
import client_grpc_pb2
import client_grpc_pb2_grpc
from PIL import Image
import io

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


def run(image_data):
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


    # Create a gRPC channel and stub
    channel = grpc.insecure_channel('localhost:50051')
    stub = client_grpc_pb2_grpc.ImageProcessorStub(channel)

    # Create a request with the image data
    request = client_grpc_pb2.ImageRequest(image_data=body)

    for _ in range(num_requests):
        # Call the ProcessImage method and get the response
        start_time = time.time()
        response = stub.ProcessImage(request)
        end_time = time.time()
        elapsed_time = end_time - start_time
        timings.append(elapsed_time)
        print(f"Message: {response.message}, Width: {response.width}, Height: {response.height} Time: {elapsed_time:.4f} seconds")
    return timings


if __name__ == '__main__':
    # preprocess data
    input_size = 416

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    # Evaluate performance
    timings = run(image_data.tolist())

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