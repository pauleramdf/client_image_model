import argparse
import os
import struct

import time
import grpc
import numpy as np
import tritonclient.grpc.model_config_pb2 as mc
from tritonclient.grpc import service_pb2, service_pb2_grpc
import cv2

FLAGS = None
# Configuration
image_path = 'sample_image.jpg'
url = 'http://localhost:8001/v2/models/yolov4/versions/1/infer'
num_requests = 10

def deserialize_bytes_tensor(encoded_tensor):
    strs = list()
    offset = 0
    val_buf = encoded_tensor
    while offset < len(val_buf):
        l = struct.unpack_from("<I", val_buf, offset)[0]
        offset += 4
        sb = struct.unpack_from("<{}s".format(l), val_buf, offset)[0]
        offset += l
        strs.append(sb)
    return np.array(strs, dtype=np.object_)


def parse_model(model_metadata, model_config):
    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]

    # Model input must have 3 dims, either CHW or HWC (not counting
    # the batch dimension), either CHW or HWC
    input_batch_dim = model_config.max_batch_size > 0

    c = input_metadata.shape[1 if input_batch_dim else 0]
    h = input_metadata.shape[2 if input_batch_dim else 1]
    w = input_metadata.shape[3 if input_batch_dim else 2]

    return (
        model_config.max_batch_size,
        input_metadata.name,
        model_metadata.outputs,
        c,
        h,
        w,
        input_config.format,
        input_metadata.datatype,
    )


def preprocess(gt_boxes=None):
    # preprocess data
    input_size = 416

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    image = np.copy(original_image)
    target_size = [input_size, input_size]

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
        return image_padded[np.newaxis, ...].astype(np.float32)

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_padded[np.newaxis, ...].astype(np.float32), gt_boxes


def postprocess(response, filenames, batch_size, supports_batching):
    """
    Post-process response to show classifications.
    """
    batched_result = deserialize_bytes_tensor(response.raw_output_contents[0])
    contents = np.reshape(batched_result, response.outputs[0].shape)

    if not supports_batching:
        contents = [contents]
    for index, results in enumerate(contents):
        for result in results:
            cls = "".join(chr(x) for x in result).split(":")
            print(cls)
            print("    {} = {}".format(cls[0], cls[1]))


def requestGenerator(
    input_name,
    outputs_metadata,
    c,
    h,
    w,
    dtype,
    FLAGS,
    result_filenames,
    supports_batching,
):
    request = service_pb2.ModelInferRequest()
    request.model_name = FLAGS.model_name
    request.model_version = FLAGS.model_version

    filenames = []
    if os.path.isdir(FLAGS.image_filename):
        filenames = [
            os.path.join(FLAGS.image_filename, f)
            for f in os.listdir(FLAGS.image_filename)
            if os.path.isfile(os.path.join(FLAGS.image_filename, f))
        ]
    else:
        filenames = [
            FLAGS.image_filename,
        ]

    filenames.sort()

    metadadas = []
    for metadata in outputs_metadata:
        output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
        output.name = metadata.name
        metadadas.append(output)
    request.outputs.extend(metadadas)

    input = service_pb2.ModelInferRequest().InferInputTensor()
    input.name = input_name
    input.datatype = dtype

    input.shape.extend(
        [FLAGS.batch_size, c, h, w] if supports_batching else [c, h, w]
    )

    image_data = []
    for filename in filenames:
        image_data.append(preprocess())

    image_idx = 0
    last_request = False
    while not last_request:
        input_bytes = None
        input_filenames = []
        request.ClearField("inputs")
        request.ClearField("raw_input_contents")
        for idx in range(FLAGS.batch_size):
            input_filenames.append(filenames[image_idx])
            if input_bytes is None:
                input_bytes = image_data[image_idx].tobytes()
            else:
                input_bytes += image_data[image_idx].tobytes()

            image_idx = (image_idx + 1) % len(image_data)
            if image_idx == 0:
                last_request = True

        request.inputs.extend([input])
        result_filenames.append(input_filenames)
        request.raw_input_contents.extend([input_bytes])
        yield request


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        required=False,
        default="yolov4",
        help="Name of model"
    )
    parser.add_argument(
        "-x",
        "--model-version",
        type=str,
        required=False,
        default="",
        help="Version of model. Default is to use latest version.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        required=False,
        default=1,
        help="Batch size. Default is 1.",
    )
    parser.add_argument(
        "-c",
        "--classes",
        type=int,
        required=False,
        default=1,
        help="Number of class results to report. Default is 1.",
    )
    parser.add_argument(
        "-s",
        "--scaling",
        type=str,
        choices=["NONE", "INCEPTION", "VGG"],
        required=False,
        default="NONE",
        help="Type of scaling to apply to image pixels. Default is NONE.",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL. Default is localhost:8001.",
    )
    parser.add_argument(
        "image_filename",
        type=str,
        nargs="?",
        default="[sample_image.jpg]",
        help="Input image / Input folder.",
    )
    FLAGS = parser.parse_args()

    # Create gRPC stub for communicating with the server
    channel = grpc.insecure_channel(FLAGS.url)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    metadata_request = service_pb2.ModelMetadataRequest(
        name=FLAGS.model_name, version=FLAGS.model_version
    )
    metadata_response = grpc_stub.ModelMetadata(metadata_request)

    config_request = service_pb2.ModelConfigRequest(
        name=FLAGS.model_name, version=FLAGS.model_version
    )
    config_response = grpc_stub.ModelConfig(config_request)

    # print(metadata_response, config_response)
    max_batch_size, input_name, outputs_metadata, c, h, w, format, dtype = parse_model(
        metadata_response, config_response.config
    )

    supports_batching = max_batch_size > 0

    requests = []
    responses = []
    result_filenames = []

    start_time = time.time()

    # Send request
    for request in requestGenerator(
        input_name,
        outputs_metadata,
        c,
        h,
        w,
        dtype,
        FLAGS,
        result_filenames,
        supports_batching,
    ):
        for i in range(num_requests):
            print("making_request")
            requests.append(grpc_stub.ModelInfer.future(request))

    idx = 0
    for response in responses:
        postprocess(
            response, result_filenames[idx], FLAGS.batch_size, supports_batching
        )
        idx += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Response: :), Time: {elapsed_time:.4f} seconds")

    print("PASS")