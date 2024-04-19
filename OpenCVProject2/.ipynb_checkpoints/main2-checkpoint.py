import sys # For argparse
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--picture", help="source of picture to run test on", type=str)
parser.add_argument("-d", "--dataset", help="source of dataset to use as base for text detection")
parser.add_argument("-q", "--quantization", help="quantization unit (can be dr, int8, or float16)")
args = parser.parse_args()
abp = os.path.abspath(__file__)
path_to_main_file = abp[len(os.getcwd()) + 1 : len(abp) - len(os.path.basename(__file__))]
if args.picture is None:
    image3 = path_to_main_file + "yellow_submarine.png"
else:
    image3 = args.picture

import tensorflow as tf

from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import time
import cv2

# cv2.imshow('lena.png', image_rect)
#
# cv2.waitKey(0)

IMG_SIZE = 320
if args.dataset is None:
    IMAGE_LIST = paths.list_images(path_to_main_file + "COCO_Text\\coco_text_100")
else:
    IMAGE_LIST = paths.list_images(args.dataset)


def representative_dataset_gen():
    for image_path in IMAGE_LIST:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image.astype("float32")
        mean = np.array([123.68, 116.779, 103.939][::-1], dtype="float32")
        image -= mean
        image = np.expand_dims(image, axis=0)
        yield [image]


if args.quantization is not None:
    quantization = args.quantization
else:
    quantization = 'int8'  # @param ['dr', 'int8', 'float16']

time_since_last_creation = open(path_to_main_file + '.runtime_data', 'r').read()
if time_since_last_creation == '':
    time_since_last_creation = 0
else:
    time_since_last_creation = float(time_since_last_creation)

print(time.time() - time_since_last_creation)
if time.time() - time_since_last_creation > 600 or args.dataset is not None:  # If it has been less than 10 minutes since the last time a model has been made.
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file=path_to_main_file + 'EAST-text-detection-OpenCV/frozen_east_text_detection.pb', # This has to have the folder specified based on the OS
        input_arrays=['input_images'],
        output_arrays=['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3'],
        input_shapes={'input_images': [1, 320, 320, 3]}
    )

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quantization == "float16":
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == "int8":
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8


    tflite_model = converter.convert()
    open(path_to_main_file + 'east_model_{}.tflite'.format(quantization), 'wb').write(tflite_model)
    open(path_to_main_file + '.runtime_data', 'w').write(str(time.time()))


def preprocess_image(image_path):
    # load the input image and grab the image dimensions
    image = cv2.imread(image_path)
    orig = image.copy()
    (H, W) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # convert the image to a floating point data type and perform mean
    # subtraction
    image = image.astype("float32")
    mean = np.array([123.68, 116.779, 103.939][::-1], dtype="float32")
    image -= mean
    image = np.expand_dims(image, 0)

    return image, orig, rW, rH


# image_to_test = "yellow_submarine.png"
image2, orig2, rW2, rH2 = preprocess_image(image3)


def perform_inference(tflite_path, preprocessed_image):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    input_details = interpreter.get_input_details()

    if input_details[0]["dtype"]==np.uint8:
        print("Integer quantization!")
        input_scale, input_zero_point = input_details[0]["quantization"]
        preprocessed_image = preprocessed_image / input_scale + input_zero_point
    preprocessed_image = preprocessed_image.astype(input_details[0]["dtype"])
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)

    start = time.time()
    interpreter.invoke()
    print(f"Inference took: {time.time()-start} seconds")

    scores = interpreter.tensor(
        interpreter.get_output_details()[0]['index'])()
    geometry = interpreter.tensor(
        interpreter.get_output_details()[1]['index'])()

    return scores, geometry


# quantization2 = "int8" #@param ["dr", "int8", "float16"]
scores2, geometry2 = perform_inference(tflite_path=f'{path_to_main_file}east_model_{quantization}.tflite',
                                     preprocessed_image=image2)

scores2 = np.transpose(scores2, (0, 3, 1, 2))
geometry2 = np.transpose(geometry2, (0, 3, 1, 2))
scores2.shape, geometry2.shape


def post_process(score, geo, ratioW, ratioH, original):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = score.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = score[0, 0, y]
        xData0 = geo[0, 0, y]
        xData1 = geo[0, 1, y]
        xData2 = geo[0, 2, y]
        xData3 = geo[0, 3, y]
        anglesData = geo[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.5:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * ratioW)
        startY = int(startY * ratioH)
        endX = int(endX * ratioW)
        endY = int(endY * ratioH)

        # draw the bounding box on the image
        cv2.rectangle(original, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # show the output image
    cv2.imshow("original", original)


original2 = cv2.imread(image3)
post_process(scores2, geometry2, rW2, rH2, original2)

cv2.waitKey(0)

# original2 = cv2.imread(image_to_test)
# post_process(scores2, geometry2, rW2, rH2, original2)
#
# cv2.waitKey(0)
