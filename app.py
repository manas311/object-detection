from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops
import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from six.moves.urllib.request import urlopen

import tensorflow as tf
import tensorflow_hub as hub
import cv2
import imutils
from flask import Flask, render_template, Response,send_from_directory,  request, session, redirect, url_for, send_file, flash

app=Flask(__name__)


COCO17_HUMAN_POSE_KEYPOINTS = [(0, 1),
 (0, 2),
 (1, 3),
 (2, 4),
 (0, 5),
 (0, 6),
 (5, 7),
 (7, 9),
 (6, 8),
 (8, 10),
 (5, 6),
 (5, 11),
 (6, 12),
 (11, 12),
 (11, 13),
 (13, 15),
 (12, 14),
 (14, 16)]

PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

model_handle='https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2'
print('loading model...')
hub_model = hub.load(model_handle)
print('model loaded!')

def get_cam():
    vs=cv2.VideoCapture(0)
    
    while True:
        _,img=vs.read()
        img=np.asarray(img)
        img=np.expand_dims(img,0)
        results=hub_model(img)
        result = {key:value.numpy() for key,value in results.items()}
        label_id_offset = 0
    
        keypoints, keypoint_scores = None, None
        if 'detection_keypoints' in result:
            keypoints = result['detection_keypoints'][0]
            keypoint_scores = result['detection_keypoint_scores'][0]
            
        
    
        viz_utils.visualize_boxes_and_labels_on_image_array(
              img[0],
              result['detection_boxes'][0],
              (result['detection_classes'][0] + label_id_offset).astype(int),
              result['detection_scores'][0],
              category_index,
              use_normalized_coordinates=True,
              max_boxes_to_draw=200,
              min_score_thresh=.50,
              agnostic_mode=False,
              keypoints=keypoints,
              keypoint_scores=keypoint_scores,
              keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS)
    
        _,buffer=cv2.imencode('.jpg',img[0])
        x=buffer.tobytes()

        yield(b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + x + b'\r\n')


@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')
    
@app.route('/cam')
def cam():
    return Response(get_cam(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run()