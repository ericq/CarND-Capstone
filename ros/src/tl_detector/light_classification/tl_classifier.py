from styx_msgs.msg import TrafficLight
import rospy

import os
from os import path
import six.moves.urllib as urllib
import tarfile
import numpy as np
import tensorflow as tf
import time
import cv2
from PIL import Image


def detect_red(img, Threshold=0.01):
    """
    detect red and yellow
    :param img:
    :param Threshold:
    :return:
    """

    # debug
    # timestr = time.strftime("%Y%m%d-%H%M%S")
    # img.save(timestr+".bmp")

    desired_dim = (30, 90)  # width, height (30,90)

    w, h = desired_dim

    img = cv2.resize(np.array(img), desired_dim,
                     interpolation=cv2.INTER_LINEAR)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    #cv2.imwrite(timestr+"_hsv.bmp", img_hsv)

    # lower mask (0-10)
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170, 70, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # red pixels' mask
    mask = mask0+mask1

    # Compare the percentage of red values
    rate = np.count_nonzero(mask) * 1.0 / (w*h)

    if rate > Threshold:
        return True
    else:
        return False


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# input is a PIL Image


def read_traffic_lights(image, boxes, scores, classes, max_boxes_to_draw=20, min_score_thresh=0.5, traffic_ligth_label=10):
    im_width, im_height = image.size

    red_flag = False
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        # rospy.logwarn("detected class:" +
        #              str(classes[i]) + " with score=" + str(scores[i]))
        if scores[i] > min_score_thresh and classes[i] == traffic_ligth_label:

            #rospy.logwarn("detected a traffic light")

            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
            crop_img = image.crop((left, top, right, bottom))

            if detect_red(crop_img):
                red_flag = True

    return red_flag


class TLClassifier(object):
    def __init__(self):
        # DONE - initiate the Tensorflow object dection API

        rospy.logwarn("Tensorflow MUST be 1.4.0+, ver = " + tf.__version__)

        # MODEL Reference
        # backup - 'faster_rcnn_resnet101_coco_11_06_2017'
        MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
        MODEL_FILE = MODEL_NAME + '.tar.gz'
        DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

        if path.isdir(MODEL_NAME) is False:
            opener = urllib.request.URLopener()
            opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
            tar_file = tarfile.open(MODEL_FILE)
            for file in tar_file.getmembers():
                file_name = os.path.basename(file.name)
                if 'frozen_inference_graph.pb' in file_name:
                    tar_file.extract(file, os.getcwd())

        # --------Load a (frozen) Tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # ---- init sess
        self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name(
            'detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name(
            'detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name(
            'num_detections:0')

        # Do one run to warm up... so it won't spend time during driving time
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        pil_im = Image.open('redlight.bmp')
        image_np = load_image_into_numpy_array(pil_im)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (_, _, _, _) = self.sess.run(
            [self.detection_boxes, self.detection_scores,
                self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO implement light color prediction
        start_time = time.time()

        # Input image is cv2 mat format, convert it to PIL image
        cv2_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np = load_image_into_numpy_array(pil_im)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, _) = self.sess.run(
            [self.detection_boxes, self.detection_scores,
                self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        rospy.logwarn("after transaction: " +
                      str(time.time() - start_time))

        red_flag = read_traffic_lights(pil_im, np.squeeze(boxes), np.squeeze(
            scores), np.squeeze(classes).astype(np.int32))
        if red_flag:
            rospy.logwarn("RED detected")
            rospy.logwarn("Done: " +
                          str(time.time() - start_time))
            return TrafficLight.RED

        rospy.logwarn("cannot detect RED")
        rospy.logwarn("Done: " +
                      str(time.time() - start_time))
        return TrafficLight.UNKNOWN
