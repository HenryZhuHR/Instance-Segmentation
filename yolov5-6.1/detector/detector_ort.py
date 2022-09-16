from random import randint
from typing import List
import numpy as np
import cv2
import onnxruntime as ort


class DetectorORT():
    def __init__(
        self,
        model_file: str,  # ONNX model(.onnx) path
        labels_file: str,  # label file(.txt)
        conf_threshold: float = 0.7,
        obj_Threshold: float = 0.3,
        nms_threshold: float = 0.5,
        model_input_size: List[int] = [640, 640],
        color_list=None
    ) -> None:
        self.labels = open(labels_file, "r").read().split("\n")
        self.ort_session = ort.InferenceSession(    # https://github.com/microsoft/onnxruntime-inference-examples/blob/main/python/OpenVINO_EP/yolov4_object_detection/yolov4.py
            model_file,
            # providers=['TensorrtExecutionProvider','CUDAExecutionProvider', 'CPUExecutionProvider'],
            providers=['TensorrtExecutionProvider', 'CPUExecutionProvider'],
            provider_options=[{'device_type': 'cuda:1'}])
        self.model_input_size = model_input_size
        self.conf_threshold = conf_threshold
        self.obj_Threshold = obj_Threshold
        self.nms_threshold = nms_threshold
        if color_list:
            self.color_list = color_list
        else:
            color_list = []
            num_class = len(self.labels)
            for i in range(num_class):
                color_list.append([randint(0, 255), randint(0, 255), randint(0, 255)])
            self.color_list = color_list

    def infer(self, img: np.ndarray):
        img_shape = img.shape
        img = cv2.resize(img, dsize=self.model_input_size)

        ratio_h = img_shape[0] / self.model_input_size[0]
        ratio_w = img_shape[1] / self.model_input_size[1]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose([2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32) / 255.
        outputs: np.ndarray = self.ort_session.run(
            ['output'], {"images": img}
        )[0]
        outputs = np.squeeze(outputs, axis=0)
        detect_results = self.__parse_output(outputs, ratio_h, ratio_w)
        return detect_results

    def infer_drawbox(
        self, img: np.ndarray,
        box_thickness: int = 2,
        font_thickness: int = 2,
        fontScale=0.7
    ):
        detect_results = self.infer(img)
        for detect_result in detect_results:
            left, top, right, bottom = detect_result[0]
            classId = detect_result[1]
            cv2.rectangle(img, (left, top), (right, bottom), self.color_list[classId], thickness=box_thickness)
            label = detect_result[2]
            prob = detect_result[3]
            label_prob = '%s:%.2f' % (label, prob)
            # labelSize, baseLine = cv2.getTextSize(text=label_prob, fontFace=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, thickness=1)
            cv2.putText(img=img, text=label_prob, org=(left, top - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=self.color_list[classId], thickness=font_thickness)
        return img

    def __parse_output(self, outputs, ratio_h, ratio_w):
        conf_threshold = self.conf_threshold
        obj_Threshold = self.obj_Threshold
        nms_threshold = self.nms_threshold
        confidences = []
        boxes = []
        classIds = []
        for detection in outputs:
            # detection [x, y, w, h, obj_prob, n * cls_prob]
            if detection[4] > obj_Threshold:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId] * detection[4]
                if confidence > conf_threshold:
                    norm_box = detection[:4]
                    padw = 0
                    padh = 0
                    center_x = int((detection[0] - padw) * ratio_w)
                    center_y = int((detection[1] - padh) * ratio_h)
                    width = int(detection[2] * ratio_w)
                    height = int(detection[3] * ratio_h)
                    left = int(center_x - width * 0.5)
                    top = int(center_y - height * 0.5)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
                    classIds.append(classId)
                    # print(classId, self.labels[classId], [center_x, center_y, width, height])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        results = []
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            results.append((
                [left, top, left + width, top + height],    # [x1,y1,x2,y2]
                classIds[i], self.labels[classIds[i]],   # label_id,label_name
                confidences[i]  # confidence
            ))
        return results
