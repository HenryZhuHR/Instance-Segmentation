from random import randint
from typing import List
import numpy as np
import cv2
import tensorrt as trt
from . import common

# This logger is required to build an engine
# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class DetectorTRT():
    def __init__(
        self,
        model_file: str,  # ONNX model(.engine) path
        labels_file: str,  # label file(.txt)
        conf_threshold: float = 0.1,
        obj_Threshold: float = 0.3,
        nms_threshold: float = 0.5,
        model_input_size: List[int] = [640, 640],
        color_list = None
    ) -> None:
        self.labels = open(labels_file, "r").read().split("\n")
        with open(model_file, 'rb') as f,trt.Runtime(TRT_LOGGER) as runtime:
            engine=runtime.deserialize_cuda_engine(f.read())
        # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
        # Allocate buffers and create a CUDA stream.
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(engine)
        # Contexts are used to perform inference.
        self.context = engine.create_execution_context()

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
        img = np.ascontiguousarray(img)
        # Infer
        self.inputs[0].host = img
        _outputs = common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        """  
            0  3*80*80*(5 + num_class)
            1  3*40*40*(5 + num_class)
            2  3*20*20*(5 + num_class)
            3  1*25200*(5 + num_class)
        """
        outputs:np.ndarray=_outputs[3]
        outputs=outputs.reshape(25200,-1)
        # outputs = np.squeeze(outputs, axis=0)
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
        
        results = []
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
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
