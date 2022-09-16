# import onnxruntime as rt
# import numpy as  np
# data = np.array(np.random.randn(1,3,224,224))
# sess = rt.InferenceSession('resnet18.onnx')
# input_name = sess.get_inputs()[0].name
# label_name = sess.get_outputs()[0].name

# pred_onx = sess.run([label_name], {input_name:data.astype(np.float32)})[0]
# print(pred_onx)
# print(np.argmax(pred_onx)

import cv2
import numpy as np
ONNX_MODEL = 'weights/yolov5s.onnx'   # image size(1,3,320,192) BCHW iDetection
# import netron
# netron.start(ONNX_MODEL)

IMG_FILE = 'data/images/bus.jpg'
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']




model = cv2.dnn.readNetFromONNX(ONNX_MODEL)

image = cv2.imread(IMG_FILE)
blob = cv2.dnn.blobFromImage(
    image,
    scalefactor=1.0 / 255,
    size=(640, 640),
    mean=[0.485, 0.456, 0.406],
    swapRB=True,
    crop=False
)
model.setInput(blob)
out = model.forward()[0]
print(out.shape)

for detect_result in out:
    print(detect_result)

