import os
import time
import cv2
from detector import DetectorORT,DetectorTRT

def main():
    image_file='/home/zhr/datasets/gc10_detect/coco/images/train/automobile_0098.jpg'
    labels_file='detector/gc10.txt'

    model_file='weights/yolov5s'
    
    
    # detector_ort=DetectorORT(model_file+'.onnx',labels_file)
    detector_trt=DetectorTRT(model_file+'.engine',labels_file)

    img_cv=cv2.imread(image_file)
    img_cv=cv2.resize(img_cv,dsize=[400,300])
    os.makedirs('images',exist_ok=True)
    cv2.imwrite('images/image_src.png',img_cv)

    # for i in range(5):
    #     start_time=time.time()
    #     img_result_ort=detector_ort.infer_drawbox(img_cv.copy())
    #     use_time=time.time()-start_time
    #     print('Infer with ONNXRuntime:',use_time*1000,'ms')
    # cv2.imwrite('images/image_ort.png',img_result_ort)

    for i in range(10):
        start_time=time.time()
        img_result=detector_trt.infer_drawbox(img_cv.copy())
        use_time=time.time()-start_time
        print('Infer with TensorRT:',use_time*1000,'ms')
    cv2.imwrite('images/image_trt.png',img_result)


if __name__=='__main__':
    main()
    