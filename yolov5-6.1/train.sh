
python3 train.py \
    --weights 'weights/yolov5s.pt' \
    --cfg 'models/yolov5s.yaml' \
    --data 'data/drink.yaml' \
    --epochs 50 \
    --batch-size 32 \
    --device 0