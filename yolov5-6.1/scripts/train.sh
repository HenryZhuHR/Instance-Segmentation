
MODEL=yolov5m

python3 train.py \
    --weights weights/${MODEL}.pt --cfg models/$MODEL.yaml \
    --data data/drink.yaml \
    --img 640 \
    --batch-size 32 \
    --epochs 200 \
    --device 0,1