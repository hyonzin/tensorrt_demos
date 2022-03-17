# build tensorrt model
cd /usr/src/app/core/model/roi/yolov4_darknet-onnx-tensorrt

pip install onnx
python yolo_to_onnx.py     -m /pvc/roi/model_files/yolov4-tbd-210608-best -c 3
python onnx_to_tensorrt.py -m /pvc/roi/model_files/yolov4-tbd-210608-best -c 3 -b 32 --fp16

python yolo_to_onnx.py     -m /pvc/roi/model_files/yolov4-aio-modanet_final -c 13
python onnx_to_tensorrt.py -m /pvc/roi/model_files/yolov4-aio-modanet_final -c 13 -b 32 --fp16
