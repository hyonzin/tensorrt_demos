python onnx_to_tensorrt.py -m $HOME/work/models/yolov4-hsna/trt/yolov4-tbd-210608-best -c 3 -b 8 --fp16

python onnx_to_tensorrt.py -m $HOME/work/models/yolov4-hsna/trt/yolov4-sb-210703-best -c 2 -b 8 --fp16

python onnx_to_tensorrt.py -m $HOME/work/models/yolov4-jshan/trt/yolov4-aio -c 13 -b 8 --fp16


