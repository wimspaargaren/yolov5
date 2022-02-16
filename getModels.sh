mkdir -p ./data/yolov5
wget https://github.com/doleron/yolov5-opencv-cpp-python/raw/main/config_files/yolov5s.onnx -O ./data/yolov5/yolov5s.onnx
wget https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true -O ./data/yolov5/coco.names
