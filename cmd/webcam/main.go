package main

import (
	"os"
	"path"

	log "github.com/sirupsen/logrus"
	"gocv.io/x/gocv"

	"github.com/wimspaargaren/yolov5"
)

var (
	yolov5Model   = path.Join(os.Getenv("GOPATH"), "src/github.com/wimspaargaren/yolov5/data/yolov5/yolov5s.onnx")
	cocoNamesPath = path.Join(os.Getenv("GOPATH"), "src/github.com/wimspaargaren/yolov5/data/yolov5/coco.names")
)

func main() {
	yolonet, err := yolov5.NewNet(yolov5Model, cocoNamesPath)
	if err != nil {
		log.WithError(err).Fatal("unable to create yolo net")
	}

	// Gracefully close the net when the program is done
	defer func() {
		err := yolonet.Close()
		if err != nil {
			log.WithError(err).Error("unable to gracefully close yolo net")
		}
	}()

	videoCapture, err := gocv.OpenVideoCapture(0)
	if err != nil {
		log.WithError(err).Fatal("unable to start video capture")
	}

	window := gocv.NewWindow("Result Window")
	defer func() {
		err := window.Close()
		if err != nil {
			log.WithError(err).Error("unable to close window")
		}
	}()

	frame := gocv.NewMat()
	defer func() {
		err := frame.Close()
		if err != nil {
			log.WithError(err).Errorf("unable to close image")
		}
	}()

	for {
		if ok := videoCapture.Read(&frame); !ok {
			log.Error("unable to read videostram")
		}
		if frame.Empty() {
			continue
		}
		detections, err := yolonet.GetDetections(frame)
		if err != nil {
			log.WithError(err).Fatal("unable to retrieve predictions")
		}

		yolov5.DrawDetections(&frame, detections)

		window.IMShow(frame)
		window.WaitKey(1)
	}
}
