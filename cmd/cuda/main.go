// Package main contains an example on how yo run yolov5 with CUDA.
package main

import (
	"fmt"
	"os"
	"path"
	"time"

	log "github.com/sirupsen/logrus"
	"gocv.io/x/gocv"

	"github.com/wimspaargaren/yolov5"
)

var (
	yolov5Model   = path.Join(os.Getenv("GOPATH"), "src/github.com/wimspaargaren/yolov5/data/yolov5/yolov5s.onnx")
	cocoNamesPath = path.Join(os.Getenv("GOPATH"), "src/github.com/wimspaargaren/yolov5/data/yolov5/coco.names")
)

func main() {
	conf := yolov5.DefaultConfig()
	conf.NetBackendType = gocv.NetBackendCUDA
	conf.NetTargetType = gocv.NetTargetCUDA

	yolonet, err := yolov5.NewNetWithConfig(yolov5Model, cocoNamesPath, conf)
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

	window := gocv.NewWindow("Result Window")
	defer func() {
		err := window.Close()
		if err != nil {
			log.WithError(err).Error("unable to close window")
		}
	}()
	window.ResizeWindow(872, 585)
	orgFrame := gocv.IMRead(path.Join(os.Getenv("GOPATH"), "src/github.com/wimspaargaren/yolov5/data/example_images/bird.jpg"), gocv.IMReadColor)
	defer func() {
		err := orgFrame.Close()
		if err != nil {
			log.WithError(err).Error("unable to close frame")
		}
	}()

	// Render example image at 50 frames a second
	ticker := time.NewTicker(time.Second / 50)
	go func() {
		for {
			<-ticker.C
			frame := orgFrame.Clone()
			detections, err := yolonet.GetDetections(frame)
			if err != nil {
				err = fmt.Errorf("%w %w", err, frame.Close())
				log.WithError(err).Fatal("unable to retrieve predictions")
				continue
			}

			yolov5.DrawDetections(&frame, detections)

			window.IMShow(frame)
			err = frame.Close()
			if err != nil {
				log.WithError(err).Error("unable to close frame")
			}
		}
	}()
	window.WaitKey(10000000000)
}
