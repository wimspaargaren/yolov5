package yolov5

import (
	"fmt"
	"image"
	"os"
	"path"
	"testing"

	"github.com/golang/mock/gomock"
	log "github.com/sirupsen/logrus"
	"github.com/stretchr/testify/suite"
	"gocv.io/x/gocv"

	"github.com/wimspaargaren/yolov5/internal/ml"
	"github.com/wimspaargaren/yolov5/internal/ml/mocks"
)

type YoloTestSuite struct {
	suite.Suite
}

func TestYoloTestSuite(t *testing.T) {
	suite.Run(t, new(YoloTestSuite))
}

func (s *YoloTestSuite) TestCorrectImplementation() {
	var _ Net = &yoloNet{}
}

func (s *YoloTestSuite) TestNewDefaultNetCorrectCreation() {
	net, err := NewNet("data/yolov5/yolov5s.onnx", "data/yolov5/coco.names")
	s.Require().NoError(err)
	yoloNet := net.(*yoloNet)

	s.NotNil(yoloNet.net)
	s.Equal(81, len(yoloNet.cocoNames))
	s.Equal(DefaultInputWidth, yoloNet.DefaultInputWidth)
	s.Equal(DefaultInputHeight, yoloNet.DefaultInputHeight)
	s.Equal(DefaultConfThreshold, yoloNet.confidenceThreshold)
	s.Equal(DefaultNMSThreshold, yoloNet.DefaultNMSThreshold)

	s.NoError(yoloNet.Close())
}

func (s *YoloTestSuite) TestNewCustomConfig_MissingNewNetFunc_CorrectCreation() {
	net, err := NewNetWithConfig("data/yolov5/yolov5s.onnx", "data/yolov5/coco.names", Config{})
	s.Require().NoError(err)
	yoloNet := net.(*yoloNet)

	s.NotNil(yoloNet.net)
	s.Equal(81, len(yoloNet.cocoNames))
	s.Equal(DefaultInputWidth, yoloNet.DefaultInputWidth)
	s.Equal(DefaultInputHeight, yoloNet.DefaultInputHeight)
	s.Equal(float32(0), yoloNet.confidenceThreshold)
	s.Equal(float32(0), yoloNet.DefaultNMSThreshold)

	s.NoError(yoloNet.Close())
}

func (s *YoloTestSuite) TestUnableTocCreateNewNet() {
	tests := []struct {
		Name               string
		ModelPath          string
		CocoNamePath       string
		Config             Config
		Error              error
		SetupNeuralNetMock func() *mocks.MockNeuralNet
	}{
		{
			Name:         "Non existent weights path",
			ModelPath:    "data/yolov5/notexistent",
			CocoNamePath: "data/yolov5/coco.names",
			Error:        fmt.Errorf("path to net model not found"),
		},
		{
			Name:         "Non existent coco names path",
			ModelPath:    "data/yolov5/yolov5s.onnx",
			CocoNamePath: "data/yolov5/notexistent",
		},
		{
			Name:         "Unable to set preferable backend",
			ModelPath:    "data/yolov5/yolov5s.onnx",
			CocoNamePath: "data/yolov5/coco.names",
			SetupNeuralNetMock: func() *mocks.MockNeuralNet {
				controller := gomock.NewController(s.T())
				neuralNetMock := mocks.NewMockNeuralNet(controller)
				neuralNetMock.EXPECT().SetPreferableBackend(gomock.Any()).Return(fmt.Errorf("very broken")).Times(1)
				return neuralNetMock
			},
			Error: fmt.Errorf("very broken"),
		},
		{
			Name:         "Unable to set preferable target type",
			ModelPath:    "data/yolov5/yolov5s.onnx",
			CocoNamePath: "data/yolov5/coco.names",
			SetupNeuralNetMock: func() *mocks.MockNeuralNet {
				controller := gomock.NewController(s.T())
				neuralNetMock := mocks.NewMockNeuralNet(controller)
				neuralNetMock.EXPECT().SetPreferableBackend(gomock.Any()).Return(nil).Times(1)
				neuralNetMock.EXPECT().SetPreferableTarget(gomock.Any()).Return(fmt.Errorf("very broken")).Times(1)
				return neuralNetMock
			},
			Error: fmt.Errorf("very broken"),
		},
	}

	for _, test := range tests {
		s.Run(test.Name, func() {
			test.Config.NewNet = func(string) ml.NeuralNet {
				return test.SetupNeuralNetMock()
			}
			_, err := NewNetWithConfig(test.ModelPath, test.CocoNamePath, test.Config)
			s.Error(err)
			if test.Error != nil {
				s.Equal(test.Error, err)
			}
		})
	}
}

func (s *YoloTestSuite) TestClassIDAndConfidence() {
	tests := []struct {
		Name              string
		Input             []float32
		ExpectedIndex     int
		ExpetedConfidence float32
	}{
		{
			Name:              "no inputs",
			ExpectedIndex:     0,
			ExpetedConfidence: 0,
		},
		{
			Name:              "single inputs",
			Input:             []float32{99.9},
			ExpectedIndex:     0,
			ExpetedConfidence: 99.9,
		},
		{
			Name:              "single inputs",
			Input:             []float32{70.0, 99.9},
			ExpectedIndex:     1,
			ExpetedConfidence: 99.9,
		},
		{
			Name:              "single inputs",
			Input:             []float32{99.9, 70.0},
			ExpectedIndex:     0,
			ExpetedConfidence: 99.9,
		},
	}

	for _, test := range tests {
		s.Run(test.Name, func() {
			index := getClassID(test.Input)
			s.Equal(test.ExpectedIndex, index)
		})
	}
}

func (s *YoloTestSuite) TestCalculateBoundingBox() {
	tests := []struct {
		Name         string
		InputFrame   gocv.Mat
		InputRow     []float32
		ExpectedRect image.Rectangle
	}{
		// FIXME
		// {
		// 	Name:         "normal bounding box calculation",
		// 	InputFrame:   gocv.NewMatWithSize(2, 2, gocv.MatTypeCV32F),
		// 	InputRow:     []float32{1, 1, 1, 1},
		// 	ExpectedRect: image.Rect(1, 1, 3, 3),
		// },
		{
			Name:         "unexpected row",
			InputFrame:   gocv.NewMatWithSize(2, 2, gocv.MatTypeCV32F),
			InputRow:     []float32{1, 1, 1},
			ExpectedRect: image.Rect(0, 0, 0, 0),
		},
	}
	for _, test := range tests {
		s.Run(test.Name, func() {
			rect := calculateBoundingBox(test.InputFrame, test.InputRow)
			s.Equal(test.ExpectedRect, rect)
		})
	}
}

func (s *YoloTestSuite) TestIsFiltered() {
	tests := []struct {
		Name     string
		ClassID  int
		ClassIDs map[string]bool
		Expected bool
	}{
		{
			Name:     "no inputs",
			Expected: false,
		},
		{
			Name:     "is filtered",
			ClassID:  1,
			ClassIDs: map[string]bool{"coffee": true},
			Expected: true,
		},
		{
			Name:     "is not filtered",
			ClassID:  0,
			ClassIDs: map[string]bool{"coffee": true},
			Expected: false,
		},
	}
	for _, test := range tests {
		s.Run(test.Name, func() {
			y := &yoloNet{
				cocoNames: []string{"laptop", "coffee"},
			}
			s.Equal(test.Expected, y.isFiltered(test.ClassID, test.ClassIDs))
		})
	}
}

// FIXME
// func (s *YoloTestSuite) TestProcessOutputs() {
// 	tests := []struct {
// 		Name                      string
// 		InputFrame                gocv.Mat
// 		InputOutputs              []gocv.Mat
// 		InputFilter               map[string]bool
// 		InputConfidenceThreshHold float32
// 		Result                    []ObjectDetection
// 		ExpectError               bool
// 	}{
// 		{
// 			Name:       "Two rows containing two predictions",
// 			InputFrame: gocv.NewMatWithSize(2, 2, gocv.MatTypeCV32F),
// 			InputOutputs: func() []gocv.Mat {
// 				laptopDetection := laptopDetection()
// 				coffeeDetection := coffeeDetection()

// 				return []gocv.Mat{laptopDetection, coffeeDetection}
// 			}(),
// 			InputFilter: map[string]bool{},
// 			Result: []ObjectDetection{
// 				{
// 					ClassID:     0,
// 					Confidence:  9,
// 					ClassName:   "laptop",
// 					BoundingBox: image.Rect(1, 1, 3, 3),
// 				},
// 				{
// 					ClassID:     1,
// 					Confidence:  9,
// 					ClassName:   "coffee",
// 					BoundingBox: image.Rect(-1, 1, 1, 3),
// 				},
// 			},
// 		},
// 		{
// 			Name:       "Incorrect input layer provided",
// 			InputFrame: gocv.NewMatWithSize(2, 2, gocv.MatTypeCV32F),
// 			InputOutputs: func() []gocv.Mat {
// 				return []gocv.Mat{gocv.NewMatWithSize(1, 10, gocv.MatTypeCV16S)}
// 			}(),
// 			ExpectError: true,
// 		},
// 		{
// 			Name:       "Result was filtered",
// 			InputFrame: gocv.NewMatWithSize(2, 2, gocv.MatTypeCV32F),
// 			InputOutputs: func() []gocv.Mat {
// 				coffeeDetection := coffeeDetection()

// 				return []gocv.Mat{coffeeDetection}
// 			}(),
// 			InputFilter: map[string]bool{"coffee": true},
// 			Result:      []ObjectDetection{},
// 		},
// 		{
// 			Name:       "Confidence not high enough",
// 			InputFrame: gocv.NewMatWithSize(2, 2, gocv.MatTypeCV32F),
// 			InputOutputs: func() []gocv.Mat {
// 				coffeeDetection := coffeeDetection()

// 				return []gocv.Mat{coffeeDetection}
// 			}(),
// 			InputConfidenceThreshHold: 999,
// 			InputFilter:               map[string]bool{"coffee": true},
// 			Result:                    []ObjectDetection{},
// 		},
// 		{
// 			Name:       "Filter overlapping frame",
// 			InputFrame: gocv.NewMatWithSize(2, 2, gocv.MatTypeCV32F),
// 			InputOutputs: func() []gocv.Mat {
// 				coffeeDetection1 := coffeeDetection()
// 				coffeeDetection2 := coffeeDetection()
// 				coffeeDetection2.SetFloatAt(0, 6, 10)
// 				return []gocv.Mat{coffeeDetection1, coffeeDetection2}
// 			}(),
// 			InputFilter: map[string]bool{},
// 			Result: []ObjectDetection{
// 				{
// 					ClassID:     1,
// 					Confidence:  10,
// 					ClassName:   "coffee",
// 					BoundingBox: image.Rect(-1, 1, 1, 3),
// 				},
// 			},
// 		},
// 	}
// 	for _, test := range tests {
// 		s.Run(test.Name, func() {
// 			y := &yoloNet{
// 				cocoNames:           []string{"laptop", "coffee"},
// 				confidenceThreshold: test.InputConfidenceThreshHold,
// 			}
// 			detections, err := y.processOutputs(test.InputFrame, test.InputOutputs, test.InputFilter)
// 			if test.ExpectError {
// 				s.Error(err)
// 			} else {
// 				s.Require().NoError(err)
// 			}
// 			s.Equal(test.Result, detections)
// 		})
// 	}
// }

// FIXME
// func (s *YoloTestSuite) TestGetDetections() {
// 	tests := []struct {
// 		Name                      string
// 		InputFrame                gocv.Mat
// 		InputConfidenceThreshHold float32
// 		Result                    []ObjectDetection
// 		ExpectError               bool
// 		SetupNeuralNetMock        func() *mocks.MockNeuralNet
// 		Panics                    bool
// 	}{
// 		{
// 			Name:       "Get successful detection",
// 			InputFrame: gocv.NewMatWithSize(2, 2, gocv.MatTypeCV32F),
// 			SetupNeuralNetMock: func() *mocks.MockNeuralNet {
// 				controller := gomock.NewController(s.T())
// 				neuralNetMock := mocks.NewMockNeuralNet(controller)
// 				neuralNetMock.EXPECT().SetInput(gomock.Any(), "data").Times(1)

// 				neuralNetMock.EXPECT().ForwardLayers(gomock.Any()).Return(func() []gocv.Mat {
// 					laptopDetection := laptopDetection()
// 					coffeeDetection := coffeeDetection()

// 					return []gocv.Mat{laptopDetection, coffeeDetection}
// 				}()).Times(1)
// 				return neuralNetMock
// 			},
// 			Result: []ObjectDetection{
// 				{
// 					ClassID:     0,
// 					Confidence:  9,
// 					ClassName:   "laptop",
// 					BoundingBox: image.Rect(1, 1, 3, 3),
// 				},
// 				{
// 					ClassID:     1,
// 					Confidence:  9,
// 					ClassName:   "coffee",
// 					BoundingBox: image.Rect(-1, 1, 1, 3),
// 				},
// 			},
// 		},
// 		{
// 			Name:       "Incorrect input layer provided",
// 			InputFrame: gocv.NewMatWithSize(2, 2, gocv.MatTypeCV32F),
// 			SetupNeuralNetMock: func() *mocks.MockNeuralNet {
// 				controller := gomock.NewController(s.T())
// 				neuralNetMock := mocks.NewMockNeuralNet(controller)
// 				neuralNetMock.EXPECT().SetInput(gomock.Any(), "data").Times(1)
// 				neuralNetMock.EXPECT().ForwardLayers(gomock.Any()).Return([]gocv.Mat{gocv.NewMatWithSize(1, 10, gocv.MatTypeCV16S)}).Times(1)
// 				return neuralNetMock
// 			},
// 			ExpectError: true,
// 		},
// 	}
// 	for _, test := range tests {
// 		s.Run(test.Name, func() {
// 			y := &yoloNet{
// 				cocoNames:           []string{"laptop", "coffee"},
// 				confidenceThreshold: test.InputConfidenceThreshHold,
// 				net:                 test.SetupNeuralNetMock(),
// 			}
// 			if test.Panics {
// 				s.Panics(func() { y.GetDetections(test.InputFrame) })
// 			} else {
// 				detections, err := y.GetDetections(test.InputFrame)
// 				if test.ExpectError {
// 					s.Error(err)
// 				} else {
// 					s.Require().NoError(err)
// 				}
// 				s.Equal(test.Result, detections)
// 			}
// 		})
// 	}
// }

func laptopDetection() gocv.Mat {
	laptopDetection := gocv.NewMatWithSize(1, 10, gocv.MatTypeCV32F)
	laptopDetection.SetFloatAt(0, 0, 1)
	laptopDetection.SetFloatAt(0, 1, 1)
	laptopDetection.SetFloatAt(0, 2, 1)
	laptopDetection.SetFloatAt(0, 3, 1)
	// Index for laptop == 5
	laptopDetection.SetFloatAt(0, 5, 9)
	return laptopDetection
}

func coffeeDetection() gocv.Mat {
	coffeeDetection := gocv.NewMatWithSize(1, 10, gocv.MatTypeCV32F)
	coffeeDetection.SetFloatAt(0, 1, 1)
	coffeeDetection.SetFloatAt(0, 2, 1)
	coffeeDetection.SetFloatAt(0, 3, 1)
	coffeeDetection.SetFloatAt(0, 3, 1)
	// Index for coffee == 6
	coffeeDetection.SetFloatAt(0, 6, 9)
	return coffeeDetection
}

func ExampleNewNet() {
	yolov5Model := path.Join(os.Getenv("GOPATH"), "src/github.com/wimspaargaren/yolov5/data/yolov5/yolov5s.onnx")
	cocoNamesPath := path.Join(os.Getenv("GOPATH"), "src/github.com/wimspaargaren/data/yolov5/coco.names")

	yolonet, err := NewNet(yolov5Model, cocoNamesPath)
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

	imagePath := path.Join(os.Getenv("GOPATH"), "src/github.com/wimspaargaren/yolov5/data/example_images/bird.jpg")
	frame := gocv.IMRead(imagePath, gocv.IMReadColor)

	detections, err := yolonet.GetDetections(frame)
	if err != nil {
		log.WithError(err).Fatal("unable to retrieve predictions")
	}

	DrawDetections(&frame, detections)

	window := gocv.NewWindow("Result Window")
	defer func() {
		err := window.Close()
		if err != nil {
			log.WithError(err).Error("unable to close window")
		}
	}()

	window.IMShow(frame)
	window.ResizeWindow(872, 585)

	window.WaitKey(10000000000)
}

func ExampleNewNetWithConfig() {
	yolov5Model := path.Join(os.Getenv("GOPATH"), "src/github.com/wimspaargaren/yolov5/data/yolov5/yolov5s.onnx")
	cocoNamesPath := path.Join(os.Getenv("GOPATH"), "src/github.com/wimspaargaren/data/yolov5/coco.names")

	conf := DefaultConfig()
	// Set the neural net to use CUDA
	conf.NetBackendType = gocv.NetBackendCUDA
	conf.NetTargetType = gocv.NetTargetCUDA

	yolonet, err := NewNetWithConfig(yolov5Model, cocoNamesPath, conf)
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

	// ...
}
