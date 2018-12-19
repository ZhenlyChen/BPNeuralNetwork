package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"io"
	"io/ioutil"
	"math"
	"os"
	"strconv"
)

func outputImage() {
	dim := 28 * 28
	// 隐含层节点数
	hiddenNode := 256

	inputLayer := make([]float64, dim)
	hiddenLayer := make([]float64, hiddenNode)
	outputLayer := make([]float64, 10)

	iTohWeight := make([][]float64, hiddenNode)

	hTooWeight := make([][]float64, 10)
	vWeightData, err := os.Open("./data/V.txt")
	check(err)
	defer vWeightData.Close()
	brV := bufio.NewReader(vWeightData)
	for i := 0; i < hiddenNode; i++ {
		iTohWeight[i] = make([]float64, dim)
		for j := 0; j < dim; j++ {
			a, _, c := brV.ReadLine()
			if c == io.EOF {
				panic("")
			}
			iTohWeight[i][j], err = strconv.ParseFloat(string(a), 64)
			check(err)
		}
	}

	wWeightData, err := os.Open("./data/W.txt")
	check(err)
	defer wWeightData.Close()
	brH := bufio.NewReader(wWeightData)

	for i := 0; i < 10; i++ {
		hTooWeight[i] = make([]float64, hiddenNode)

		for j := 0; j < hiddenNode; j++ {
			a, _, c := brH.ReadLine()
			if c == io.EOF {
				panic("")
			}
			hTooWeight[i][j], err = strconv.ParseFloat(string(a), 64)
			check(err)
		}
	}

	// 测试
	testLabelData, err := ioutil.ReadFile("./data/t10k-labels.idx1-ubyte")
	check(err)
	testLabelData = testLabelData[8:]
	// fmt.Println(testLabelData)
	outputLabel(testLabelData)

	testImageData, err := ioutil.ReadFile("./data/t10k-images.idx3-ubyte")
	check(err)
	testImageCount := int(binary.BigEndian.Uint32(testImageData[4:8]))
	testImageRows := int(binary.BigEndian.Uint32(testImageData[8:12]))
	testImageCols := int(binary.BigEndian.Uint32(testImageData[12:16]))
	testImageData = testImageData[16:]
	fmt.Println(testImageCount, testImageRows, testImageCols)

	// 测试图片
	successCase := 0
	for i := 0; i < 100; i++ {
		// 初始化输入层
		imagePart := testImageData[i*testImageRows*testImageRows : (i+1)*testImageRows*testImageRows]
		// 输出图片
		//img := image.NewGray(image.Rect(0, 0, imageRows, imageCols))
		//for x := 0; x < imageRows; x++ {
		//	for y := 0; y < imageCols; y++ {
		//		img.Set(y, x, color.Gray{Y: testImageData[i*dim+x*imageRows+y]})
		//	}
		//}
		//dst, err := os.Create("./data/image" + strconv.Itoa(i) + "-" + string(testLabelData[i]+'0') + ".jpeg")
		//check(err)
		//err = jpeg.Encode(dst, img, &jpeg.Options{Quality: 100})
		//check(err)

		for j := 0; j < dim; j++ {
			if imagePart[j] != 0 {
				inputLayer[j] = 1
			} else {
				inputLayer[j] = 0
			}
		}
		// 计算隐含层
		for j := 0; j < hiddenNode; j++ {
			hiddenLayer[j] = 0
			for k := 0; k < dim; k++ {
				hiddenLayer[j] += iTohWeight[j][k] * inputLayer[k]
			}
			// 单极性Sigmoid函数
			hiddenLayer[j] = 1.0 / (1 + math.Exp(-hiddenLayer[j]))
		}
		// 计算输出层
		for j := 0; j < 10; j++ {
			outputLayer[j] = 0
			for k := 0; k < hiddenNode; k++ {
				outputLayer[j] += hTooWeight[j][k] * hiddenLayer[k]
			}
			// 单极性Sigmoid函数
			outputLayer[j] = 1.0 / (1 + math.Exp(-outputLayer[j]))
		}

		outputNumber := 0
		outputMax := 0.0
		for j := 0; j < 10; j++ {
			if outputLayer[j] > outputMax {
				outputMax = outputLayer[j]
				outputNumber = j
			}
		}
		// fmt.Println(outputLayer)
		// fmt.Println("Output: ", outputNumber, "Expect: ", testLabelData[i])
		if byte(outputNumber) == testLabelData[i] {
			successCase++
		}

		img := image.NewGray(image.Rect(0, 0, 28, 28))
		for x := 0; x < 28; x++ {
			for y := 0; y < 28; y++ {
				img.Set(y, x, color.Gray{Y: testImageData[i*dim+x*28+y]})
			}
		}
		dst, err := os.Create("./data/image" + strconv.Itoa(i) + "-" + string(testLabelData[i]+'0') + "-" + strconv.Itoa(outputNumber) + ".jpeg")
		check(err)
		err = jpeg.Encode(dst, img, &jpeg.Options{Quality: 100})
		check(err)
	}
}
