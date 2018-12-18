package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

func main() {
	labelData, err := ioutil.ReadFile("./data/train-labels.idx1-ubyte")
	check(err)
	labelData = labelData[8:]
	// fmt.Println(labelData)
	// outputLabel(labelData)

	imageData, err := ioutil.ReadFile("./data/train-images.idx3-ubyte")
	// outputImage(imageData)
	check(err)
	imageCount := int(binary.BigEndian.Uint32(imageData[4:8]))
	imageRows := int(binary.BigEndian.Uint32(imageData[8:12]))
	imageCols := int(binary.BigEndian.Uint32(imageData[12:16]))
	imageData = imageData[16:]
	fmt.Println(imageCount, imageRows, imageCols)

	dim := imageRows * imageCols
	hiddenNode := 144
	alphaV := 0.2
	alphaW := 0.2
	alphaX := 0.1
	trainingTimes := 1
	readOld := true

	inputLayer := make([]float64, dim)
	hiddenLayer := make([]float64, hiddenNode)
	outputLayer := make([]float64, 10)

	iTohWeight := make([][]float64, hiddenNode)

	hTooWeight := make([][]float64, 10)

	if readOld {
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
	} else {
		rand.Seed(time.Now().Unix())
		for i := 0; i < hiddenNode; i++ {
			iTohWeight[i] = make([]float64, dim)
			for j := 0; j < dim; j++ {
				iTohWeight[i][j] = rand.Float64()
				//iTohWeight[i][j] = 0
			}
		}
		for i := 0; i < 10; i++ {
			hTooWeight[i] = make([]float64, hiddenNode)
			for j := 0; j < hiddenNode; j++ {
				hTooWeight[i][j] = rand.Float64()
				//hTooWeight[i][j] = 0
			}
		}
	}

	// fmt.Println(iTohWeight)
	fmt.Println(hTooWeight)

	// 训练次数
	for t := 0; t < trainingTimes; t++ {
		mse := 0.0
		// 读取图片次数
		for i := 0; i < imageCount; i++ {
			// 初始化输入层
			imagePart := imageData[i*dim : (i+1)*dim]
			for j := 0; j < dim; j++ {
				if imagePart[j] != 0 {
					inputLayer[j] = 1
				} else {
					inputLayer[j] = 0
				}
				// inputLayer[j] = float64(imagePart[j] ==)
			}
			// 计算隐含层
			for j := 0; j < hiddenNode; j++ {
				hiddenLayer[j] = 0
				for k := 0; k < dim; k++ {
					hiddenLayer[j] += iTohWeight[j][k] * inputLayer[k]
				}
				// fmt.Println(hiddenLayer[j])
				// 单极性Sigmoid函数
				hiddenLayer[j] = 1.0 / (1 + math.Exp(-hiddenLayer[j]*alphaX))
				// fmt.Println(hiddenLayer[j])
			}
			// 计算输出层
			for j := 0; j < 10; j++ {
				outputLayer[j] = 0
				for k := 0; k < hiddenNode; k++ {
					outputLayer[j] += hTooWeight[j][k] * hiddenLayer[k]
				}
				// 单极性Sigmoid函数
				outputLayer[j] = 1.0 / (1 + math.Exp(-outputLayer[j]*alphaX))
			}
			// fmt.Println("hidden: ",hiddenLayer)
			// 计算预期输出
			expectOutput := make([]float64, 10)
			expectOutput[labelData[i]] = 1.0
			// fmt.Println(expectOutput)
			// fmt.Println(outputLayer)
			// 计算误差
			delta := make([]float64, 10)
			for j := 0; j < 10; j++ {
				delta[j] = (expectOutput[j] - outputLayer[j]) * outputLayer[j] * (1 - outputLayer[j])
				for k := 0; k < hiddenNode; k++ {
					hTooWeight[j][k] += alphaV * delta[j] * hiddenLayer[k]
				}
			}
			// fmt.Println(delta)
			// fmt.Println(hTooWeight)

			for j := 0; j < hiddenNode; j++ {
				deltaY := 0.0
				for k := 0; k < 10; k++ {
					deltaY += delta[k] * hTooWeight[k][j]
				}
				deltaY *= (1 - hiddenLayer[j]) * hiddenLayer[j]

				for k := 0; k < dim; k++ {
					iTohWeight[j][k] += alphaW * deltaY * inputLayer[k]
				}
			}

			// 均方差
			sum := 0.0
			for j := 0; j < 10; j++ {
				sum += (expectOutput[j] - outputLayer[j]) * (expectOutput[j] - outputLayer[j])
			}
			mse += sum / (10 * float64(imageCount))
		}
		fmt.Println(mse)
	}
	// 保存参数
	var sb strings.Builder
	for i := 0; i < hiddenNode; i++ {
		for j := 0; j < dim; j++ {
			sb.WriteString(strconv.FormatFloat(iTohWeight[i][j], 'g', 20, 64))
			sb.WriteByte('\n')
		}
	}
	err = ioutil.WriteFile("./data/V.txt", []byte(sb.String()), 0777)
	check(err)
	sb.Reset()
	// fmt.Println(iTohWeight)
	for i := 0; i < 10; i++ {
		for j := 0; j < hiddenNode; j++ {
			sb.WriteString(strconv.FormatFloat(hTooWeight[i][j], 'g', 20, 64))
			sb.WriteByte('\n')
		}
	}
	err = ioutil.WriteFile("./data/W.txt", []byte(sb.String()), 0777)
	// fmt.Println(hTooWeight)

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
	for i := 0; i < testImageCount; i++ {
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
	}
	fmt.Println("Recognition rate: ", float64(successCase)/float64(testImageCount))

}

func outputLabel(labelData []byte) {
	var labelStrBuild strings.Builder
	for i := 8; i < len(labelData); i++ {
		labelStrBuild.WriteByte('0' + labelData[i])
		labelStrBuild.WriteByte(',')
	}
	err := ioutil.WriteFile("./data/label-text.txt", []byte(labelStrBuild.String()), 777)
	check(err)

}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
