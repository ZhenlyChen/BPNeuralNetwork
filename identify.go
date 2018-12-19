package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"fmt"
	"github.com/segmentio/objconv/json"
	"image/png"
	"io"
	"io/ioutil"
	"log"
	"math"
	"net/http"
	"os"
	"strconv"
)

func identify() {

	// 初始化数据
	// 隐含层节点数
	dim := 28 * 28
	// 陡度因子，防止过饱和（推荐值0.03-0.05）
	hiddenNode := 256
	alphaX := 0.04
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

	type ImageData struct {
		Data string `json:"data"`
	}

	log.Printf("listening on %q...", 30018)
	http.HandleFunc("/identify", func(resp http.ResponseWriter, req *http.Request) {

		fmt.Println(req.RequestURI)
		body, _ := ioutil.ReadAll(req.Body)
		var imageData ImageData
		err := json.Unmarshal(body, &imageData)
		if err != nil {
			log.Fatal(resp.Write([]byte("error")))
			return
		}
		// fmt.Println(imageData.Data[22:])
		unbased, err := base64.StdEncoding.DecodeString(imageData.Data[22:])
		if err != nil {
			log.Fatal(resp.Write([]byte("error")))
			fmt.Println("Cannot decode b64")
			return
		}

		r := bytes.NewReader(unbased)
		im, err := png.Decode(r)
		if err != nil {
			log.Fatal(resp.Write([]byte("error")))
			fmt.Println("Bad png")
			return
		}

		f, err := os.OpenFile("example.png", os.O_WRONLY|os.O_CREATE, 0777)
		if err != nil {
			log.Fatal(resp.Write([]byte("error")))
			fmt.Println("Cannot open file")
			return
		}
		err = png.Encode(f, im)
		if err != nil {
			log.Fatal(resp.Write([]byte("error")))
			fmt.Println(err)
			return
		}
		f.Close()

		nf ,err := os.OpenFile("example.png", os.O_RDONLY, 0777)
		if err != nil {
			log.Fatal(resp.Write([]byte("error")))
			fmt.Println(err)
			return
		}
		image, err := png.Decode(nf)

		height:= 140

		imagePart  := make([]byte, 28 * 28)
		for x := 0; x < height;  x+=5 {
			for y := 0; y < height; y += 5 {
				pix := 0
				for kx := 0; kx < 5; kx++ {
					for ky := 0; ky < 5; ky++ {
						_, _, _, a := image.At(x+kx, y+ky).RGBA()
						if a != 0 {
							pix = 1
						}
					}
				}
				imagePart[(y / 5) * 28 + (x / 5)] = byte(pix)
			}
		}
		// fmt.Println(imagePart)
		// 初始化输入层
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

		outputNumber := 0
		outputMax := 0.0
		for j := 0; j < 10; j++ {
			if outputLayer[j] > outputMax {
				outputMax = outputLayer[j]
				outputNumber = j
			}
		}
		_, err = resp.Write([]byte(strconv.Itoa(outputNumber)))
		if err != nil {
			fmt.Println(err)
		}
		// fmt.Println(num, err)

	})
	http.Handle("/bp/", http.StripPrefix("/bp/", http.FileServer(http.Dir("./web"))))
	log.Fatal(http.ListenAndServe(":30018", nil))
}
