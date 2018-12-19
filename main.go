package main

import (
	"io/ioutil"
	"os"
	"strings"
)

func main() {
	if len(os.Args) < 2 {
		return
	}
	switch os.Args[1] {
	case "train":
		train()
	case "web":
		identify()
	case "test":
		test()
	case "output":
		outputImage()
	}
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
