package main

import (
	"io/ioutil"
	"strings"
)

func main() {
	// train()
	identify()
	// test()
	// outputImage()

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
