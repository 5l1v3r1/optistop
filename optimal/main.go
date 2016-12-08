package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/unixpickle/approb"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	var sampleLen int
	var numSamples int
	flag.IntVar(&sampleLen, "len", 50, "number of timesteps")
	flag.IntVar(&numSamples, "num", 10000, "number of trials")
	flag.Parse()

	headLen := int(float64(sampleLen)/math.E + 0.5)
	res, variance := approb.Indicator(numSamples, func() bool {
		p := rand.Perm(sampleLen)
		bestHead := sampleLen
		for _, x := range p[:headLen] {
			if x < bestHead {
				bestHead = x
			}
		}
		for i := headLen; i < sampleLen; i++ {
			if p[i] < bestHead {
				return p[i] == 0
			}
		}
		return false
	})

	fmt.Println("Success rate: ", res)
	fmt.Println("Rate variance:", variance)
}
