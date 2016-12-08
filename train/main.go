package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/optistop"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const StateSize = 40

func main() {
	rand.Seed(time.Now().UnixNano())

	var netPath string
	var sampleLen int
	var batchSize int
	var stepSize float64
	var logInterval int
	flag.StringVar(&netPath, "file", "out_net", "network file")
	flag.IntVar(&sampleLen, "len", 50, "number of timesteps")
	flag.IntVar(&logInterval, "log", 4, "log interval")
	flag.IntVar(&batchSize, "batch", 4, "batch size")
	flag.Float64Var(&stepSize, "step", 0.001, "step size")

	flag.Parse()

	block, err := readNetwork(netPath, sampleLen)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Read network:", err)
		os.Exit(1)
	}

	model := &Gradienter{
		Block:      block,
		Activation: &optistop.StopActivation{TimeCount: sampleLen},
	}
	g := &sgd.Adam{
		Gradienter: model,
	}

	samples := createSampleSet(sampleLen)

	var iter int
	var last sgd.SampleSet
	sgd.SGDMini(g, samples, stepSize, batchSize, func(batch sgd.SampleSet) bool {
		if iter%logInterval == 0 {
			var lastCost float64
			if last != nil {
				lastCost = model.Cost(last).Output()[0]
			}
			last = batch.Copy()
			thisCost := model.Cost(batch).Output()[0]
			log.Printf("iter %d: cost=%f last=%f", iter, thisCost, lastCost)
		}
		iter++
		return true
	})

	data, err := serializer.SerializeAny(block)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Serialize block error:", err)
		os.Exit(1)
	}
	if err := ioutil.WriteFile(netPath, data, 0755); err != nil {
		fmt.Fprintln(os.Stderr, "Write network file:", err)
		os.Exit(1)
	}
}

func readNetwork(path string, numSteps int) (rnn.Block, error) {
	if data, err := ioutil.ReadFile(path); err == nil {
		var res rnn.Block
		if err := serializer.DeserializeAny(data, &res); err != nil {
			return nil, err
		}
		return res, nil
	} else if !os.IsNotExist(err) {
		return nil, err
	}

	outNet := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  StateSize,
			OutputCount: 1,
		},
	}
	outNet.Randomize()

	// Set the start bias so the probabilities don't decay
	// too fast (aiming to have a 0.2 probability left over
	// for the final timestep).
	startBias := math.Log(1 - math.Pow(0.2, 1/float64(numSteps)))
	outNet[0].(*neuralnet.DenseLayer).Biases.Var.Vector[0] = startBias

	block := rnn.StackedBlock{
		rnn.NewLSTM(1, StateSize),
		rnn.NewNetworkBlock(outNet, 0),
	}
	return block, nil
}

func createSampleSet(timeCount int) sgd.SampleSet {
	var res optistop.SampleSet
	for i := 0; i < 100000; i++ {
		res = append(res, optistop.Sample(rand.Perm(timeCount)))
	}
	return res
}
