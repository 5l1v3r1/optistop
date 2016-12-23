package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"time"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/optistop"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const (
	Stop     = 0
	Continue = 1
)

func main() {
	rand.Seed(time.Now().UnixNano())

	var netPath string
	var sampleLen int
	var roundSize int
	var roundSteps int
	var stepSize float64
	var exploreCoeff float64
	var logSteps bool
	flag.StringVar(&netPath, "file", "out_net", "network file")
	flag.IntVar(&sampleLen, "len", 50, "number of timesteps")
	flag.IntVar(&roundSize, "round", 100, "episodes per round")
	flag.IntVar(&roundSteps, "steps", 10, "SGD steps per round")
	flag.Float64Var(&stepSize, "step", 0.001, "step size")
	flag.Float64Var(&exploreCoeff, "explore", -1, "chance of random decision")
	flag.BoolVar(&logSteps, "logsteps", false, "log SGD steps")

	flag.Parse()

	if exploreCoeff == -1 {
		// Improve the likelihood of stopping anywhere in the
		// sequences (rather than stopping very early).
		// The math is setup so that we have a 1/n probability
		// of finishing the sequence of length n, assuming that
		// the policy will never decide to stop.
		exploreCoeff = 2 * (1 - math.Pow(1/float64(sampleLen), 1/(float64(sampleLen)-1)))
	}
	log.Println("Explore coefficient:", exploreCoeff)

	model, err := readModel(netPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Read network file:", err)
		os.Exit(1)
	}

	var g sgd.Adam

	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)

	var roundIdx int
TrainLoop:
	for {
		var totalReward float64
		var comps [][]bool
		var targets [][]linalg.Vector
		for i := 0; i < roundSize; i++ {
			comp, target, rew := runEpisode(model, sampleLen, exploreCoeff)
			comps = append(comps, comp)
			targets = append(targets, target)
			totalReward += rew
		}

		log.Printf("round=%d reward=%f", roundIdx, totalReward/float64(roundSize))
		roundIdx++

		grad := autofunc.NewGradient(model.Block.(sgd.Learner).Parameters())
		for i := 0; i < roundSteps; i++ {
			select {
			case <-c:
				break TrainLoop
			default:
			}
			if i != 0 {
				grad.Zero()
			}
			cost := episodeCost(model, comps, targets)
			cost.PropagateGradient([]float64{1}, grad)
			g.Transform(grad).AddToVars(-stepSize)
			if logSteps {
				log.Printf("step=%d cost=%f", i, cost.Output()[0])
			}
		}
	}

	if err := model.Save(netPath); err != nil {
		fmt.Fprintln(os.Stderr, "Write network file:", err)
		os.Exit(1)
	}
}

func readModel(path string) (*optistop.Model, error) {
	model, err := optistop.LoadModel(path)
	if err == nil {
		return model, nil
	} else if !os.IsNotExist(err) {
		return nil, err
	}
	model = optistop.NewModel()
	params := model.Block.(sgd.Learner).Parameters()
	biases := params[len(params)-1]
	biases.Vector[Stop] = -10
	return model, nil
}

func runEpisode(m *optistop.Model, size int, explore float64) (comps []bool,
	targets []linalg.Vector, reward float64) {
	s := optistop.NewSample(size)

	r := rnn.Runner{Block: m.Block}
	var lastPrediction linalg.Vector
	for j := 1; j <= size; j++ {
		comparison := s.Comparisons()[j-1]
		input := []float64{0}
		if comparison {
			input[0] = 1
		}
		prediction := r.StepTime(input)
		maxValue, maxAction := prediction.Max()

		if lastPrediction != nil {
			target := append(linalg.Vector{}, lastPrediction...)
			target[Continue] = maxValue
			targets = append(targets, target)
		}
		lastPrediction = prediction

		if rand.Float64() < explore {
			maxAction = rand.Intn(2)
		}
		if maxAction == Stop {
			if s[j-1] == 0 {
				reward = 1.0
			}
			lastPrediction = nil
			target := append(linalg.Vector{}, prediction...)
			target[Stop] = reward
			targets = append(targets, target)
			break
		}
	}
	if lastPrediction != nil {
		target := append(linalg.Vector{}, lastPrediction...)
		target[Continue] = 0
		targets = append(targets, target)
	}
	return s.Comparisons()[:len(targets)], targets, reward
}

func episodeCost(m *optistop.Model, comps [][]bool, targets [][]linalg.Vector) autofunc.Result {
	var cost autofunc.Result
	for i, seq := range comps {
		target := targets[i]
		allPred := m.AllActionValues(seq)
		allTarget := seqfunc.ConstResult([][]linalg.Vector{target})
		c := seqfunc.AddAll(seqfunc.MapN(func(in ...autofunc.Result) autofunc.Result {
			pred := in[0]
			targ := in[1]
			return neuralnet.MeanSquaredCost{}.Cost(targ.Output(), pred)
		}, allPred, allTarget))
		if cost == nil {
			cost = c
		} else {
			cost = autofunc.Add(cost, c)
		}
	}
	return cost
}
