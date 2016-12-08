package main

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/optistop"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

type Gradienter struct {
	Block      rnn.Block
	Activation *optistop.StopActivation
}

func (g *Gradienter) Gradient(s sgd.SampleSet) autofunc.Gradient {
	cost := g.Cost(s)
	grad := autofunc.NewGradient(g.Block.(sgd.Learner).Parameters())
	cost.PropagateGradient([]float64{1}, grad)
	return grad
}

func (g *Gradienter) Cost(s sgd.SampleSet) autofunc.Result {
	f := rnn.BlockSeqFunc{B: g.Block}
	inSeqs := make([][]linalg.Vector, s.Len())
	var out linalg.Vector
	for i := range inSeqs {
		sample := s.GetSample(i).(seqtoseq.Sample)
		inSeqs[i] = sample.Inputs
		for _, o := range sample.Outputs {
			out = append(out, o...)
		}
	}

	res := f.ApplySeqs(seqfunc.ConstResult(inSeqs))
	joined := seqfunc.ConcatAll(res)
	activated := g.Activation.Apply(joined)
	activated = autofunc.Exp{}.Apply(activated)
	return autofunc.Scale(neuralnet.DotCost{}.Cost(out, activated), 1/float64(s.Len()))
}
