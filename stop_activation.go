package optistop

import (
	"encoding/json"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
)

func init() {
	var s StopActivation
	serializer.RegisterTypedDeserializer(s.SerializerType(), DeserializeStopActivation)
}

// The StopActivation is basically a "temporal softmax",
// where each component of the output vector corresponds
// to a timestep, and each timestep's stopping probability
// is limited by the complement of the probability that an
// earlier timestep chose to stop.
type StopActivation struct {
	TimeCount int
}

// DeserializeStopActivation deserializes a StopActivation.
func DeserializeStopActivation(d []byte) (*StopActivation, error) {
	var res StopActivation
	if err := json.Unmarshal(d, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

// Apply applies the activation to one or more time series
// vectors.
//
// The output is a vector of log-probabilities.
func (s *StopActivation) Apply(in autofunc.Result) autofunc.Result {
	split := autofunc.Split(len(in.Output())/s.TimeCount, in)
	var res []autofunc.Result
	for _, x := range split {
		res = append(res, s.applySingle(x))
	}
	return autofunc.Concat(res...)
}

// ApplyR is like Apply, but for RResults.
func (s *StopActivation) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	split := autofunc.SplitR(len(in.Output())/s.TimeCount, in)
	var res []autofunc.RResult
	for _, x := range split {
		res = append(res, s.applySingleR(x))
	}
	return autofunc.ConcatR(res...)
}

// SerializerType returns the unique ID used to serialize
// a StopActivation with the serializer package.
func (s *StopActivation) SerializerType() string {
	return "github.com/unixpickle/optistop.StopActivation"
}

// Serialize serializes the StopActivation.
func (s *StopActivation) Serialize() ([]byte, error) {
	return json.Marshal(s)
}

func (s *StopActivation) applySingle(in autofunc.Result) autofunc.Result {
	res := &activationResult{CondProbs: neuralnet.Sigmoid{}.Apply(in)}
	cumulative := 1.0
	for _, x := range res.CondProbs.Output() {
		res.Cumulative = append(res.Cumulative, cumulative)
		res.OutVec = append(res.OutVec, cumulative*x)
		cumulative *= (1 - x)
	}
	return res
}

func (s *StopActivation) applySingleR(in autofunc.RResult) autofunc.RResult {
	ins := autofunc.SplitR(s.TimeCount, in)
	rv := autofunc.RVector{}
	zero := autofunc.NewRVariable(&autofunc.Variable{Vector: []float64{0}}, rv)
	one := autofunc.NewRVariable(&autofunc.Variable{Vector: []float64{1}}, rv)
	res := autofunc.FoldR(one, ins, func(s, energy autofunc.RResult) autofunc.RResult {
		probRemaining := autofunc.SliceR(s, 0, 1)
		lastProbs := autofunc.SliceR(s, 1, len(s.Output()))

		sm := neuralnet.SoftmaxLayer{}
		probs := sm.ApplyR(rv, autofunc.ConcatR(zero, energy))
		probContinue := autofunc.SliceR(probs, 0, 1)
		probStop := autofunc.SliceR(probs, 1, 2)
		stay := autofunc.MulR(probRemaining, probContinue)
		stop := autofunc.MulR(probRemaining, probStop)
		return autofunc.ConcatR(stay, lastProbs, stop)
	})
	return autofunc.SliceR(res, 1, len(res.Output()))
}

type activationResult struct {
	CondProbs  autofunc.Result
	Cumulative linalg.Vector
	OutVec     linalg.Vector
}

func (a *activationResult) Output() linalg.Vector {
	return a.OutVec
}

func (a *activationResult) Constant(g autofunc.Gradient) bool {
	return a.CondProbs.Constant(g)
}

func (a *activationResult) PropagateGradient(u linalg.Vector, g autofunc.Gradient) {
	if a.Constant(g) {
		return
	}
	condGrad := make(linalg.Vector, len(u))
	var nextCumGrad float64
	for i := len(u) - 1; i >= 0; i-- {
		condProb := a.CondProbs.Output()[i]
		condGrad[i] = a.Cumulative[i] * (u[i] - nextCumGrad)
		nextCumGrad = condProb*u[i] + nextCumGrad*(1-condProb)
	}
	a.CondProbs.PropagateGradient(condGrad, g)
}
