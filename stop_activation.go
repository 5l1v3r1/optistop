package optistop

import (
	"encoding/json"

	"github.com/unixpickle/autofunc"
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
	ins := autofunc.Split(s.TimeCount, in)
	one := &autofunc.Variable{Vector: []float64{1}}
	zero := &autofunc.Variable{Vector: []float64{0}}
	res := autofunc.Fold(one, ins, func(s, energy autofunc.Result) autofunc.Result {
		probRemaining := autofunc.Slice(s, 0, 1)
		lastProbs := autofunc.Slice(s, 1, len(s.Output()))

		sm := neuralnet.SoftmaxLayer{}
		probs := sm.Apply(autofunc.Concat(zero, energy))
		probContinue := autofunc.Slice(probs, 0, 1)
		probStop := autofunc.Slice(probs, 1, 2)
		stay := autofunc.Mul(probRemaining, probContinue)
		stop := autofunc.Mul(probRemaining, probStop)
		return autofunc.Concat(stay, lastProbs, stop)
	})
	return autofunc.Slice(res, 1, len(res.Output()))
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
