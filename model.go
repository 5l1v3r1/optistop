package optistop

import (
	"io/ioutil"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

// A Model decides which action to take having seen a list
// of candidate comparisons.
type Model struct {
	Block rnn.Block
}

// NewModel creates a new untrained model.
func NewModel() *Model {
	return &Model{
		Block: rnn.StackedBlock{
			rnn.NewLSTM(1, 20),
			rnn.NewLSTM(20, 30),
			rnn.NewNetworkBlock(neuralnet.Network{
				neuralnet.NewDenseLayer(30, 2),
			}, 0),
		},
	}
}

// LoadModel loads a model from a file.
func LoadModel(path string) (*Model, error) {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var res Model
	if err := serializer.DeserializeAny(data, &res.Block); err != nil {
		return nil, err
	}
	return &res, nil
}

// ActionValues approximates the expected reward of taking
// actions after seeing the sequence of comparisons (where
// true indicates that the candidate was the best one seen
// at the time).
//
// The first value in the result corresponds to choosing
// the current candidate, while the second corresponds to
// continuing.
func (m *Model) ActionValues(seq []bool) autofunc.Result {
	return seqfunc.ConcatLast(m.AllActionValues(seq))
}

// AllActionValues is like ActivationValues, but it gives
// the value function at every timestep.
func (m *Model) AllActionValues(seq []bool) seqfunc.Result {
	inSeq := make([]linalg.Vector, len(seq))
	for i, x := range seq {
		if x {
			inSeq[i] = []float64{1}
		} else {
			inSeq[i] = []float64{0}
		}
	}
	inSeqs := seqfunc.ConstResult([][]linalg.Vector{inSeq})
	sf := &rnn.BlockSeqFunc{B: m.Block}
	return sf.ApplySeqs(inSeqs)
}

// ShouldChoose decides whether or not to select the last
// candidate, given a sequence of comparisons.
func (m *Model) ShouldChoose(seq []bool) bool {
	res := m.ActionValues(seq)
	return res.Output()[0] > res.Output()[1]
}

// Save saves the model to a file.
func (m *Model) Save(path string) error {
	data, err := serializer.SerializeAny(m.Block)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(path, data, 0644)
}
