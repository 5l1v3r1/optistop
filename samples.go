package optistop

import (
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

// A SampleSet is an sgd.SampleSet of Sample objects.
type SampleSet []Sample

// Len returns the number of samples.
func (s SampleSet) Len() int {
	return len(s)
}

// GetSample returns the given sample's seqtoseq.Sample
// representation for supervised learning.
func (s SampleSet) GetSample(i int) interface{} {
	return s[i].SeqToSeq()
}

// Swap swaps two samples.
func (s SampleSet) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

// Copy creates a shallow copy of the SampleSet.
func (s SampleSet) Copy() sgd.SampleSet {
	return append(SampleSet{}, s...)
}

// Subset slices the SampleSet.
func (s SampleSet) Subset(i, j int) sgd.SampleSet {
	return s[i:j]
}

// Hash produces a hash for the given sample.
func (s SampleSet) Hash(i int) []byte {
	return s[i].SeqToSeq().Hash()
}

// A Sample represents an optimal stopping scenario.
// The "optimal" choice is represented in the permutation
// as 0, and higher values indicate less-optimal choices.
//
// Essentially, a sample can be thought of as an ordered
// list of candidates, and the goal is to select the best
// candidate only knowing the ordering of the previous
// candidates.
type Sample []int

// SeqToSeq generates a sample for supervised learning.
// The input sequence indicates whether or not each
// candidate was the best up to that point.
// The output sequence indicates only the optimal choice.
func (s Sample) SeqToSeq() seqtoseq.Sample {
	var res seqtoseq.Sample
	best := len(s)
	for _, x := range s {
		if x < best {
			best = x
			res.Inputs = append(res.Inputs, []float64{1})
		} else {
			res.Inputs = append(res.Inputs, []float64{0})
		}
		if x == 0 {
			res.Outputs = append(res.Outputs, []float64{1})
		} else {
			res.Outputs = append(res.Outputs, []float64{0})
		}
	}
	return res
}
