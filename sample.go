package optistop

import "math/rand"

// A Sample represents an optimal stopping scenario.
// The "optimal" choice is represented in the permutation
// as 0, and higher values indicate less-optimal choices.
//
// Essentially, a sample can be thought of as an ordered
// list of candidates, and the goal is to select the best
// candidate only knowing the ordering of the previous
// candidates.
type Sample []int

// NewSample creates a random sample.
func NewSample(size int) Sample {
	return Sample(rand.Perm(size))
}

// Comparisons creates a comparison list where each entry
// is true if that entry in the sample is better than any
// of the previous entries.
func (s Sample) Comparisons() []bool {
	lowestSeen := len(s)
	res := make([]bool, len(s))
	for i, x := range s {
		if x < lowestSeen {
			res[i] = true
			lowestSeen = x
		}
	}
	return res
}
