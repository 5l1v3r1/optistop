package optistop

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/num-analysis/linalg"
)

func TestStopActivationOutput(t *testing.T) {
	input := &autofunc.Variable{Vector: []float64{1, -1, 2}}
	sm := autofunc.Sigmoid{}
	conditional := sm.Apply(input)
	expected := []float64{}
	remainingProb := 1.0
	for _, p := range conditional.Output() {
		expected = append(expected, remainingProb*p)
		remainingProb *= (1 - p)
	}

	sa := StopActivation{TimeCount: 3}
	actual := sa.Apply(input).Output()

	if actual.Copy().Scale(-1).Add(expected).MaxAbs() > 1e-5 {
		t.Errorf("expected %v but got %v", expected, actual)
	}
}

func TestStopActivation(t *testing.T) {
	in := &autofunc.Variable{Vector: make(linalg.Vector, 12)}
	rv := autofunc.RVector{in: make(linalg.Vector, len(in.Vector))}
	for i := range in.Vector {
		rv[in][i] = rand.NormFloat64()
		in.Vector[i] = rand.NormFloat64()
	}
	checker := &functest.RFuncChecker{
		F:     &StopActivation{TimeCount: 3},
		Vars:  []*autofunc.Variable{in},
		Input: in,
		RV:    rv,
	}
	checker.FullCheck(t)
}

func BenchmarkStopActivation(b *testing.B) {
	in := &autofunc.Variable{Vector: make(linalg.Vector, 50)}
	upstream := make(linalg.Vector, len(in.Vector))
	for i := range in.Vector {
		upstream[i] = rand.NormFloat64()
		in.Vector[i] = rand.NormFloat64()
	}
	sa := &StopActivation{TimeCount: len(in.Vector)}
	grad := autofunc.NewGradient([]*autofunc.Variable{in})
	for i := 0; i < b.N; i++ {
		sa.Apply(in).PropagateGradient(upstream.Copy(), grad)
	}
}
