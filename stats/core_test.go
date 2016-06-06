package stats

import (
	"testing"

	"github.com/ledao/ndarray/nd"
)

func TestSumAll(t *testing.T) {
	arr := nd.Array(1, 2, 3, 4, 5, 6, 7, 8).Reshape(2, 4)

	if SumAll(arr) != 36 {
		t.Error("Expected 36, got ", SumAll(arr))
	}
}
func TestMean(t *testing.T) {
	a := nd.Arange(4)
	mean := Mean(a).Get(0)

	if mean != 1.5 {
		t.Error("Expected 1.5, got ", mean)
	}

	a = nd.Arange(4).Reshape(2, 2)
	means := Mean(a)

	if !means.Equals(nd.Array(0.5, 2.5)) {
		t.Error("Expected [0.5, 2.5], got ", means)
	}

	defer func() {
		p := recover()
		if p != "shape error" {
			t.Error("Expected 'shape error', got ", p)
		}
	}()

	a = nd.Arange(8).Reshape(2, 2, 2)
	Mean(a)
}

func TestSum(t *testing.T) {
	a := nd.Arange(4)
	sum := Sum(a).Get(0)

	if sum != 6 {
		t.Error("Expected 6, got ", sum)
	}

	a = nd.Arange(4).Reshape(2, 2)
	sums := Sum(a)

	if !sums.Equals(nd.Array(1, 5)) {
		t.Error("Expected [1, 5], got ", sums)
	}

	defer func() {
		p := recover()
		if p != "shape error" {
			t.Error("Expected 'shape error', got ", p)
		}
	}()
	a = nd.Arange(8).Reshape(2, 2, 2)
	Sum(a)
}

func TestStd(t *testing.T) {
	mat := nd.Array(2, 3, 1, 4)
	std := Std(mat).Get(0)

	if std != 1.118033988749895 {
		t.Error("expected 1.118033988749895, got ", std)
	}

	mat = mat.Reshape(2, 2)
	stds := Std(mat)

	if !stds.Equals(nd.Array(0.5, 1.5)) {
		t.Error("Expected [0.5, 1.5], got ", stds)
	}

	defer func() {
		p := recover()
		if p != "shape error" {
			t.Error("Expected 'shape error', got ", p)
		}
	}()

	Std(nd.Arange(8).Reshape(2, 2, 2))
}

func TestVar(t *testing.T) {
	mat := nd.Array(2, 3, 1, 4)
	vars := Var(mat).Get(0)

	if vars != 1.25 {
		t.Error("expected 1.25, got ", vars)
	}

	mat = mat.Reshape(2, 2)
	vars_ := Var(mat)

	if !vars_.Equals(nd.Array(0.25, 2.25)) {
		t.Error("Expected [0.25, 2.25], got ", vars_)
	}

	defer func() {
		p := recover()
		if p != "shape error" {
			t.Error("Expected 'shape error', got ", p)
		}
	}()

	Var(nd.Arange(8).Reshape(2, 2, 2))
}

func TestMax(t *testing.T) {
	a := nd.Arange(4)
	max := Max(a).Get(0)

	if max != 3 {
		t.Error("Expected 3, got ", max)
	}

	a = nd.Arange(4).Reshape(2, 2)
	maxs := Max(a)

	if !maxs.Equals(nd.Array(1, 3)) {
		t.Error("Expected [1,3], got ", maxs)
	}

	defer func() {
		p := recover()
		if p != "shape error" {
			t.Error("Expected 'shape error', got ", p)
		}
	}()

	Max(nd.Arange(8).Reshape(2, 2, 2))
}

func TestMin(t *testing.T) {
	a := nd.Arange(4)
	min := Min(a).Get(0)

	if min != 0 {
		t.Error("Expected 0, got ", min)
	}

	a = nd.Arange(4).Reshape(2, 2)
	mins := Min(a)

	if !mins.Equals(nd.Array(0, 2)) {
		t.Error("Expected [0,2], got ", mins)
	}

	defer func() {
		p := recover()
		if p != "shape error" {
			t.Error("Expected 'shape error', got ", p)
		}
	}()

	Min(nd.Arange(8).Reshape(2, 2, 2))
}
