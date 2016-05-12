package nd

import (
	"math"
	"testing"
)

func TestExp(t *testing.T) {
	a := Array(1, 2, 3)
	a_exp := Exp(a)

	if !a_exp.Equals(Array(2.7182817, 7.389056, 20.085537)) {
		t.Error("Expected [2.7182817, 7.389056, 20.085537], got ", a_exp)
	}
}

func TestMap(t *testing.T) {
	a := Array(1, 2, 3)
	a_map := Map(a, func(e float64) float64 {
		return math.Exp(e)
	})

	if !a_map.Equals(Array(2.7182817, 7.389056, 20.085537)) {
		t.Error("Expected [2.7182817, 7.389056, 20.085537], got ", a_map)
	}
}

func TestNonZero(t *testing.T) {
	a := Array(1, 2, 3, 0, 0, 3, 4)
	index := NonZero(a)

	if len(index) != 1 {
		t.Error("Expected 1, got ", len(index))
	}
	if !EqualOfIntSlice(index[0], []int{0, 1, 2, 5, 6}) {
		t.Error("Expected [0,1,2,5,6}, got ", index[0])
	}

	a = Array(1, 2, 3, 0, 0, 3).Reshape(2, 3)
	index = NonZero(a)

	if len(index) != 2 {
		t.Error("Expected 2, got ", len(index))
	}

	if !EqualOfIntSlice(index[0], []int{0, 0, 0, 1}) {
		t.Error("Expected [0,0,0,1], got ", index[0])
	}

	if !EqualOfIntSlice(index[1], []int{0, 1, 2, 2}) {
		t.Error("Expected [0,1,2,2], got ", index[1])
	}
}

func TestCopyTo(t *testing.T) {
	a := Arange(4).Reshape(2, 2)
	b := Zeros(2, 2)

	CopyTo(a, b)

	a.Set(10, 0)

	if !b.Equals(Arange(4).Reshape(2, 2)) {
		t.Error("Expected [[0,1],[2,3]], got ", b)
	}

	defer func() {
		res := recover()
		if res != "src shape != dst shape" {
			t.Error("Expected 'src shape != dst shape', got ", res)
		}
	}()

	b = Zeros(4)
	CopyTo(a, b)
}

func TestRavel(t *testing.T) {
	a := Arange(4).Reshape(2, 2)
	b := Ravel(a)
	a.Set(10, 0)

	if !b.Equals(Arange(4)) {
		t.Error("Expected [0,1,2,3], got ", b)
	}
}

func TestAtleast2D(t *testing.T) {
	a := Arange(4)
	b := Atleast2D(a)

	if !b.Equals(Arange(4).Reshape(4, 1)) {
		t.Error("Expected [[0,1,2,3]], got ", b)
	}

	b = Atleast2D(a.Reshape(2, 2))

	if !b.Equals(Arange(4).Reshape(2, 2)) {
		t.Error("Expected [[0,1],[2,3]], got ", b)
	}
}

func TestAtleast3D(t *testing.T) {
	a := Arange(4)
	b := Atleast3D(a)

	if !b.Equals(Arange(4).Reshape(4, 1, 1)) {
		t.Error("Expected [[[0,1,2,3]]], got ", b)
	}

	b = Atleast3D(a.Reshape(2, 2))

	if !b.Equals(Arange(4).Reshape(2, 2, 1)) {
		t.Error("Expected [[[0,1],[2,3]]], got ", b)
	}
}

func TestMean(t *testing.T) {
	a := Arange(4)
	mean := Mean(a).Get(0)

	if mean != 1.5 {
		t.Error("Expected   , got ", mean)
	}
}
