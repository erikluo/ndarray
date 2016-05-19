package nd

import (
	"fmt"
	"testing"
)

func TestNdProductOfSlice(t *testing.T) {
	s := []int{1, 2, 3, 4, 5}

	if ProductOfIntSlice(s) != 120 {
		t.Error(fmt.Sprintf("Expected 120, got %v", ProductOfIntSlice(s)))
	}
}

func TestNdSumOfFloat64Slice(t *testing.T) {
	s := []float64{1, 2, 3, 4, 5}

	if SumOfFloat64Slice(s) != 15 {
		t.Error("Expected 15, got ", SumOfFloat64Slice(s))
	}
}

func TestEqualOfIntSlice(t *testing.T) {
	a := []int{2, 3, 4}
	b := []int{2, 3}

	if EqualOfIntSlice(a, b) != false {
		t.Error("Expecte false, got true")
	}

	if EqualOfIntSlice(a, a) != true {
		t.Error("Expected true, got false")
	}
}

func TestEqualOfFloat64Slice(t *testing.T) {
	a := []float64{2, 3, 4}
	b := []float64{2, 3}

	if EqualOfFloat64Slice(a, b) != false {
		t.Error("Expecte false, got true")
	}

	if EqualOfFloat64Slice(a, a) != true {
		t.Error("Expected true, got false")
	}
}

func TestAll(t *testing.T) {
	b := []bool{true, true, true}
	r := All(b...)

	if r != true {
		t.Error("Expected true, got ", r)
	}

	b = append(b, false)
	r = All(b...)

	if r != false {
		t.Error("Expected false, got ", r)
	}
}

func TestAny(t *testing.T) {
	b := []bool{false, false}
	r := Any(b...)

	if r != false {
		t.Error("Expected false, got ", r)
	}

	b = append(b, true)
	r = Any(b...)

	if r != true {
		t.Error("Expected true, got ", false)
	}
}
