package nd

import (
	"math"
)

func ProductOfIntSlice(s []int) int {
	var p int = 1
	for i := range s {
		p *= s[i]
	}
	return p
}

func SumOfFloat64Slice(s []float64) float64 {
	var sum float64 = 0.0
	for _, v := range s {
		sum += v
	}

	return sum
}

func EqualOfIntSlice(a []int, b []int) bool {
	if len(a) != len(b) {
		return false
	} else {
		for i := range a {
			if a[i] != b[i] {
				return false
			}
		}
	}
	return true
}

func EqualOfFloat64Slice(a []float64, b []float64) bool {
	if len(a) != len(b) {
		return false
	} else {
		for i := range a {
			if math.Abs(a[i]-b[i]) > 1e-5 {
				return false
			}
		}
	}
	return true
}
