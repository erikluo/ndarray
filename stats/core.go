package stats

import (
	"math"

	"github.com/ledao/ndarray/nd"
	"github.com/ledao/ndarray/util"
)

func SumAll(a *nd.NdArray) float64 {
	return util.SumOfFloat64Slice(a.Values())
}

//if a's shape is [m],
//    then the mean of all elements will be returned in a ndarray;
//if a's shape is [m, n],
//    then the mean of each row will be returned in a 1darray;
func Mean(a *nd.NdArray) *nd.NdArray {
	if len(a.Shape()) == 1 {
		sum := 0.0
		for _, v := range a.Values() {
			sum += v
		}
		mean := sum / float64(len(a.Values()))
		return nd.Array(mean)
	}

	if len(a.Shape()) == 2 {
		means := make([]float64, a.Shape()[0])
		for i := 0; i < a.Shape()[0]; i++ {
			sum := 0.0
			for j := 0; j < a.Shape()[1]; j++ {
				sum += a.Get(i, j)
			}
			means[i] = sum / float64(a.Shape()[1])
		}
		return nd.Array(means...)
	}

	panic("shape error")
}

//if a's shape is [m],
//    then the sum of all elements will be returned in a ndarray;
//if a's shape is [m, n],
//    then the sum of each row will be returned in a 1d array;
func Sum(a *nd.NdArray) *nd.NdArray {
	if len(a.Shape()) == 1 {
		sum := 0.0
		for _, v := range a.Values() {
			sum += v
		}
		return nd.Array(sum)
	}

	if len(a.Shape()) == 2 {
		sums := make([]float64, a.Shape()[0])
		for i := 0; i < a.Shape()[0]; i++ {
			sum := 0.0
			for j := 0; j < a.Shape()[1]; j++ {
				sum += a.Get(i, j)
			}
			sums[i] = sum
		}
		return nd.Array(sums...)
	}

	panic("shape error")
}

//if a's shape is [m],
//    then the std of all elements will be returned in a ndarray;
//if a's shape is [m, n],
//    then the std of each row will be returned in a 1d array;
func Std(a *nd.NdArray) *nd.NdArray {
	if len(a.Shape()) == 1 {
		mean := Mean(a).Get(0)
		sum := 0.0
		for _, v := range a.Values() {
			sum += (v - mean) * (v - mean)
		}
		std := math.Sqrt(sum / float64(len(a.Values())))
		return nd.Array(std)
	}

	if len(a.Shape()) == 2 {
		stds := make([]float64, a.Shape()[0])
		for i := 0; i < a.Shape()[0]; i++ {
			mean := Mean(a.NthRow(i)).Get(0)
			sum := 0.0
			for j := 0; j < a.Shape()[1]; j++ {
				sum += (a.Get(i, j) - mean) * (a.Get(i, j) - mean)
			}

			stds[i] = math.Sqrt(sum / float64(a.Shape()[1]))
		}
		return nd.Array(stds...)
	}

	panic("shape error")
}

//if a's shape is [m],
//    then the variance of all elements will be returned in a ndarray;
//if a's shape is [m, n],
//    then the variance of each row will be returned in a 1d array;
func Var(a *nd.NdArray) *nd.NdArray {
	if len(a.Shape()) == 1 {
		mean := Mean(a).Get(0)
		sum := 0.0
		for _, v := range a.Values() {
			sum += (v - mean) * (v - mean)
		}
		std := sum / float64(len(a.Values()))
		return nd.Array(std)
	}

	if len(a.Shape()) == 2 {
		stds := make([]float64, a.Shape()[0])
		for i := 0; i < a.Shape()[0]; i++ {
			mean := Mean(a.NthRow(i)).Get(0)
			sum := 0.0
			for j := 0; j < a.Shape()[1]; j++ {
				sum += (a.Get(i, j) - mean) * (a.Get(i, j) - mean)
			}

			stds[i] = sum / float64(a.Shape()[1])
		}
		return nd.Array(stds...)
	}

	panic("shape error")
}

//if a's shape is [m],
//    then the max value of all elements will be returned in a ndarray;
//if a's shape is [m, n],
//    then the max value of each row will be returned in a 1d array;
func Max(a *nd.NdArray) *nd.NdArray {
	if len(a.Shape()) == 1 {
		max := math.Inf(-1)
		for _, v := range a.Values() {
			if v > max {
				max = v
			}
		}
		return nd.Array(max)
	}

	if len(a.Shape()) == 2 {
		maxs := make([]float64, a.Shape()[0])
		for i := 0; i < a.Shape()[0]; i++ {
			max := math.Inf(-1)
			for j := 0; j < a.Shape()[1]; j++ {
				if a.Get(i, j) > max {
					max = a.Get(i, j)
				}
			}
			maxs[i] = max
		}

		return nd.Array(maxs...)
	}

	panic("shape error")
}

//if a's shape is [m],
//    then the min value of all elements will be returned in a ndarray;
//if a's shape is [m, n],
//    then the min value of each row will be returned in a 1d array;
func Min(a *nd.NdArray) *nd.NdArray {
	if len(a.Shape()) == 1 {
		min := math.Inf(1)
		for _, v := range a.Values() {
			if v < min {
				min = v
			}
		}
		return nd.Array(min)
	}

	if len(a.Shape()) == 2 {
		mins := make([]float64, a.Shape()[0])
		for i := 0; i < a.Shape()[0]; i++ {
			min := math.Inf(1)
			for j := 0; j < a.Shape()[1]; j++ {
				if a.Get(i, j) < min {
					min = a.Get(i, j)
				}
			}
			mins[i] = min
		}

		return nd.Array(mins...)
	}

	panic("shape error")
}
