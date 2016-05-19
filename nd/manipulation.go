package nd

import (
	"math"
	"sort"
)

func Exp(A *NdArray) *NdArray {
	tn := Zeros(A.shape...)

	for i := range tn.data {
		tn.data[i] = math.Exp(A.data[i])
	}

	return tn
}

func Map(A *NdArray, f func(e float64) float64) *NdArray {
	tn := Zeros(A.shape...)
	for i := range tn.data {
		tn.data[i] = f(A.data[i])
	}

	return tn
}

//如果A是一维的NdArray，则返回一个长度1的切片，里面是A中不等于零的元素的下标。
//如果A是2维的NdArray，则返回一个长度2的切片，里面第一个切片是A中不等于0的元素的行下标，第二个切片是A中不等于0的元素的列下标。
func NonZero(A *NdArray) [][]int {
	if len(A.shape) == 1 {
		nonZeroindexs := make([]int, 0, A.shape[0])
		for i := range A.data {
			if math.Abs(A.data[i]-0.0) > 1e-5 {
				nonZeroindexs = append(nonZeroindexs, i)
			}
		}
		return [][]int{nonZeroindexs}
	}

	if len(A.shape) == 2 {
		rowIndexs := make([]int, 0, A.Rows())
		colIndexs := make([]int, 0, A.Cols())
		for i := 0; i < A.Rows(); i++ {
			for j := 0; j < A.Cols(); j++ {
				if math.Abs(A.Get(i, j)-0.0) > 1e-5 {
					rowIndexs = append(rowIndexs, i)
					colIndexs = append(colIndexs, j)
				}
			}
		}

		return [][]int{rowIndexs, colIndexs}
	}

	panic("shape error")
}

//Copies values from one array to another.
//Raises a "src shape != dst shape" panic if the src shape != dst shape.
func CopyTo(src *NdArray, dst *NdArray) {
	if !EqualOfIntSlice(src.shape, dst.shape) {
		panic("src shape != dst shape")
	}
	copy(dst.data, src.data)
}

//Return a contiguous ﬂattened 1-d array.
//A 1-D array,containing the elements of the input, is returned. A copy is made.
func Ravel(a *NdArray) *NdArray {
	data := make([]float64, len(a.data))
	copy(data, a.data)

	return &NdArray{
		shape: []int{len(data)},
		data:  data,
	}
}

//View inputs as arrays with at least two dimensions.
func Atleast2D(a *NdArray) *NdArray {
	if len(a.shape) == 1 {
		return a.Reshape(a.shape[0], 1)
	} else {
		return a
	}
}

//View inputs as arrays with at least three dimensions.
func Atleast3D(a *NdArray) *NdArray {
	if len(a.shape) == 1 {
		return a.Reshape(a.shape[0], 1, 1)
	} else if len(a.shape) == 2 {
		return a.Reshape(a.shape[0], a.shape[1], 1)
	} else {
		return a
	}
}

//if a's shape is [m],
//    then the mean of all elements will be returned in a ndarray;
//if a's shape is [m, n],
//    then the mean of each row will be returned in a 1darray;
func Mean(a *NdArray) *NdArray {
	if len(a.shape) == 1 {
		sum := 0.0
		for _, v := range a.data {
			sum += v
		}
		mean := sum / float64(len(a.data))
		return &NdArray{
			shape: []int{1},
			data:  []float64{mean},
		}
	}

	if len(a.shape) == 2 {
		means := make([]float64, a.Rows())
		for i := 0; i < a.Rows(); i++ {
			sum := 0.0
			for j := 0; j < a.Cols(); j++ {
				sum += a.Get(i, j)
			}
			means[i] = sum / float64(a.Cols())
		}
		return &NdArray{
			shape: []int{a.Rows()},
			data:  means,
		}
	}

	panic("shape error")
}

//if a's shape is [m],
//    then the sum of all elements will be returned in a ndarray;
//if a's shape is [m, n],
//    then the sum of each row will be returned in a 1d array;
func Sum(a *NdArray) *NdArray {
	if len(a.shape) == 1 {
		sum := 0.0
		for _, v := range a.data {
			sum += v
		}
		return &NdArray{
			shape: []int{1},
			data:  []float64{sum},
		}
	}

	if len(a.shape) == 2 {
		sums := make([]float64, a.Rows())
		for i := 0; i < a.Rows(); i++ {
			sum := 0.0
			for j := 0; j < a.Cols(); j++ {
				sum += a.Get(i, j)
			}
			sums[i] = sum
		}
		return &NdArray{
			shape: []int{a.Rows()},
			data:  sums,
		}
	}

	panic("shape error")
}

//if a's shape is [m],
//    then the std of all elements will be returned in a ndarray;
//if a's shape is [m, n],
//    then the std of each row will be returned in a 1d array;
func Std(a *NdArray) *NdArray {
	if len(a.shape) == 1 {
		mean := Mean(a).Get(0)
		sum := 0.0
		for _, v := range a.data {
			sum += (v - mean) * (v - mean)
		}
		std := math.Sqrt(sum / float64(len(a.data)))
		return &NdArray{
			shape: []int{1},
			data:  []float64{std},
		}
	}

	if len(a.shape) == 2 {
		stds := make([]float64, a.Rows())
		for i := 0; i < a.Rows(); i++ {
			mean := Mean(a.NthRow(i)).Get(0)
			sum := 0.0
			for j := 0; j < a.Cols(); j++ {
				sum += (a.Get(i, j) - mean) * (a.Get(i, j) - mean)
			}

			stds[i] = math.Sqrt(sum / float64(a.Cols()))
		}
		return &NdArray{
			shape: []int{a.Rows()},
			data:  stds,
		}
	}

	panic("shape error")
}

//if a's shape is [m],
//    then the variance of all elements will be returned in a ndarray;
//if a's shape is [m, n],
//    then the variance of each row will be returned in a 1d array;
func Var(a *NdArray) *NdArray {
	if len(a.shape) == 1 {
		mean := Mean(a).Get(0)
		sum := 0.0
		for _, v := range a.data {
			sum += (v - mean) * (v - mean)
		}
		std := sum / float64(len(a.data))
		return &NdArray{
			shape: []int{1},
			data:  []float64{std},
		}
	}

	if len(a.shape) == 2 {
		stds := make([]float64, a.Rows())
		for i := 0; i < a.Rows(); i++ {
			mean := Mean(a.NthRow(i)).Get(0)
			sum := 0.0
			for j := 0; j < a.Cols(); j++ {
				sum += (a.Get(i, j) - mean) * (a.Get(i, j) - mean)
			}

			stds[i] = sum / float64(a.Cols())
		}
		return &NdArray{
			shape: []int{a.Rows()},
			data:  stds,
		}
	}

	panic("shape error")
}

//if a's shape is [m],
//    then the max value of all elements will be returned in a ndarray;
//if a's shape is [m, n],
//    then the max value of each row will be returned in a 1d array;
func Max(a *NdArray) *NdArray {
	if len(a.shape) == 1 {
		max := math.Inf(-1)
		for _, v := range a.data {
			if v > max {
				max = v
			}
		}
		return &NdArray{
			shape: []int{1},
			data:  []float64{max},
		}
	}

	if len(a.shape) == 2 {
		maxs := make([]float64, a.Rows())
		for i := 0; i < a.Rows(); i++ {
			max := math.Inf(-1)
			for j := 0; j < a.Cols(); j++ {
				if a.Get(i, j) > max {
					max = a.Get(i, j)
				}
			}
			maxs[i] = max
		}

		return &NdArray{
			shape: []int{a.Rows()},
			data:  maxs,
		}
	}

	panic("shape error")
}

//if a's shape is [m],
//    then the min value of all elements will be returned in a ndarray;
//if a's shape is [m, n],
//    then the min value of each row will be returned in a 1d array;
func Min(a *NdArray) *NdArray {
	if len(a.shape) == 1 {
		min := math.Inf(1)
		for _, v := range a.data {
			if v < min {
				min = v
			}
		}
		return &NdArray{
			shape: []int{1},
			data:  []float64{min},
		}
	}

	if len(a.shape) == 2 {
		mins := make([]float64, a.Rows())
		for i := 0; i < a.Rows(); i++ {
			min := math.Inf(1)
			for j := 0; j < a.Cols(); j++ {
				if a.Get(i, j) < min {
					min = a.Get(i, j)
				}
			}
			mins[i] = min
		}

		return &NdArray{
			shape: []int{a.Rows()},
			data:  mins,
		}
	}

	panic("shape error")
}

//Stack arrays in sequence vertically (rowwise).
func VStack(nds ...*NdArray) *NdArray {
	tn := Empty()
	var col = 0
	if len(nds) == 0 {
		return Empty()
	} else {

		if len(nds[0].shape) == 1 {
			col = nds[0].Rows()
		} else {
			col = nds[0].Cols()
		}
		for i := range nds {
			if len(nds[i].shape) == 1 {
				if col != nds[i].Rows() {
					panic("shape error")
				} else {
					tn.PushEles(nds[i].data...)
				}
			} else if len(nds[i].shape) == 2 {
				if col != nds[i].Cols() {
					panic("shape error")
				} else {
					tn.PushEles(nds[i].data...)
				}
			} else {
				panic("shape error")
			}
		}
	}
	rows := len(tn.data) / col

	return tn.Reshape(rows, col)
}

//Stack arrays in sequence horizontally (columnwise)
func HStack(nds ...*NdArray) *NdArray {
	tn := Empty()
	var row = 0
	if len(nds) == 0 {
		return Empty()
	} else {
		if nds[0].IsEmpty() {
			panic("shape error")
		}

		row = nds[0].Rows()

		for i := range nds {
			if len(nds[i].shape) == 1 {
				if row != nds[i].Rows() {
					panic("shape error")
				} else {
					tn.PushEles(nds[i].data...)
				}
			} else if len(nds[i].shape) == 2 {
				if row != nds[i].Rows() {
					panic("shape error")
				} else {
					tn.PushEles(nds[i].T().data...)
				}
			} else {
				panic("shape error")
			}
		}
	}
	col := len(tn.data) / row

	return tn.Reshape(col, row).T()
}

//sort the ndarray
func Sort(a *NdArray) *NdArray {
	if a.NDims() == 1 {
		sort.Float64s(a.data)
		return a
	} else if a.NDims() == 2 {
		for i := 0; i < a.Rows(); i++ {
			sort.Float64s(a.data[i*a.Cols() : (i+1)*a.Cols()])
		}
		return a
	}

	panic("shape error")
}

//Split an array into multiple sub-arrays horizontally(column-wise).
func HSplit(a *NdArray) []*NdArray {
	if a.NDims() == 2 {
		nds := make([]*NdArray, a.Rows())
		for i := range nds {
			data := make([]float64, a.Cols())
			copy(data, a.data[i*a.Cols():(i+1)*a.Cols()])
			nds[i] = &NdArray{
				shape: []int{a.Cols()},
				data:  data,
			}
		}

		return nds
	}

	panic("shape error")
}

//Split an array into multiple sub-arrays vertically(row-wise).
func VSplit(a *NdArray) []*NdArray {
	if a.NDims() == 2 {
		nds := make([]*NdArray, a.Cols())
		for j := range nds {
			nds[j] = a.NthCol(j)
		}
		return nds
	}

	panic("shape error")
}

// Construct an array by repeating A the number of times given by reps.
func Tile(a *NdArray, reps ...int) *NdArray {
	if len(reps) == 0 {
		return a.Clone()
	}

	d := len(reps)
	if All(func() []bool {
		bools := make([]bool, d)
		for i := range bools {
			if reps[i] == 1 {
				bools[i] = true
			} else {
				bools[i] = false
			}
		}
		return bools
	}()...) {
		return a.Clone()
	}

	if d == 1 {
		if a.NDims() == 1 {
			tn := Empty()
			for r := 0; r < reps[0]; r++ {
				tn.PushEles(a.data...)
			}
			return tn.Reshape(a.Size() * reps[0])
		} else if a.NDims() == 2 {
			tn := Zeros(a.Rows(), a.Cols()*reps[0])
			for r := 0; r < reps[0]; r++ {
				for i := 0; i < a.Rows(); i++ {
					for j := 0; j < a.Cols(); j++ {
						tn.Set(a.Get(i, j), i, j+r*reps[0])
					}
				}
			}
			return tn
		} else {
			panic("shape error")
		}
	}

	if d == 2 {
		if a.NDims() == 2 {
			tn := Zeros(a.Rows()*reps[0], a.Cols()*reps[1])
			for ri := 0; ri < reps[0]; ri++ {
				for rj := 0; rj < reps[1]; rj++ {
					for i := 0; i < a.Rows(); i++ {
						for j := 0; j < a.Cols(); j++ {
							tn.Set(a.Get(i, j), i+ri*a.Rows(), j+rj*a.Cols())
						}
					}
				}
			}
			return tn
		} else {
			panic("shape error")
		}
	}

	panic("shape error")
}

func Unique(a *NdArray) []float64 {
	if a.IsEmpty() {
		return []float64{}
	}

	uniqueEles := make([]float64, 0, a.Size())
	for _, v := range a.data {
		if func() bool {
			for _, uv := range uniqueEles {
				if v == uv {
					return false
				}
			}
			return true
		}() {
			uniqueEles = append(uniqueEles, v)
		}
	}

	sort.Float64s(uniqueEles)
	return uniqueEles
}
