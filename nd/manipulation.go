package nd

import (
	"math"
	"sort"

	"github.com/ledao/ndarray/util"
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
		rowIndexs := make([]int, 0, A.shape[0])
		colIndexs := make([]int, 0, A.shape[1])
		for i := 0; i < A.shape[0]; i++ {
			for j := 0; j < A.shape[1]; j++ {
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
	if !util.EqualOfIntSlice(src.shape, dst.shape) {
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

//Stack arrays in sequence vertically (rowwise).
func VStack(nds ...*NdArray) *NdArray {
	tn := Empty()
	var col = 0
	if len(nds) == 0 {
		return Empty()
	} else {
		if len(nds[0].shape) == 1 {
			col = nds[0].shape[0]
		} else {
			col = nds[0].shape[1]
		}
		for i := range nds {
			if len(nds[i].shape) == 1 {
				if col != nds[i].shape[0] {
					panic("shape error")
				} else {
					tn.PushEles(nds[i].data...)
				}
			} else if len(nds[i].shape) == 2 {
				if col != nds[i].shape[1] {
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

		row = nds[0].shape[0]

		for i := range nds {
			if len(nds[i].shape) == 1 {
				if row != nds[i].shape[0] {
					panic("shape error")
				} else {
					tn.PushEles(nds[i].data...)
				}
			} else if len(nds[i].shape) == 2 {
				if row != nds[i].shape[0] {
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
		for i := 0; i < a.shape[0]; i++ {
			sort.Float64s(a.data[i*a.shape[1] : (i+1)*a.shape[1]])
		}
		return a
	}

	panic("shape error")
}

//Split an array into multiple sub-arrays horizontally(column-wise).
func HSplit(a *NdArray) []*NdArray {
	if a.NDims() == 2 {
		nds := make([]*NdArray, a.shape[0])
		for i := range nds {
			data := make([]float64, a.shape[1])
			copy(data, a.data[i*a.shape[1]:(i+1)*a.shape[1]])
			nds[i] = &NdArray{
				shape: []int{a.shape[1]},
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
		nds := make([]*NdArray, a.shape[1])
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
	if util.All(func() []bool {
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
			tn := Zeros(a.shape[0], a.shape[1]*reps[0])
			for r := 0; r < reps[0]; r++ {
				for i := 0; i < a.shape[0]; i++ {
					for j := 0; j < a.shape[1]; j++ {
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
			tn := Zeros(a.shape[0]*reps[0], a.shape[1]*reps[1])
			for ri := 0; ri < reps[0]; ri++ {
				for rj := 0; rj < reps[1]; rj++ {
					for i := 0; i < a.shape[0]; i++ {
						for j := 0; j < a.shape[1]; j++ {
							tn.Set(a.Get(i, j), i+ri*a.shape[0], j+rj*a.shape[1])
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

//Returns the indices of the maximum values along an axis.
func ArgMax(a *NdArray) []int {
	if a.NDims() == 1 {
		maxValue := math.Inf(-1)
		maxIndex := -1
		for i, v := range a.data {
			if v > maxValue {
				maxValue = v
				maxIndex = i
			}
		}

		return []int{maxIndex}
	}

	if a.NDims() == 2 {
		maxIndexs := make([]int, a.shape[0])
		for i := 0; i < a.shape[0]; i++ {
			maxValue := math.Inf(-1)
			maxIndex := -1
			for j := 0; j < a.shape[1]; j++ {
				if a.Get(i, j) > maxValue {
					maxValue = a.Get(i, j)
					maxIndex = j
				}
			}
			maxIndexs[i] = maxIndex
		}

		return maxIndexs
	}

	panic("shape error")
}

//Returns the indices of the minimum values along an axis.
func ArgMin(a *NdArray) []int {
	if a.NDims() == 1 {
		minValue := math.Inf(1)
		minIndex := -1
		for i, v := range a.data {
			if v < minValue {
				minValue = v
				minIndex = i
			}
		}

		return []int{minIndex}
	}

	if a.NDims() == 2 {
		minIndexs := make([]int, a.shape[0])
		for i := 0; i < a.shape[0]; i++ {
			minValue := math.Inf(1)
			minIndex := -1
			for j := 0; j < a.shape[1]; j++ {
				if a.Get(i, j) < minValue {
					minValue = a.Get(i, j)
					minIndex = j
				}
			}
			minIndexs[i] = minIndex
		}

		return minIndexs
	}

	panic("shape error")
}

//Returntheelementsofanarraythatsatisfysomecondition.
func Extract(a *NdArray, condition func(ele float64) bool) *NdArray {
	tn := Empty()
	for _, v := range a.data {
		if condition(v) {
			tn.PushEles(v)
		}
	}
	tn = tn.Reshape(len(tn.data))

	return tn
}

//Counts the number of non-zero values in the array a.
func CountNonZero(a *NdArray) int {
	count := 0
	for _, v := range a.data {
		if math.Abs(v-1e-10) > 1e-10 {
			count += 1
		}
	}

	return count
}
