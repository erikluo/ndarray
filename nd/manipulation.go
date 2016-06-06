package nd

import (
	"fmt"
	"math"
	"sort"

	"github.com/ledao/ndarray/util"
)

func (A *NdArray) Exp() *NdArray {
	tn := Zeros(A.shape...)

	for i := range tn.data {
		tn.data[i] = math.Exp(A.data[i])
	}

	return tn
}

func (A *NdArray) Map(f func(e float64) float64) *NdArray {
	tn := Zeros(A.shape...)
	for i := range tn.data {
		tn.data[i] = f(A.data[i])
	}

	return tn
}

//如果A是一维的NdArray，则返回一个长度1的切片，里面是A中不等于零的元素的下标。
//如果A是2维的NdArray，则返回一个长度2的切片，里面第一个切片是A中不等于0的元素的行下标，第二个切片是A中不等于0的元素的列下标。
func (A *NdArray) NonZero() [][]int {
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
func (src *NdArray) CopyTo(dst *NdArray) {
	if !util.EqualOfIntSlice(src.shape, dst.shape) {
		panic("src shape != dst shape")
	}
	copy(dst.data, src.data)
}

//Return a contiguous ﬂattened 1-d array.
//A 1-D array,containing the elements of the input, is returned. A copy is made.
func (a *NdArray) Ravel() *NdArray {
	data := make([]float64, len(a.data))
	copy(data, a.data)

	return &NdArray{
		shape: []int{len(data)},
		data:  data,
	}
}

//View inputs as arrays with at least two dimensions.
func (a *NdArray) Atleast2D() *NdArray {
	if len(a.shape) == 1 {
		return a.Reshape(a.shape[0], 1)
	} else {
		return a
	}
}

//View inputs as arrays with at least three dimensions.
func (a *NdArray) Atleast3D() *NdArray {
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
func (a *NdArray) Sort() *NdArray {
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
func (a *NdArray) HSplit() []*NdArray {
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
func (a *NdArray) VSplit() []*NdArray {
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
func (a *NdArray) Tile(reps ...int) *NdArray {
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

func (a *NdArray) Unique() []float64 {
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
func (a *NdArray) ArgMax() []int {
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
func (a *NdArray) ArgMin() []int {
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
func (a *NdArray) Extract(condition func(ele float64) bool) *NdArray {
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
func (a *NdArray) CountNonZero() int {
	count := 0
	for _, v := range a.data {
		if math.Abs(v-1e-10) > 1e-10 {
			count += 1
		}
	}

	return count
}

func (self *NdArray) MulBit(that *NdArray) *NdArray {
	if !util.EqualOfIntSlice(self.shape, that.shape) {
		panic(fmt.Errorf("shape doesn't equals"))
	}
	tn := Zeros(self.shape...)
	for i := range self.data {
		tn.data[i] = self.data[i] * that.data[i]
	}

	return tn
}

func (self *NdArray) Dot(that *NdArray) *NdArray {
	//[m n] * [n h]
	if len(self.shape) == 2 && len(that.shape) == 2 && self.shape[1] == that.shape[0] {
		tn := Zeros(self.shape[0], that.shape[1])
		for i := 0; i < self.shape[0]; i++ {
			for j := 0; j < that.shape[1]; j++ {
				sum := self.NthRow(i).Dot(that.NthCol(j))
				tn.Set(sum.Get(0), i, j)
			}
		}

		return tn
	}

	// 1 [m] * [m n]
	if len(self.shape) == 1 && len(that.shape) == 2 && self.shape[0] == that.shape[0] {
		tn := Zeros(1, that.shape[1])
		for j := 0; j < that.shape[1]; j++ {
			sum := self.Dot(that.NthCol(j))
			tn.Set(sum.Get(0), 0, j)
		}

		return tn
	}

	//[m n] * [n] 1
	if len(that.shape) == 1 && len(self.shape) == 2 && self.shape[1] == that.shape[0] {
		tn := Zeros(self.shape[0], 1)
		for i := 0; i < self.shape[0]; i++ {
			sum := self.NthRow(i).Dot(that)
			tn.Set(sum.Get(0), i, 0)
		}

		return tn
	}

	//[m] * [m]
	if len(self.shape) == 1 && len(that.shape) == 1 && util.EqualOfIntSlice(self.shape, that.shape) {
		return &NdArray{
			shape: []int{1},
			data:  []float64{self.Mul(that).SumAll()},
		}
	}

	panic(fmt.Errorf("shape error"))
}

func (self *NdArray) Inv() *NdArray {
	if len(self.shape) != 2 {
		panic(fmt.Errorf("Only matrix support Transpose"))
	} else if self.shape[0] != self.shape[1] {
		panic(fmt.Sprintf("rows does not equal to cols"))
	}
	a := self.Clone()
	n := a.shape[1]
	var d float64
	for k := 0; k < n; k++ {
		d = 1.0 / a.Get(k, k)
		a.Set(d, k, k)
		for i := 0; i < n; i++ {
			if i != k {
				ki := a.Get(k, i)
				a.Set(ki*(-d), k, i)
			}
		}
		for i := 0; i < n; i++ {
			if i != k {
				ik := a.Get(i, k)
				a.Set(ik*d, i, k)
			}
		}
		for i := 0; i < n; i++ {
			if i != k {
				for j := 0; j < n; j++ {
					if j != k {
						ij := a.Get(i, j)
						ik := a.Get(i, k)
						kj := a.Get(k, j)
						a.Set(ij+ik*kj/d, i, j)
					}
				}
			}
		}
	}
	return a
}

// Calculates the determinant of the matrix
func (self *NdArray) Det() float64 {
	if len(self.shape) != 2 || self.shape[0] != self.shape[1] {
		panic(fmt.Errorf("shape error"))
	}
	matrixLength := self.shape[0]
	sums := make([]float64, matrixLength*2)
	for ii := 0; ii < len(sums); ii++ {
		sums[ii] = 1
	}

	for ii := 0; ii < matrixLength; ii++ {
		for jj := 0; jj < matrixLength; jj++ {
			if ii-jj < 0 {
				sums[matrixLength+ii-jj] *= self.Get(ii, jj)
			} else {
				sums[ii-jj] *= self.Get(ii, jj)
			}

			if ii+jj >= matrixLength {
				sums[ii+jj] *= self.Get(ii, jj)
			} else {
				sums[ii+jj+matrixLength] *= self.Get(ii, jj)
			}
		}
	}

	dim := matrixLength * 2
	if matrixLength == 2 {
		dim = 2
		matrixLength = 1
	}

	result := 0.0

	for ii := 0; ii < dim; ii++ {
		if ii >= matrixLength {
			result -= sums[ii]
		} else {
			result += sums[ii]
		}
	}
	return result
}

//if self's shape == that's shape，then bit wise addition will be token;
//if that's size is 1, then all elements of self will be added by the element of that;
//if self's shape is [m, n] and that's shape is [m, 1]，then the elements in ith row of self will be added by the ith element of that;
//if self's shape is [m, n] and that's shape is [1, n]，then the elements in jth col of self will be added by the jth element of that;
func (self *NdArray) Add(that *NdArray) *NdArray {
	//equal shape, bit wise addition
	if util.EqualOfIntSlice(self.shape, that.shape) {
		tn := Zeros(self.shape...)
		for i := range tn.data {
			tn.data[i] = self.data[i] + that.data[i]
		}

		return tn
	}

	//that's size is 1, scale addition
	if util.ProductOfIntSlice(that.shape) == 1 {
		tn := self.Clone()
		for i := range tn.data {
			tn.data[i] = tn.data[i] + that.Get(0)
		}

		return tn
	}

	// self' shape [m, n], that's shape [m, 1]
	if len(that.shape) == 2 && that.shape[1] == 1 && len(self.shape) == 2 && self.shape[0] == that.shape[0] {
		tn := Zeros(self.shape...)
		for i := 0; i < tn.shape[0]; i++ {
			for j := 0; j < tn.shape[1]; j++ {
				tn.Set(self.Get(i, j)+that.Get(i, 0), i, j)
			}
		}

		return tn
	}

	// self's shape [m, n], that's shape [1, n]
	if len(that.shape) == 2 && that.shape[0] == 1 && len(self.shape) == 2 && self.shape[1] == that.shape[1] {
		tn := Zeros(self.shape...)
		for i := 0; i < tn.shape[0]; i++ {
			for j := 0; j < tn.shape[1]; j++ {
				tn.Set(self.Get(i, j)+that.Get(0, j), i, j)
			}
		}

		return tn
	}

	panic(fmt.Errorf("shape error"))
}

//如果self的shape == that的shape，则对应元素的值相减。
//如果that是列向量(shape为[n, 1])，则self中的第i行每个元素减去that第i个元素的值。
//如果that是行向量(shape为[1, n]), 则self中的第j列每个元素减去that第j个元素的值。
//如果that数据个数为1，则该值被self的所有值减去，返回与self一样大小的nd.
func (self *NdArray) Sub(that *NdArray) *NdArray {
	if util.EqualOfIntSlice(self.shape, that.shape) {
		tn := Zeros(self.shape...)
		for i := range tn.data {
			tn.data[i] = self.data[i] - that.data[i]
		}

		return tn
	}

	if len(that.shape) == 2 && that.shape[1] == 1 && len(self.shape) == 2 && self.shape[0] == that.shape[0] {
		tn := Zeros(self.shape...)
		for i := 0; i < tn.shape[0]; i++ {
			for j := 0; j < tn.shape[1]; j++ {
				tn.Set(self.Get(i, j)-that.Get(i, 0), i, j)
			}
		}

		return tn
	}

	if len(that.shape) == 2 && that.shape[0] == 1 && len(self.shape) == 2 && self.shape[1] == that.shape[1] {
		tn := Zeros(self.shape...)
		for i := 0; i < tn.shape[0]; i++ {
			for j := 0; j < tn.shape[1]; j++ {
				tn.Set(self.Get(i, j)-that.Get(0, j), i, j)
			}
		}

		return tn
	}

	if util.ProductOfIntSlice(that.shape) == 1 {
		tn := self.Clone()
		for i := range tn.data {
			tn.data[i] = tn.data[i] - that.Get(0)
		}

		return tn
	}

	panic("shape error")
}

//如果self的shape == that的shape，则对应元素的值相乘。
//如果that是列向量(shape为[n, 1])，则self中的第i行每个元素乘以that第i个元素的值。
//如果that是行向量(shape为[1, n]), 则self中的第j列每个元素乘以that第j个元素的值。
//如果that数据个数为1，则该值被self的所有值乘，返回与self一样大小的nd.
func (self *NdArray) Mul(that *NdArray) *NdArray {
	if util.EqualOfIntSlice(self.shape, that.shape) {
		tn := Zeros(self.shape...)
		for i := range tn.data {
			tn.data[i] = self.data[i] * that.data[i]
		}

		return tn
	}

	if len(that.shape) == 2 && that.shape[1] == 1 && len(self.shape) == 2 && self.shape[0] == that.shape[0] {
		tn := Zeros(self.shape...)
		for i := 0; i < tn.shape[0]; i++ {
			for j := 0; j < tn.shape[1]; j++ {
				tn.Set(self.Get(i, j)*that.Get(i, 0), i, j)
			}
		}

		return tn
	}

	if len(that.shape) == 2 && that.shape[0] == 1 && len(self.shape) == 2 && self.shape[1] == that.shape[1] {
		tn := Zeros(self.shape...)
		for i := 0; i < tn.shape[0]; i++ {
			for j := 0; j < tn.shape[1]; j++ {
				tn.Set(self.Get(i, j)*that.Get(0, j), i, j)
			}
		}

		return tn
	}

	if util.ProductOfIntSlice(that.shape) == 1 {
		tn := self.Clone()
		for i := range tn.data {
			tn.data[i] = tn.data[i] * that.Get(0)
		}

		return tn
	}

	panic("shape error")
}

//如果self的shape == that的shape，则对应元素的值相乘。
//如果that是列向量(shape为[n, 1])，则self中的第i行每个元素乘以that第i个元素的值。
//如果that是行向量(shape为[1, n]), 则self中的第j列每个元素乘以that第j个元素的值。
//如果that数据个数为1，则该值被self的所有值乘，返回与self一样大小的nd.
func (self *NdArray) Div(that *NdArray) *NdArray {
	if util.EqualOfIntSlice(self.shape, that.shape) {
		tn := Zeros(self.shape...)
		for i := range tn.data {
			tn.data[i] = self.data[i] / that.data[i]
		}

		return tn
	}

	if len(that.shape) == 2 && that.shape[1] == 1 && len(self.shape) == 2 && self.shape[0] == that.shape[0] {
		tn := Zeros(self.shape...)
		for i := 0; i < tn.shape[0]; i++ {
			for j := 0; j < tn.shape[1]; j++ {
				tn.Set(self.Get(i, j)/that.Get(i, 0), i, j)
			}
		}

		return tn
	}

	if len(that.shape) == 2 && that.shape[0] == 1 && len(self.shape) == 2 && self.shape[1] == that.shape[1] {
		tn := Zeros(self.shape...)
		for i := 0; i < tn.shape[0]; i++ {
			for j := 0; j < tn.shape[1]; j++ {
				tn.Set(self.Get(i, j)/that.Get(0, j), i, j)
			}
		}

		return tn
	}

	if util.ProductOfIntSlice(that.shape) == 1 {
		tn := self.Clone()
		for i := range tn.data {
			tn.data[i] = tn.data[i] / that.Get(0)
		}

		return tn
	}

	panic("shape error")
}
