package nd

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/ledao/ndarray/util"
)

//The same as numpy ndarray
type NdArray struct {
	shape []int
	data  []float64
}

func Zeros(shape ...int) *NdArray {
	if len(shape) == 0 {
		panic(fmt.Errorf("poses length: %v = 0 ", len(shape)))
	}
	data := make([]float64, util.ProductOfIntSlice(shape))
	return &NdArray{
		shape: shape,
		data:  data,
	}
}

func Ones(shape ...int) *NdArray {
	tn := Zeros(shape...)
	for i := range tn.data {
		tn.data[i] = 1
	}

	return tn
}

func Eye(n int) *NdArray {
	tn := Zeros(n, n)
	for i := 0; i < n; i++ {
		tn.Set(1, i, i)
	}

	return tn
}

func Empty() *NdArray {
	return &NdArray{
		shape: []int{},
		data:  []float64{},
	}
}

func Array(datas ...float64) *NdArray {
	shape := []int{len(datas)}
	data := make([]float64, len(datas))
	copy(data, datas)
	return &NdArray{
		shape: shape,
		data:  data,
	}
}

//this function has different consequence according to params' length
// params' length: 1
//    generate a []float64 of 0 : params[0], and params[0] is excluded;
// params' length: 2
//    generate a []float64 of params[0] : params[1], and params[1] is excluded;
// params' length: 3
//    generate a []float64 of params[0] : params[1], and params[1] is excluded, and the step is params[2].
//    when params[2] > 0, the []float64 is assending, and params[2] < 0, the []float64 is descending.
func Arange(params ...int) *NdArray {
	var tr []float64
	switch len(params) {
	case 1:
		tr = make([]float64, params[0])
		for i := 0; i < params[0]; i++ {
			tr[i] = float64(i)
		}
	case 2:
		tr = make([]float64, params[1]-params[0])
		for i := 0; i < params[1]-params[0]; i++ {
			tr[i] = float64(params[0] + i)
		}
	case 3:
		if params[2] > 0 {
			tr = make([]float64, int(math.Ceil(float64(params[1]-params[0])/float64(params[2]))))
			for i := 0; i < len(tr); i++ {
				tr[i] = float64(params[0] + i*params[2])
			}
		} else if params[2] < 0 {
			tr = make([]float64, int(math.Ceil(float64(params[0]-params[1])/float64(-params[2]))))
			for i := 0; i < len(tr); i++ {
				tr[i] = float64(params[0] + i*params[2])
			}
		} else {
			panic(fmt.Errorf("params[2] == 0"))
		}
	default:
		panic(fmt.Errorf("you can only put there parameters."))
	}

	return &NdArray{
		shape: []int{len(tr)},
		data:  tr,
	}
}

//没有重新分配新的数据空间，新的NdArray指向同样的data.
//判断self的data的个数是否与newShape的个数相同，如果不同，则抛出异常
func (self *NdArray) Reshape(newShape ...int) *NdArray {
	if len(self.data) != util.ProductOfIntSlice(newShape) {
		panic(fmt.Errorf("New shape length: %v != original shape length: %v ", util.ProductOfIntSlice(newShape), util.ProductOfIntSlice(self.shape)))
	}

	return &NdArray{
		shape: newShape,
		data:  self.data,
	}
}

//没有重新分配新的数据空间，新的NdArray指向同样的data.
//判断self的data的个数是否与newShape的个数相同，如果不同，则会
//根据newShape的大小决定是否对self.data进行截断，或者在末尾补0.
func (self *NdArray) ReshapeUnsafe(newShape ...int) *NdArray {
	if len(self.data) >= util.ProductOfIntSlice(newShape) {
		return &NdArray{
			shape: newShape,
			data:  self.data[0:util.ProductOfIntSlice(newShape)],
		}
	} else {
		newData := make([]float64, util.ProductOfIntSlice(newShape))
		copy(newData, self.data)
		return &NdArray{
			shape: newShape,
			data:  newData,
		}
	}
}

func (self *NdArray) Get(poses ...int) float64 {
	return self.Ix(poses...).Value()
}

func (self *NdArray) Set(v float64, poses ...int) {
	pos := self.startPosOfIx(poses)
	self.data[pos] = v
}

func (self *NdArray) SumOfAll() float64 {
	return util.SumOfFloat64Slice(self.data)
}

func (self *NdArray) Shape() []int {
	return self.shape
}

func (self *NdArray) T() *NdArray {
	if len(self.shape) != 2 {
		panic(fmt.Errorf("Only matrix support Transpose"))
	}

	tn := Zeros(self.shape[1], self.shape[0])
	for i := 0; i < self.shape[0]; i++ {
		for j := 0; j < self.shape[1]; j++ {
			tn.Set(self.Get(i, j), j, i)
		}
	}

	return tn
}

func (self *NdArray) Equals(that *NdArray) bool {
	if len(self.shape) != len(that.shape) {
		return false
	} else {
		for i := range self.shape {
			if self.shape[i] != that.shape[i] {
				return false
			}
		}
		for i := range self.data {
			if math.Abs(self.data[i]-that.data[i]) > 1e-5 {
				return false
			}
		}
	}
	return true
}

func (self *NdArray) Clone() *NdArray {
	tn := Zeros(self.shape...)
	copy(tn.data, self.data)
	return tn
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

func (self *NdArray) NthRow(i int) *NdArray {
	tn := Zeros(self.shape[1])
	for j := 0; j < self.shape[1]; j++ {
		tn.Set(self.Get(i, j), j)
	}

	return tn
}

func (self *NdArray) NthCol(j int) *NdArray {
	tn := Zeros(self.shape[0])
	for i := 0; i < self.shape[0]; i++ {
		tn.Set(self.Get(i, j), i)
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
			data:  []float64{self.Mul(that).SumOfAll()},
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

type poseValue struct {
	pose  []int
	value float64
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
				tn.Set(self.Get(i, j)+that.Get(i), i, j)
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
				tn.Set(self.Get(i, j)-that.Get(i), i, j)
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
				tn.Set(self.Get(i, j)*that.Get(i), i, j)
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
				tn.Set(self.Get(i, j)/that.Get(i), i, j)
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

func (self *NdArray) SetRow(row *NdArray, i int) *NdArray {
	if len(self.shape) == 2 && len(row.data) == self.shape[1] {
		for j := 0; j < self.shape[1]; j++ {
			self.Set(row.Get(j), i, j)
		}

		return self
	}

	panic("shape error")
}

func (self *NdArray) SetCol(col *NdArray, j int) *NdArray {
	if len(self.shape) == 2 && len(col.data) == self.shape[0] {
		for i := 0; i < self.shape[0]; i++ {
			self.Set(col.Get(i), i, j)
		}

		return self
	}

	panic("shape error")
}

//在原来NdArray的基础上在添加eles元素，不论形状是否合适,
//将添加eles后的self返回。
func (self *NdArray) PushEles(eles ...float64) *NdArray {
	self.data = append(self.data, eles...)

	return self
}

func (self *NdArray) Size() int {
	return len(self.data)
}

//选出is所指定的行，形成新的NdArray。
//self必须是matrix，is的范围不能超过self的行范围
func (self *NdArray) GetRows(is ...int) *NdArray {
	if len(self.shape) != 2 {
		panic("shape error")
	}

	rows := Empty()
	for _, i := range is {
		rows.PushEles(self.NthRow(i).data...)
	}

	return rows.Reshape(len(is), self.shape[1])
}

//选出js所指定的列，形成新的NdArray。
//self必须是matrix，js的范围不能超过self的列范围
func (self *NdArray) GetCols(js ...int) *NdArray {
	if len(self.shape) != 2 {
		panic("shape error")
	}

	cols := Empty()
	for _, j := range js {
		cols.PushEles(self.NthCol(j).data...)
	}

	return cols.Reshape(len(js), self.shape[0]).T()
}

func (self *NdArray) GetEles(is ...int) *NdArray {
	eles := Zeros(len(is))
	for c, i := range is {
		eles.Set(self.Get(i), c)
	}

	return eles
}

func (self *NdArray) Values() []float64 {
	return self.data
}

func (self *NdArray) Flat() *NdArray {
	return Ravel(self)
}

func (self *NdArray) IsEmpty() bool {
	if len(self.shape) == 0 || len(self.data) == 0 {
		return true
	} else {
		return false
	}
}

func (self *NdArray) NDims() int {
	return len(self.shape)
}

func (self *NdArray) startPosOfIx(poses []int) int {
	pos := 0
	for i := 0; i < len(poses); i++ {
		pos += poses[i] * util.ProductOfIntSlice(self.shape[i+1:len(self.shape)])
	}
	return pos
}

func (self *NdArray) Ix(poses ...int) *NdArray {
	if len(poses) == len(self.shape) {
		pos := self.startPosOfIx(poses)
		return &NdArray{
			shape: []int{1},
			data:  []float64{self.data[pos]},
		}
	}

	if len(poses) <= len(self.shape) {
		newShape := make([]int, len(self.shape)-len(poses))
		copy(newShape, self.shape[len(poses):len(self.shape)])
		pos := self.startPosOfIx(poses)
		data := make([]float64, util.ProductOfIntSlice(self.shape[len(poses):len(self.shape)]))
		copy(data, self.data[pos:pos+util.ProductOfIntSlice(self.shape[len(poses):len(self.shape)])])
		return &NdArray{
			shape: newShape,
			data:  data,
		}
	}

	panic("shape error")
}

func (self *NdArray) makeStr() string {
	if self.NDims() == 1 {
		lineEles := make([]string, 0, self.shape[0])
		for i := 0; i < self.shape[0]; i++ {
			lineEles = append(lineEles, strconv.FormatFloat(self.data[i], 'f', -1, 32))
		}
		return "[" + strings.Join(lineEles, ", ") + "]"
	} else {
		eles := make([]string, self.shape[0])
		for i := 0; i < self.shape[0]; i++ {
			eles[i] = self.Ix(i).makeStr()
		}
		return "[" + strings.Join(eles, ", \n") + "]"
	}
}

func (self *NdArray) String() string {
	return fmt.Sprintf("ndarray<%v>\n(%v)", self.shape, self.makeStr())
}

func (self *NdArray) Value() float64 {
	if self.IsEmpty() {
		panic("empty ndarray")
	}
	return self.data[0]
}
