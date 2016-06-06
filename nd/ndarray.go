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

//Only shape is changed.
func (self *NdArray) Reshape(newShape ...int) *NdArray {
	if len(self.data) != util.ProductOfIntSlice(newShape) {
		panic(fmt.Errorf("New shape length: %v != original shape length: %v ", util.ProductOfIntSlice(newShape), util.ProductOfIntSlice(self.shape)))
	}

	return &NdArray{
		shape: newShape,
		data:  self.data,
	}
}

//If new shape is bigger than original, then new memory is allocated,
//otherwise only shape is changed.
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

//Get element in the specified pose, which length is same as self.shape .
func (self *NdArray) Get(poses ...int) float64 {
	if self.NDims() != len(poses) {
		panic("shape error")
	}
	return self.data[self.startPosOfIx(poses)]
}

//Set element in the specified pose of self.
func (self *NdArray) Set(v float64, poses ...int) {
	if self.NDims() != len(poses) {
		panic("shape error")
	}
	pos := self.startPosOfIx(poses)
	self.data[pos] = v
}

func (self *NdArray) SumAll() float64 {
	return util.SumOfFloat64Slice(self.data)
}

func (self *NdArray) Shape() []int {
	return self.shape
}

func (self *NdArray) T() *NdArray {
	nshape := make([]int, len(self.shape))
	for i, v := range self.shape {
		nshape[len(self.shape)-i-1] = v
	}

	if len(self.shape) != 2 {
		panic(fmt.Errorf("Only matrix support Transpose"))
	}

	tn := Zeros(nshape...)
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

//Only for matrix(dimentions = 2)
func (self *NdArray) NthRow(i int) *NdArray {
	tn := Zeros(self.shape[1])
	for j := 0; j < self.shape[1]; j++ {
		tn.Set(self.Get(i, j), j)
	}

	return tn
}

//Only for matrix(dimentions = 2)
func (self *NdArray) NthCol(j int) *NdArray {
	tn := Zeros(self.shape[0])
	for i := 0; i < self.shape[0]; i++ {
		tn.Set(self.Get(i, j), i)
	}

	return tn
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
	return self.Ravel()
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

//calculate index of poses in self.data
func (self *NdArray) startPosOfIx(poses []int) int {
	if len(poses) > len(self.shape) {
		panic("shape error")
	}

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
