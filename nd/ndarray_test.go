package nd

import (
	"fmt"
	"testing"

	"github.com/ledao/ndarray/util"
)

func TestNdZeros(t *testing.T) {
	arr := Zeros(2, 3, 2)
	shape := []int{2, 3, 2}
	data := []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	for i := range data {
		if arr.data[i] != data[i] {
			t.Error(fmt.Sprintf("Expected %v, got %v ", data[i], arr.data[i]))
		}
	}
	for i := range shape {
		if arr.shape[i] != shape[i] {
			t.Error(fmt.Sprintf("Expected %v, got %v", shape[i], arr.shape[i]))
		}
	}
}

func TestNdOnes(t *testing.T) {
	o := Ones(2, 2)

	if !o.Equals(Array(1, 1, 1, 1).Reshape(2, 2)) {
		t.Error("Expected [[1,1],[1,1]], got ", o)
	}
}

func TestNdEye(t *testing.T) {
	e := Eye(2)

	if !e.Equals(Array(1, 0, 0, 1).Reshape(2, 2)) {
		t.Error("Expecte [[1,0], [0, 1]], got ", e)
	}
}

func TestNdEmpty(t *testing.T) {
	e := Empty()
	if len(e.shape) != 0 || len(e.data) != 0 {
		t.Error("Expected 0, got ", len(e.shape), len(e.data))
	}
}

func TestNdGet(t *testing.T) {
	arr := Array([]float64{1, 2, 3, 4, 5, 6, 7, 8}...).Reshape(1, 4, 2)

	if arr.Get(0, 1, 1) != 4 {
		t.Error("Expected 4, got ", arr.Get(0, 1, 1))
	}
	if arr.Get(0, 1, 0) != 3 {
		t.Error("Expected 3, got ", arr.Get(0, 1, 0))
	}
}

func TestNdArray(t *testing.T) {
	arr := Array([]float64{1, 2, 3, 4, 5, 6, 7, 8}...)

	if len(arr.shape) != 1 {
		t.Error("Expected 1, got ", len(arr.shape))
	}

	if arr.shape[0] != 8 {
		t.Error("Expected 8, got ", arr.shape[0])
	}
}

func TestNdArange(t *testing.T) {
	arr := Arange(4)
	if !arr.Equals(Array(0, 1, 2, 3)) {
		t.Error("Expected [0,1,2,3], got ", arr)
	}

	arr = Arange(2, 4)
	if !arr.Equals(Array(2, 3)) {
		t.Error("Expected [2,3], got ", arr)
	}

	arr = Arange(1, 4, 2)
	if !arr.Equals(Array(1, 3)) {
		t.Error("Expected [1,3], got ", arr)
	}

	arr = Arange(4, 1, -2)
	if !arr.Equals(Array(4, 2)) {
		t.Error("Expected [4,2], got ", arr)
	}
}

func TestNdReshape(t *testing.T) {
	arr := Array([]float64{1, 2, 3, 4, 5, 6, 7, 8}...)
	narr := arr.Reshape(2, 2, 2)

	if len(narr.shape) != 3 {
		t.Error("Expected 3, got ", len(arr.shape))
	}

	if len(arr.shape) != 1 {
		t.Error("Expected 1, got ", len(arr.shape))
	}
}

func TestNdReshapeUnsafe(t *testing.T) {
	arr := Array(1, 2, 3)

	narr := arr.ReshapeUnsafe(4)

	if len(narr.shape) != 1 {
		t.Error("Expected 1, got ", len(arr.shape))
	}

	if narr.shape[0] != 4 {
		t.Error("Expected 1, got ", narr.shape[0])
	}

	if !util.EqualOfFloat64Slice(narr.data, []float64{1, 2, 3, 0}) {
		t.Error("Expected [1,2,3,0], got ", narr.data)
	}

	narr = arr.ReshapeUnsafe(2, 2)

	if len(narr.shape) != 2 {
		t.Error("Expected 2, got ", len(arr.shape))
	}

	if narr.shape[0] != 2 || narr.shape[1] != 2 {
		t.Error("Expected 2, got ", narr.shape[0], narr.shape[1])
	}

	if !util.EqualOfFloat64Slice(narr.data, []float64{1, 2, 3, 0}) {
		t.Error("Expected [1,2,3,0], got ", narr.data)
	}

	narr = arr.ReshapeUnsafe(2)

	if len(narr.shape) != 1 {
		t.Error("Expected 1, got ", len(arr.shape))
	}

	if narr.shape[0] != 2 {
		t.Error("Expected 2, got ", narr.shape[0])
	}

	if !util.EqualOfFloat64Slice(narr.data, []float64{1, 2}) {
		t.Error("Expected [1,2], got ", narr.data)
	}
}

func TestNdSet(t *testing.T) {
	arr := Array([]float64{1, 2, 3, 4, 5, 6, 7, 8}...).Reshape(2, 4)
	arr.Set(10, 0, 2)

	if arr.Get(0, 2) != 10 {
		t.Error("Expected 10, got ", arr.Get(0, 2))
	}
}

func TestNdShape(t *testing.T) {
	arr := Array([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}...).Reshape(3, 3)

	shape := arr.Shape()

	if shape[0] != 3 || shape[1] != 3 {
		t.Errorf("Expecte 3, 3 , got %v, %v ", shape[0], shape[1])
	}

	shape[0] = 2

	if arr.shape[0] != 2 {
		t.Error("Expected 2, got ", arr.shape[0])
	}
}

func TestNdT_Equals(t *testing.T) {
	arr := Array([]float64{1, 2, 3, 4}...).Reshape(2, 2)
	tn := arr.T()

	if tn.Equals(Array([]float64{1, 3, 2, 4}...).Reshape(2, 2)) != true {
		t.Error("Expected true, got false")
	}

	if tn.Equals(Array([]float64{1, 3, 2, 4}...)) != false {
		t.Error("Expected false, got true")
	}
}

func TestNdClone(t *testing.T) {
	arr := Array([]float64{1, 2, 3, 4}...).Reshape(2, 2)
	tc := arr.Clone()

	if tc.Equals(arr) != true {
		t.Error("Expected true, got false")
	}

	arr.Set(10, 0, 0)

	if tc.Equals(arr) != false {
		t.Error("Expected false, got true")
	}
}

func TestNdNthRow(t *testing.T) {
	a1 := Array(2, 3, 4, 5, 6, 7).Reshape(2, 3)
	r0 := a1.NthRow(0)

	if !r0.Equals(Array(2, 3, 4)) {
		t.Error("Expected [2,3,4], got ", r0)
	}
}

func TestNdNthCol(t *testing.T) {
	a1 := Array(2, 3, 4, 5, 6, 7).Reshape(2, 3)
	c0 := a1.NthCol(0)

	if !c0.Equals(Array(2, 5)) {
		t.Error("Expected [2,3,4], got ", c0)
	}
}

func TestNdSetRow(t *testing.T) {
	a := Array(1, 2, 3, 4, 5, 6).Reshape(2, 3)
	b := Array(1, 2, 3)
	c := a.SetRow(b, 1)

	if !c.Equals(Array(1, 2, 3, 1, 2, 3).Reshape(2, 3)) || !a.Equals(Array(1, 2, 3, 1, 2, 3).Reshape(2, 3)) {
		t.Error("Expected [[1,2,3],[1,2,3]], got ", c)
	}
}

func TestNdSetCol(t *testing.T) {
	a := Array(1, 2, 3, 4, 5, 6).Reshape(2, 3)
	b := Array(1, 2)
	c := a.SetCol(b, 1)

	if !c.Equals(Array(1, 1, 3, 4, 2, 6).Reshape(2, 3)) || !a.Equals(Array(1, 1, 3, 4, 2, 6).Reshape(2, 3)) {
		t.Error("Expected [[1,1,3], [4,2,6]], got ", c)
	}
}

func TestNdPushEles(t *testing.T) {
	a := Array(1, 2, 3)

	a = a.PushEles(4, 5, 6)

	if len(a.shape) != 1 {
		t.Error("Expected 1, got ", len(a.shape))
	}

	if a.shape[0] != 3 {
		t.Error("Expected 3, got ", a.shape[0])
	}

	if !util.EqualOfFloat64Slice(a.data, []float64{1, 2, 3, 4, 5, 6}) {
		t.Error("Expected [1,2,3,4,5,6], got ", a.data)
	}
}

func TestNdSize(t *testing.T) {
	a := Array(1, 2, 3)
	size := a.Size()

	if size != 3 {
		t.Error("Expected 3, got ", size)
	}
}

func TestNdGetRows(t *testing.T) {
	a := Arange(10).Reshape(5, 2)
	rows := a.GetRows(0, 4)
	if !rows.Equals(Array(0, 1, 8, 9).Reshape(2, 2)) {
		t.Error("Expected [[0,1], [8,9]], got ", rows)
	}
}

func TestNdGetCols(t *testing.T) {
	a := Arange(10).Reshape(2, 5)
	cols := a.GetCols(0, 4)

	if !cols.Equals(Array(0, 4, 5, 9).Reshape(2, 2)) {
		t.Error("Expected [[0,4], [5,9]], got ", cols)
	}
}

func TestNdGetEles(t *testing.T) {
	a := Array(2, 3, 4, 3, 2, 1)
	eles := a.GetEles(0, 2, 3)

	if !eles.Equals(Array(2, 4, 3)) {
		t.Error("Expected [2,4,3], got ", eles)
	}
}

func TestNdValues(t *testing.T) {
	a := Arange(3)
	values := a.Values()

	if !util.EqualOfFloat64Slice(values, a.data) {
		t.Error("Expected [0,1,2], got ", values)
	}
}

func TestNdFlat(t *testing.T) {
	a := Arange(4).Reshape(2, 2)
	b := a.Flat()
	a.Set(10, 0, 0)

	if !b.Equals(Arange(4)) {
		t.Error("Expected [[0,1],[2,3]], got ", b)
	}
}

func TestIsEmpty(t *testing.T) {
	a := Arange(20)

	if a.IsEmpty() {
		t.Error("Expected false, got ", true)
	}

	a = Empty()

	if !a.IsEmpty() {
		t.Error("Expected true, got ", false)
	}
}

func TestNdims(t *testing.T) {
	a := Arange(4).Reshape(2, 2)

	if a.NDims() != 2 {
		t.Error("Expected 2, got ", a.NDims())
	}

	a = Arange(4)

	if a.NDims() != 1 {
		t.Error("Expected 1, got ", a.NDims())
	}

	a = Empty()

	if a.NDims() != 0 {
		t.Error("Expected 0, got ", a.NDims())
	}
}

func TestNdarrayIx(t *testing.T) {
	a := Arange(8).Reshape(4, 2)
	b := a.Ix(3, 1)

	if !b.Equals(Array(7)) {
		t.Error("Expected [7], got ", b)
	}

	c := a.Ix(1)

	if !c.Equals(Array(2, 3)) {
		t.Error("Expected [2,3], got ", c)
	}

	a = a.Reshape(2, 2, 2)
	d := a.Ix(1, 1)

	if !d.Equals(Array(6, 7)) {
		t.Error("Expected [6,7], got ", d)
	}

	e := a.Ix(1)

	if !e.Equals(Array(4, 5, 6, 7).Reshape(2, 2)) {
		t.Error("Expected [[4,5], [6,7]], got ", e)
	}

	defer func() {
		p := recover()
		if p != "shape error" {
			t.Error("Expected 'shape error', got ", p)
		}
	}()
	a.Ix(4, 5, 6, 3)
}

func TestNdArrayString(t *testing.T) {
	a := Arange(8).Reshape(2, 2, 2)
	b := a.Ix(0, 0).String()

	if b != fmt.Sprintf("ndarray<[2]>\n([0, 1])") {
		t.Error("Expected ndarray<[2]>\n([0, 1]), got", b)
	}
}

func TestNdarrayValue(t *testing.T) {
	a := Arange(8).Reshape(2, 4)
	b := a.Ix(0, 0).Value()

	if b != 0 {
		t.Error("Expected 0, got ", b)
	}
}
