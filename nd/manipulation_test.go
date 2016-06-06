package nd

import (
	"fmt"
	"math"
	"testing"

	"github.com/ledao/ndarray/util"
)

func TestExp(t *testing.T) {
	a := Array(1, 2, 3)
	a_exp := a.Exp()

	if !a_exp.Equals(Array(2.7182817, 7.389056, 20.085537)) {
		t.Error("Expected [2.7182817, 7.389056, 20.085537], got ", a_exp)
	}
}

func TestMap(t *testing.T) {
	a := Array(1, 2, 3)
	a_map := a.Map(func(e float64) float64 {
		return math.Exp(e)
	})

	if !a_map.Equals(Array(2.7182817, 7.389056, 20.085537)) {
		t.Error("Expected [2.7182817, 7.389056, 20.085537], got ", a_map)
	}
}

func TestNonZero(t *testing.T) {
	a := Array(1, 2, 3, 0, 0, 3, 4)
	index := a.NonZero()

	if len(index) != 1 {
		t.Error("Expected 1, got ", len(index))
	}
	if !util.EqualOfIntSlice(index[0], []int{0, 1, 2, 5, 6}) {
		t.Error("Expected [0,1,2,5,6}, got ", index[0])
	}

	a = Array(1, 2, 3, 0, 0, 3).Reshape(2, 3)
	index = a.NonZero()

	if len(index) != 2 {
		t.Error("Expected 2, got ", len(index))
	}

	if !util.EqualOfIntSlice(index[0], []int{0, 0, 0, 1}) {
		t.Error("Expected [0,0,0,1], got ", index[0])
	}

	if !util.EqualOfIntSlice(index[1], []int{0, 1, 2, 2}) {
		t.Error("Expected [0,1,2,2], got ", index[1])
	}
}

func TestCopyTo(t *testing.T) {
	a := Arange(4).Reshape(2, 2)
	b := Zeros(2, 2)

	a.CopyTo(b)

	a.Set(10, 0, 0)

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
	a.CopyTo(b)
}

func TestRavel(t *testing.T) {
	a := Arange(4).Reshape(2, 2)
	b := a.Ravel()
	a.Set(10, 0, 0)

	if !b.Equals(Arange(4)) {
		t.Error("Expected [0,1,2,3], got ", b)
	}
}

func TestAtleast2D(t *testing.T) {
	a := Arange(4)
	b := a.Atleast2D()

	if !b.Equals(Arange(4).Reshape(4, 1)) {
		t.Error("Expected [[0,1,2,3]], got ", b)
	}

	b = a.Reshape(2, 2).Atleast2D()

	if !b.Equals(Arange(4).Reshape(2, 2)) {
		t.Error("Expected [[0,1],[2,3]], got ", b)
	}
}

func TestAtleast3D(t *testing.T) {
	a := Arange(4)
	b := a.Atleast3D()

	if !b.Equals(Arange(4).Reshape(4, 1, 1)) {
		t.Error("Expected [[[0,1,2,3]]], got ", b)
	}

	b = a.Reshape(2, 2).Atleast3D()

	if !b.Equals(Arange(4).Reshape(2, 2, 1)) {
		t.Error("Expected [[[0,1],[2,3]]], got ", b)
	}
}

func TestVStack(t *testing.T) {
	a := Arange(3)
	b := Arange(6).Reshape(2, 3)
	c := VStack(a, b)

	if !c.Equals(Array(0, 1, 2, 0, 1, 2, 3, 4, 5).Reshape(3, 3)) {
		t.Error("Expected [[0,1,2], [0,1,2], [3,4,5]], got ", c)
	}

	c = VStack()

	if !c.Equals(Empty()) {
		t.Error("Expected [], got ", c)
	}

	defer func() {
		p := recover()
		if p != "shape error" {
			t.Error("Expected 'shape error', got ", p)
		}
	}()

	VStack(a, b, Arange(4))
}

func TestHStack(t *testing.T) {
	a := Arange(3)
	b := Array(2, 3, 4, 1, 2, 3).Reshape(3, 2)
	c := Arange(1, 4)
	d := HStack(a, b, c)

	if !d.Equals(Array(0, 2, 3, 1, 1, 4, 1, 2, 2, 2, 3, 3).Reshape(3, 4)) {
		t.Error("Expected [[0,2,3,1],[1,4,1,2],[2,2,3,3]], got ", d)
	}

	d = HStack()

	if !d.Equals(Empty()) {
		t.Error("Expected [], got ", d)
	}

	defer func() {
		p := recover()
		if p != "shape error" {
			t.Error("Expected 'shape error', got ", p)
		}
	}()

	HStack(a, b, Arange(2))
}

func TestSort(t *testing.T) {
	a := Array(3, 4, 2, 3, 1)

	a.Sort()

	if !a.Equals(Array(1, 2, 3, 3, 4)) {
		t.Error("Expected [1,2,3,3,4], got ", a)
	}

	a = Array(3, 4, 5, 1, 3, 2).Reshape(2, 3)
	a.Sort()

	if !a.Equals(Array(3, 4, 5, 1, 2, 3).Reshape(2, 3)) {
		t.Error("Expected [[3,4,5], [1,2,3]], got ", a)
	}

	defer func() {
		p := recover()
		if p != "shape error" {
			t.Error("Expected 'shape error', got ", p)
		}
	}()
	a = Arange(8).Reshape(2, 2, 2)
	a.Sort()
}

func TestHSplit(t *testing.T) {
	a := Arange(4).Reshape(2, 2)

	b := a.HSplit()

	if !b[0].Equals(Array(0, 1)) {
		t.Error("Expected [0, 1], got ", b[0])
	}

	if !b[1].Equals(Array(2, 3)) {
		t.Error("Expected [2,3], got ", b[1])
	}

	defer func() {
		p := recover()
		if p != "shape error" {
			t.Error("Expected 'shape error', got ", p)
		}
	}()
	a = Arange(3)
	a.HSplit()
}

func TestVSplit(t *testing.T) {
	a := Arange(4).Reshape(2, 2)
	b := a.VSplit()

	if !b[0].Equals(Array(0, 2)) {
		t.Error("Expected [0,2], got ", b[0])
	}

	if !b[1].Equals(Array(1, 3)) {
		t.Error("Expected [1,3], got ", b[1])
	}

	defer func() {
		p := recover()
		if p != "shape error" {
			t.Error("Expected 'shape error', got ", p)
		}
	}()

	a = Array(9)
	a.VSplit()
}

func TestTile(t *testing.T) {
	a := Arange(3)
	b := a.Tile(1, 1, 1, 1)

	if !b.Equals(Array(0, 1, 2)) {
		t.Error(fmt.Sprintf("Expected [0,1,2], got "), b)
	}

	b = a.Tile(2)

	if !b.Equals(Array(0, 1, 2, 0, 1, 2)) {
		t.Error("Expected [0,1,2,0,1,2], got ", b)
	}

	a = Arange(4).Reshape(2, 2)
	b = a.Tile(2)

	if !b.Equals(Array(0, 1, 0, 1, 2, 3, 2, 3).Reshape(2, 4)) {
		t.Error("Expected [0,1,0,1,2,3,2,3], got ", b)
	}

	b = a.Tile(2, 1)

	if !b.Equals(Array(0, 1, 2, 3, 0, 1, 2, 3).Reshape(4, 2)) {
		t.Error("Expected [0,1,0,1,2,3,2,3], got ", b)
	}
}

func TestUnique(t *testing.T) {
	a := Array(3, 4, 2, 3, 1, 2, 4, 3, 2)
	us := a.Unique()

	if !util.EqualOfFloat64Slice(us, []float64{1, 2, 3, 4}) {
		t.Error("Expected [1,2,3,4], got ", us)
	}
}

func TestArgMax(t *testing.T) {
	a := Arange(4)
	maxIndex := a.ArgMax()[0]

	if maxIndex != 3 {
		t.Error("Expected 3, got ", maxIndex)
	}

	a = Arange(4).Reshape(2, 2)
	maxIndexs := a.ArgMax()
	if !util.EqualOfIntSlice(maxIndexs, []int{1, 1}) {
		t.Error("Expected [1,1], got ", maxIndexs)
	}

	defer func() {
		p := recover()
		if p != "shape error" {
			t.Error("Expected 'shape error', got ", p)
		}
	}()

	Empty().ArgMax()
}

func TestArgMin(t *testing.T) {
	a := Arange(3)
	minIndex := a.ArgMin()[0]

	if minIndex != 0 {
		t.Error("Expected 0, got ", minIndex)
	}

	a = Arange(4).Reshape(2, 2)
	minIndexs := a.ArgMin()

	if !util.EqualOfIntSlice(minIndexs, []int{0, 0}) {
		t.Error("Expected [0,0], got ", minIndexs)
	}
}

func TestExtract(t *testing.T) {
	a := Array(3, -1, 4, -2)

	es := a.Extract(func(ele float64) bool {
		if ele > 0 {
			return true
		} else {
			return false
		}
	})

	if !es.Equals(Array(3, 4)) {
		t.Error("Expected [3,4], got ", es)
	}
}

func TestCountNonZero(t *testing.T) {
	a := Array(0, 9, 3, 2, 0.000000003)
	count := a.CountNonZero()

	if count != 4 {
		t.Error("Expected 4, got ", count)
	}
}

func TestNdSumOfAll(t *testing.T) {
	arr := Array([]float64{1, 2, 3, 4, 5, 6, 7, 8}...).Reshape(2, 4)

	if arr.SumAll() != 36 {
		t.Error("Expected 36, got ", arr.SumAll())
	}
}

func TestNdMulBit(t *testing.T) {
	arr := Array(2, 3, 4)
	a2 := arr.MulBit(arr)

	if !a2.Equals(Array(4, 9, 16)) {
		t.Error("Expected [4, 9, 16], got %v", a2)
	}
}

func TestNdDot(t *testing.T) {
	a1 := Array(2, 3, 4, 5, 6, 7).Reshape(2, 3)
	a2 := Array(2, 3, 4, 5, 6, 7).Reshape(3, 2)

	m := a1.Dot(a2)

	if !m.Equals(Array(40, 49, 76, 94).Reshape(2, 2)) {
		t.Error("Expected [[40,49],[76,94]], got ", m)
	}

	a := Array(1, 2, 3, 4).Reshape(2, 2)
	b := Array(2, 3)

	mul := b.Dot(a)
	if !mul.Reshape(2).Equals(Array(11, 16)) {
		t.Error("Expected [11,16], got ", mul)
	}

	mul = a.Dot(b)
	if !mul.Reshape(2).Equals(Array(8, 18)) {
		t.Error("Expected [8, 18], got ", mul)
	}

	a1 = Array(2, 3, 4)
	a2 = Array(1, 2, 3)

	dot := a1.Dot(a2).Get(0)
	if dot != 20 {
		t.Error("Expected 20, got ", dot)
	}
}

func TestNdInv(t *testing.T) {
	arr := Array([]float64{1, 2, 3, 4}...).Reshape(2, 2)

	inv := arr.Inv()

	if inv.Equals(Array(-2, 1, 1.5, -0.5).Reshape(2, 2)) != true {
		t.Error("Expected [[-2,1],[1.5,-0.5]], got ", inv)
	}
}

func TestNdDet(t *testing.T) {
	arr := Array(1, 2, 3, 4).Reshape(2, 2)
	det := arr.Det()

	if det != -2.0 {
		t.Error("Expected -2.0, got ", det)
	}
}

func TestNdAdd(t *testing.T) {
	a1 := Array(1, 2, 3)
	a2 := Array(2, 3, 4)

	a3 := a1.Add(a2)

	if !a3.Equals(Array(3, 5, 7)) {
		t.Error("Expected [3,5,7], got ", a3)
	}

	a2 = Array(3)
	a3 = a1.Add(a2)

	if !a3.Equals(Array(4, 5, 6)) {
		t.Error("Expected [3,4,5], got ", a3)
	}

	a1 = Array(1, 2, 3, 4, 5, 6).Reshape(3, 2)
	a2 = Array(1, 2, 3).Reshape(3, 1)

	a3 = a1.Add(a2)

	if !a3.Equals(Array(2, 3, 5, 6, 8, 9).Reshape(3, 2)) {
		t.Error("Expected [[2,3],[5,6],[8,9]], got ", a3)
	}

	a1 = a1.Reshape(2, 3)
	a2 = a2.Reshape(1, 3)
	a3 = a1.Add(a2)

	if !a3.Equals(Array(2, 4, 6, 5, 7, 9).Reshape(2, 3)) {
		t.Error("Expected [[2,4,6],[5,7,9]], got ", a3)
	}
}

func TestNdSub(t *testing.T) {
	a1 := Array(1, 2, 3)
	a2 := Array(2, 3, 4)

	a3 := a1.Sub(a2)

	if !a3.Equals(Array(-1, -1, -1)) {
		t.Error("Expected [-1,-1,-1], got ", a3)
	}

	a2 = Array(3)
	a3 = a1.Sub(a2)

	if !a3.Equals(Array(-2, -1, 0)) {
		t.Error("Expected [-2,-1,0], got ", a3)
	}

	a1 = Array(1, 2, 3, 4, 5, 6).Reshape(2, 3)
	a2 = Array(1, 2, 3).Reshape(1, 3)
	a3 = a1.Sub(a2)

	if !a3.Equals(Array(0, 0, 0, 3, 3, 3).Reshape(2, 3)) {
		t.Error("Expected [[0,0,0], [3,3,3]], got ", a3)
	}

	a1 = a1.Reshape(3, 2)
	a2 = a2.Reshape(3, 1)
	a3 = a1.Sub(a2)

	if !a3.Equals(Array(0, 1, 1, 2, 2, 3).Reshape(3, 2)) {
		t.Error("Expected [[0,1],[1,2],[2,3]], got ", a3)
	}

	defer func() {
		if err := recover(); err != nil {
			if err != "shape error" {
				t.Error("Expected 'Shape error', got ", err)
			}
		}
	}()
	a2.Sub(a1)
}

func TestNdMul(t *testing.T) {
	a1 := Array(1, 2, 3, 4, 5, 6).Reshape(2, 3)
	a2 := Array(1, 2, 3, 4, 5, 6).Reshape(2, 3)
	a3 := a1.Mul(a2)

	if !a3.Equals(Array(1, 4, 9, 16, 25, 36).Reshape(2, 3)) {
		t.Error("Expected [[1,4,9], [16,25,36]], got ", a3)
	}

	a2 = Array(1, 2, 3).Reshape(1, 3)
	a3 = a1.Mul(a2)

	if !a3.Equals(Array(1, 4, 9, 4, 10, 18).Reshape(2, 3)) {
		t.Error("Expected [[1,4,9],[4,10,18]], got ", a3)
	}

	a2 = Array(1, 2).Reshape(2, 1)
	a3 = a1.Mul(a2)

	if !a3.Equals(Array(1, 2, 3, 8, 10, 12).Reshape(2, 3)) {
		t.Error("Expected [[1,2,3], [8,10,12]], got ", a3)
	}

	a2 = Array(2)
	a3 = a1.Mul(a2)

	if !a3.Equals(Array(2, 4, 6, 8, 10, 12).Reshape(2, 3)) {
		t.Error("Expected [[2,4,6],[8,10,12]], got ", a3)
	}
}

func TestNdDiv(t *testing.T) {
	a1 := Array(1, 2, 3, 4, 5, 6).Reshape(2, 3)
	a2 := Array(1, 2, 3, 4, 5, 6).Reshape(2, 3)
	a3 := a1.Div(a2)

	if !a3.Equals(Array(1, 1, 1, 1, 1, 1).Reshape(2, 3)) {
		t.Error("Expected [[1,1,1], [1,1,1]], got ", a3)
	}

	a2 = Array(1, 2, 3).Reshape(1, 3)
	a3 = a1.Div(a2)

	if !a3.Equals(Array(1, 1, 1, 4, 2.5, 2).Reshape(2, 3)) {
		t.Error("Expected [[1,1,1],[4,2.5,2]], got ", a3)
	}

	a2 = Array(1, 2).Reshape(2, 1)
	a3 = a1.Div(a2)

	if !a3.Equals(Array(1, 2, 3, 2, 2.5, 3).Reshape(2, 3)) {
		t.Error("Expected [[1,2,3], [2,2.5,3]], got ", a3)
	}

	a2 = Array(2)
	a3 = a1.Div(a2)

	if !a3.Equals(Array(0.5, 1, 1.5, 2, 2.5, 3).Reshape(2, 3)) {
		t.Error("Expected [[0.5, 1, 1.5],[2, 2.5, 3]], got ", a3)
	}
}
