package nd

import (
	"fmt"
	"math"
	"testing"
)

func TestExp(t *testing.T) {
	a := Array(1, 2, 3)
	a_exp := Exp(a)

	if !a_exp.Equals(Array(2.7182817, 7.389056, 20.085537)) {
		t.Error("Expected [2.7182817, 7.389056, 20.085537], got ", a_exp)
	}
}

func TestMap(t *testing.T) {
	a := Array(1, 2, 3)
	a_map := Map(a, func(e float64) float64 {
		return math.Exp(e)
	})

	if !a_map.Equals(Array(2.7182817, 7.389056, 20.085537)) {
		t.Error("Expected [2.7182817, 7.389056, 20.085537], got ", a_map)
	}
}

func TestNonZero(t *testing.T) {
	a := Array(1, 2, 3, 0, 0, 3, 4)
	index := NonZero(a)

	if len(index) != 1 {
		t.Error("Expected 1, got ", len(index))
	}
	if !EqualOfIntSlice(index[0], []int{0, 1, 2, 5, 6}) {
		t.Error("Expected [0,1,2,5,6}, got ", index[0])
	}

	a = Array(1, 2, 3, 0, 0, 3).Reshape(2, 3)
	index = NonZero(a)

	if len(index) != 2 {
		t.Error("Expected 2, got ", len(index))
	}

	if !EqualOfIntSlice(index[0], []int{0, 0, 0, 1}) {
		t.Error("Expected [0,0,0,1], got ", index[0])
	}

	if !EqualOfIntSlice(index[1], []int{0, 1, 2, 2}) {
		t.Error("Expected [0,1,2,2], got ", index[1])
	}
}

func TestCopyTo(t *testing.T) {
	a := Arange(4).Reshape(2, 2)
	b := Zeros(2, 2)

	CopyTo(a, b)

	a.Set(10, 0)

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
	CopyTo(a, b)
}

func TestRavel(t *testing.T) {
	a := Arange(4).Reshape(2, 2)
	b := Ravel(a)
	a.Set(10, 0)

	if !b.Equals(Arange(4)) {
		t.Error("Expected [0,1,2,3], got ", b)
	}
}

func TestAtleast2D(t *testing.T) {
	a := Arange(4)
	b := Atleast2D(a)

	if !b.Equals(Arange(4).Reshape(4, 1)) {
		t.Error("Expected [[0,1,2,3]], got ", b)
	}

	b = Atleast2D(a.Reshape(2, 2))

	if !b.Equals(Arange(4).Reshape(2, 2)) {
		t.Error("Expected [[0,1],[2,3]], got ", b)
	}
}

func TestAtleast3D(t *testing.T) {
	a := Arange(4)
	b := Atleast3D(a)

	if !b.Equals(Arange(4).Reshape(4, 1, 1)) {
		t.Error("Expected [[[0,1,2,3]]], got ", b)
	}

	b = Atleast3D(a.Reshape(2, 2))

	if !b.Equals(Arange(4).Reshape(2, 2, 1)) {
		t.Error("Expected [[[0,1],[2,3]]], got ", b)
	}
}

func TestMean(t *testing.T) {
	a := Arange(4)
	mean := Mean(a).Get(0)

	if mean != 1.5 {
		t.Error("Expected 1.5, got ", mean)
	}

	a = Arange(4).Reshape(2, 2)
	means := Mean(a)

	if !means.Equals(Array(0.5, 2.5)) {
		t.Error("Expected [0.5, 2.5], got ", means)
	}

	defer func() {
		p := recover()
		if p != "shape error" {
			t.Error("Expected 'shape error', got ", p)
		}
	}()

	a = Arange(8).Reshape(2, 2, 2)
	Mean(a)
}

func TestSum(t *testing.T) {
	a := Arange(4)
	sum := Sum(a).Get(0)

	if sum != 6 {
		t.Error("Expected 6, got ", sum)
	}

	a = Arange(4).Reshape(2, 2)
	sums := Sum(a)

	if !sums.Equals(Array(1, 5)) {
		t.Error("Expected [1, 5], got ", sums)
	}

	defer func() {
		p := recover()
		if p != "shape error" {
			t.Error("Expected 'shape error', got ", p)
		}
	}()
	a = Arange(8).Reshape(2, 2, 2)
	Sum(a)
}

func TestStd(t *testing.T) {
	mat := Array(2, 3, 1, 4)
	std := Std(mat).Get(0)

	if std != 1.118033988749895 {
		t.Error("expected 1.118033988749895, got ", std)
	}

	mat = mat.Reshape(2, 2)
	stds := Std(mat)

	if !stds.Equals(Array(0.5, 1.5)) {
		t.Error("Expected [0.5, 1.5], got ", stds)
	}

	defer func() {
		p := recover()
		if p != "shape error" {
			t.Error("Expected 'shape error', got ", p)
		}
	}()

	Std(Arange(8).Reshape(2, 2, 2))
}

func TestVar(t *testing.T) {
	mat := Array(2, 3, 1, 4)
	vars := Var(mat).Get(0)

	if vars != 1.25 {
		t.Error("expected 1.25, got ", vars)
	}

	mat = mat.Reshape(2, 2)
	vars_ := Var(mat)

	if !vars_.Equals(Array(0.25, 2.25)) {
		t.Error("Expected [0.25, 2.25], got ", vars_)
	}

	defer func() {
		p := recover()
		if p != "shape error" {
			t.Error("Expected 'shape error', got ", p)
		}
	}()

	Var(Arange(8).Reshape(2, 2, 2))
}

func TestMax(t *testing.T) {
	a := Arange(4)
	max := Max(a).Get(0)

	if max != 3 {
		t.Error("Expected 3, got ", max)
	}

	a = Arange(4).Reshape(2, 2)
	maxs := Max(a)

	if !maxs.Equals(Array(1, 3)) {
		t.Error("Expected [1,3], got ", maxs)
	}

	defer func() {
		p := recover()
		if p != "shape error" {
			t.Error("Expected 'shape error', got ", p)
		}
	}()

	Max(Arange(8).Reshape(2, 2, 2))
}

func TestMin(t *testing.T) {
	a := Arange(4)
	min := Min(a).Get(0)

	if min != 0 {
		t.Error("Expected 0, got ", min)
	}

	a = Arange(4).Reshape(2, 2)
	mins := Min(a)

	if !mins.Equals(Array(0, 2)) {
		t.Error("Expected [0,2], got ", mins)
	}

	defer func() {
		p := recover()
		if p != "shape error" {
			t.Error("Expected 'shape error', got ", p)
		}
	}()

	Min(Arange(8).Reshape(2, 2, 2))
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

	Sort(a)

	if !a.Equals(Array(1, 2, 3, 3, 4)) {
		t.Error("Expected [1,2,3,3,4], got ", a)
	}

	a = Array(3, 4, 5, 1, 3, 2).Reshape(2, 3)
	Sort(a)

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
	Sort(a)
}

func TestHSplit(t *testing.T) {
	a := Arange(4).Reshape(2, 2)

	b := HSplit(a)

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
	HSplit(a)
}

func TestVSplit(t *testing.T) {
	a := Arange(4).Reshape(2, 2)
	b := VSplit(a)

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
	VSplit(a)
}

func TestTile(t *testing.T) {
	a := Arange(3)
	b := Tile(a, 1, 1, 1, 1)

	if !b.Equals(Array(0, 1, 2)) {
		t.Error(fmt.Sprintf("Expected [0,1,2], got "), b)
	}

	b = Tile(a, 2)

	if !b.Equals(Array(0, 1, 2, 0, 1, 2)) {
		t.Error("Expected [0,1,2,0,1,2], got ", b)
	}

	a = Arange(4).Reshape(2, 2)
	b = Tile(a, 2)

	if !b.Equals(Array(0, 1, 0, 1, 2, 3, 2, 3).Reshape(2, 4)) {
		t.Error("Expected [0,1,0,1,2,3,2,3], got ", b)
	}

	b = Tile(a, 2, 1)

	if !b.Equals(Array(0, 1, 2, 3, 0, 1, 2, 3).Reshape(4, 2)) {
		t.Error("Expected [0,1,0,1,2,3,2,3], got ", b)
	}
}

func TestUnique(t *testing.T) {
	a := Array(3, 4, 2, 3, 1, 2, 4, 3, 2)
	us := Unique(a)

	if !EqualOfFloat64Slice(us, []float64{1, 2, 3, 4}) {
		t.Error("Expected [1,2,3,4], got ", us)
	}
}

func TestArgMax(t *testing.T) {
	a := Arange(4)
	maxIndex := ArgMax(a)[0]

	if maxIndex != 3 {
		t.Error("Expected 3, got ", maxIndex)
	}

	a = Arange(4).Reshape(2, 2)
	maxIndexs := ArgMax(a)
	if !EqualOfIntSlice(maxIndexs, []int{1, 1}) {
		t.Error("Expected [1,1], got ", maxIndexs)
	}

	defer func() {
		p := recover()
		if p != "shape error" {
			t.Error("Expected 'shape error', got ", p)
		}
	}()

	ArgMax(Empty())
}

func TestArgMin(t *testing.T) {
	a := Arange(3)
	minIndex := ArgMin(a)[0]

	if minIndex != 0 {
		t.Error("Expected 0, got ", minIndex)
	}

	a = Arange(4).Reshape(2, 2)
	minIndexs := ArgMin(a)

	if !EqualOfIntSlice(minIndexs, []int{0, 0}) {
		t.Error("Expected [0,0], got ", minIndexs)
	}
}

func TestExtract(t *testing.T) {
	a := Array(3, -1, 4, -2)

	es := Extract(a, func(ele float64) bool {
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
	count := CountNonZero(a)

	if count != 4 {
		t.Error("Expected 4, got ", count)
	}
}

func demo() {
	a := Array(34)
	a.Add(Array(2))
	fmt.Println(a)
}
