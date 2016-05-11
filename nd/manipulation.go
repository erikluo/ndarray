package nd

import (
	"math"
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
