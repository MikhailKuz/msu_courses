class Matrix_row {
public:
	Matrix_row(const size_t i, const size_t cols, int64_t* mat_arr) : i_row{ i }, cols{ cols }, matrix_arr{ mat_arr } {}
	const int64_t& operator[](const int64_t j) const
	{
		if ((j < 0) || (j >= static_cast<int64_t>(cols)))
			throw std::out_of_range("");
		return *(matrix_arr + (i_row)*cols + j);
	}

	int64_t& operator[](const int64_t j)
	{
		if ((j < 0) || (j >= static_cast<int64_t>(cols)))
			throw std::out_of_range("");
		return *(matrix_arr + (i_row)*cols + j);
	}
private:
	int64_t* matrix_arr;
	const size_t i_row;
	const size_t cols;
	size_t i_col;
};

class Matrix {
public:
	Matrix(int64_t rows, int64_t cols) : rows{ static_cast<size_t>(rows) }, cols(static_cast<size_t>(cols)) {
		if ((rows < 0) || (cols < 0))
			throw std::out_of_range("");
		matrix_arr = new int64_t[rows*cols];
	}
	const Matrix_row operator[](int64_t i) const
	{
		if ((i < 0) || (i >= static_cast<int64_t>(rows)))
			throw std::out_of_range("");
		return Matrix_row(i, cols, matrix_arr);
	}

	Matrix_row operator[](int64_t i)
	{
		if ((i < 0) || (i >= static_cast<int64_t>(rows)))
			throw std::out_of_range("");
		return Matrix_row(i, cols, matrix_arr);
	}
	const Matrix& operator*=(const int64_t other)
	{
		int64_t i = rows * cols - 1;
		while (i >= 0) {
			matrix_arr[i] *= other;
			--i;
		}
		return *this;
	}

	bool operator==(const Matrix& other) const {
		if ((rows != other.getRows()) || (cols != other.getColumns()))
			return false;
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
				if ((*this)[i][j] != other[i][j])
					return false;
		return true;
	}
	bool operator!=(const Matrix& other) const
	{
		return !(*this == other);
	}
	size_t getRows() const {
		return rows;
	}
	size_t getColumns() const {
		return cols;
	}
	~Matrix() {
		delete [] matrix_arr;
	}
private:
	int64_t* matrix_arr;
	const size_t rows;
	const size_t cols;
};
