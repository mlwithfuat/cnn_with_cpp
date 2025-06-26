#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <stdexcept>
#include <iomanip>

class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows, cols;

public:
    // Constructors
    Matrix() : rows(0), cols(0) {}
    
    Matrix(size_t r, size_t c) : rows(r), cols(c) {
        data.resize(rows, std::vector<double>(cols, 0.0));
    }
    
    Matrix(size_t r, size_t c, double value) : rows(r), cols(c) {
        data.resize(rows, std::vector<double>(cols, value));
    }
    
    Matrix(const std::vector<std::vector<double>>& input_data) {
        rows = input_data.size();
        cols = rows > 0 ? input_data[0].size() : 0;
        data = input_data;
    }

    // Getters
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    
    // Element access
    double& operator()(size_t r, size_t c) {
        if (r >= rows || c >= cols) {
            throw std::out_of_range("Matrix index out of range");
        }
        return data[r][c];
    }
    
    const double& operator()(size_t r, size_t c) const {
        if (r >= rows || c >= cols) {
            throw std::out_of_range("Matrix index out of range");
        }
        return data[r][c];
    }

    // Matrix operations
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }
        
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = data[i][j] + other(i, j);
            }
        }
        return result;
    }
    
    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must match for subtraction");
        }
        
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = data[i][j] - other(i, j);
            }
        }
        return result;
    }
    
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Invalid dimensions for matrix multiplication");
        }
        
        Matrix result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                for (size_t k = 0; k < cols; ++k) {
                    result(i, j) += data[i][k] * other(k, j);
                }
            }
        }
        return result;
    }
    
    // Scalar operations
    Matrix operator*(double scalar) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = data[i][j] * scalar;
            }
        }
        return result;
    }
    
    Matrix operator/(double scalar) const {
        if (std::abs(scalar) < 1e-10) {
            throw std::invalid_argument("Division by zero");
        }
        return *this * (1.0 / scalar);
    }

    // Element-wise operations (Hadamard product)
    Matrix hadamard(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must match for Hadamard product");
        }
        
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = data[i][j] * other(i, j);
            }
        }
        return result;
    }

    // Transpose
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(j, i) = data[i][j];
            }
        }
        return result;
    }

    // Apply function to all elements
    Matrix apply(double (*func)(double)) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = func(data[i][j]);
            }
        }
        return result;
    }

    // CNN specific operations
    
    // Convolution operation (simplified 2D convolution)
    Matrix convolve2D(const Matrix& kernel, size_t stride = 1, size_t padding = 0) const {
        size_t kernel_rows = kernel.getRows();
        size_t kernel_cols = kernel.getCols();
        
        // Calculate output dimensions
        size_t out_rows = (rows + 2 * padding - kernel_rows) / stride + 1;
        size_t out_cols = (cols + 2 * padding - kernel_cols) / stride + 1;
        
        Matrix result(out_rows, out_cols);
        
        for (size_t i = 0; i < out_rows; ++i) {
            for (size_t j = 0; j < out_cols; ++j) {
                double sum = 0.0;
                for (size_t ki = 0; ki < kernel_rows; ++ki) {
                    for (size_t kj = 0; kj < kernel_cols; ++kj) {
                        int row_idx = i * stride + ki - padding;
                        int col_idx = j * stride + kj - padding;
                        
                        if (row_idx >= 0 && row_idx < (int)rows && 
                            col_idx >= 0 && col_idx < (int)cols) {
                            sum += data[row_idx][col_idx] * kernel(ki, kj);
                        }
                    }
                }
                result(i, j) = sum;
            }
        }
        return result;
    }
    
    // Max pooling
    Matrix maxPool(size_t pool_size, size_t stride = 0) const {
        if (stride == 0) stride = pool_size;
        
        size_t out_rows = (rows - pool_size) / stride + 1;
        size_t out_cols = (cols - pool_size) / stride + 1;
        
        Matrix result(out_rows, out_cols);
        
        for (size_t i = 0; i < out_rows; ++i) {
            for (size_t j = 0; j < out_cols; ++j) {
                double max_val = data[i * stride][j * stride];
                for (size_t pi = 0; pi < pool_size; ++pi) {
                    for (size_t pj = 0; pj < pool_size; ++pj) {
                        size_t row_idx = i * stride + pi;
                        size_t col_idx = j * stride + pj;
                        if (row_idx < rows && col_idx < cols) {
                            max_val = std::max(max_val, data[row_idx][col_idx]);
                        }
                    }
                }
                result(i, j) = max_val;
            }
        }
        return result;
    }
    
    // Flatten matrix to column vector
    Matrix flatten() const {
        Matrix result(rows * cols, 1);
        size_t idx = 0;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(idx++, 0) = data[i][j];
            }
        }
        return result;
    }
    
    // Reshape matrix
    Matrix reshape(size_t new_rows, size_t new_cols) const {
        if (new_rows * new_cols != rows * cols) {
            throw std::invalid_argument("New dimensions must have same total size");
        }
        
        Matrix result(new_rows, new_cols);
        size_t idx = 0;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                size_t new_i = idx / new_cols;
                size_t new_j = idx % new_cols;
                result(new_i, new_j) = data[i][j];
                idx++;
            }
        }
        return result;
    }

    // Utility functions
    void randomize(double min_val = -1.0, double max_val = 1.0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(min_val, max_val);
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                data[i][j] = dis(gen);
            }
        }
    }
    
    void randomizeXavier() {
        double limit = std::sqrt(6.0 / (rows + cols));
        randomize(-limit, limit);
    }
    
    void zero() {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                data[i][j] = 0.0;
            }
        }
    }
    
    void fill(double value) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                data[i][j] = value;
            }
        }
    }
    
    // Statistics
    double sum() const {
        double total = 0.0;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                total += data[i][j];
            }
        }
        return total;
    }
    
    double mean() const {
        return sum() / (rows * cols);
    }
    
    double max() const {
        if (rows == 0 || cols == 0) return 0.0;
        double max_val = data[0][0];
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                max_val = std::max(max_val, data[i][j]);
            }
        }
        return max_val;
    }
    
    double min() const {
        if (rows == 0 || cols == 0) return 0.0;
        double min_val = data[0][0];
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                min_val = std::min(min_val, data[i][j]);
            }
        }
        return min_val;
    }

    // Display
    void print() const {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(3) 
                         << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

// Activation functions (CNN'de kullanılacak)
namespace Activation {
    inline double relu(double x) {
        return std::max(0.0, x);
    }
    
    inline double relu_derivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }
    
    inline double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    
    inline double sigmoid_derivative(double x) {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }
    
    inline double tanh_func(double x) {
        return std::tanh(x);
    }
    
    inline double tanh_derivative(double x) {
        double t = std::tanh(x);
        return 1.0 - t * t;
    }
    
    // Softmax (matrix için)
    Matrix softmax(const Matrix& input) {
        Matrix result = input;
        size_t rows = input.getRows();
        size_t cols = input.getCols();
        
        for (size_t i = 0; i < rows; ++i) {
            // Find max for numerical stability
            double max_val = input(i, 0);
            for (size_t j = 1; j < cols; ++j) {
                max_val = std::max(max_val, input(i, j));
            }
            
            // Calculate exp and sum
            double sum = 0.0;
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = std::exp(input(i, j) - max_val);
                sum += result(i, j);
            }
            
            // Normalize
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) /= sum;
            }
        }
        return result;
    }
}

// Scalar * Matrix operator
Matrix operator*(double scalar, const Matrix& matrix) {
    return matrix * scalar;
}

#endif // MATRIX_H