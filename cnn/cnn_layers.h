#ifndef CNN_LAYERS_H
#define CNN_LAYERS_H

#include "matrix.h"
#include <vector>
#include <memory>
#include <string>

// Abstract base class for all layers
class Layer {
public:
    virtual ~Layer() = default;
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& gradient) = 0;
    virtual void updateWeights(double learning_rate) {}
    virtual std::string getType() const = 0;
    virtual void print() const {}
};

// Convolutional Layer
class ConvolutionalLayer : public Layer {
private:
    std::vector<Matrix> kernels;
    std::vector<double> biases;
    size_t num_filters;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    
    // For backpropagation
    Matrix last_input;
    std::vector<Matrix> kernel_gradients;
    std::vector<double> bias_gradients;
    
public:
    ConvolutionalLayer(size_t num_filters, size_t kernel_size, size_t stride = 1, size_t padding = 0)
        : num_filters(num_filters), kernel_size(kernel_size), stride(stride), padding(padding) {
        
        // Initialize kernels and biases
        kernels.resize(num_filters);
        biases.resize(num_filters);
        kernel_gradients.resize(num_filters);
        bias_gradients.resize(num_filters);
        
        for (size_t i = 0; i < num_filters; ++i) {
            kernels[i] = Matrix(kernel_size, kernel_size);
            kernels[i].randomizeXavier();
            biases[i] = 0.0;
            
            kernel_gradients[i] = Matrix(kernel_size, kernel_size);
            bias_gradients[i] = 0.0;
        }
    }
    
    Matrix forward(const Matrix& input) override {
        last_input = input;
        
        // Calculate output dimensions
        size_t input_rows = input.getRows();
        size_t input_cols = input.getCols();
        size_t output_rows = (input_rows + 2 * padding - kernel_size) / stride + 1;
        size_t output_cols = (input_cols + 2 * padding - kernel_size) / stride + 1;
        
        // Create output matrix (flattened multiple feature maps)
        Matrix output(output_rows * num_filters, output_cols);
        
        for (size_t f = 0; f < num_filters; ++f) {
            Matrix feature_map = input.convolve2D(kernels[f], stride, padding);
            
            // Add bias
            for (size_t i = 0; i < feature_map.getRows(); ++i) {
                for (size_t j = 0; j < feature_map.getCols(); ++j) {
                    feature_map(i, j) += biases[f];
                }
            }
            
            // Copy to output
            for (size_t i = 0; i < output_rows; ++i) {
                for (size_t j = 0; j < output_cols; ++j) {
                    output(f * output_rows + i, j) = feature_map(i, j);
                }
            }
        }
        
        return output;
    }
    
    Matrix backward(const Matrix& gradient) override {
        // Simplified backpropagation for convolution
        // In practice, this would be more complex with proper gradient computation
        
        size_t input_rows = last_input.getRows();
        size_t input_cols = last_input.getCols();
        Matrix input_gradient(input_rows, input_cols);
        input_gradient.zero();
        
        // Reset gradients
        for (size_t f = 0; f < num_filters; ++f) {
            kernel_gradients[f].zero();
            bias_gradients[f] = 0.0;
        }
        
        // Compute gradients (simplified version)
        size_t output_rows = (input_rows + 2 * padding - kernel_size) / stride + 1;
        size_t output_cols = (input_cols + 2 * padding - kernel_size) / stride + 1;
        
        for (size_t f = 0; f < num_filters; ++f) {
            for (size_t i = 0; i < output_rows; ++i) {
                for (size_t j = 0; j < output_cols; ++j) {
                    double grad_val = gradient(f * output_rows + i, j);
                    bias_gradients[f] += grad_val;
                    
                    // Update kernel gradients
                    for (size_t ki = 0; ki < kernel_size; ++ki) {
                        for (size_t kj = 0; kj < kernel_size; ++kj) {
                            int input_i = i * stride + ki - padding;
                            int input_j = j * stride + kj - padding;
                            
                            if (input_i >= 0 && input_i < (int)input_rows && 
                                input_j >= 0 && input_j < (int)input_cols) {
                                kernel_gradients[f](ki, kj) += grad_val * last_input(input_i, input_j);
                            }
                        }
                    }
                }
            }
        }
        
        return input_gradient;
    }
    
    void updateWeights(double learning_rate) override {
        for (size_t f = 0; f < num_filters; ++f) {
            // Update kernels
            for (size_t i = 0; i < kernel_size; ++i) {
                for (size_t j = 0; j < kernel_size; ++j) {
                    kernels[f](i, j) -= learning_rate * kernel_gradients[f](i, j);
                }
            }
            
            // Update biases
            biases[f] -= learning_rate * bias_gradients[f];
        }
    }
    
    std::string getType() const override {
        return "Convolutional";
    }
    
    void print() const override {
        std::cout << "Convolutional Layer: " << num_filters << " filters, " 
                  << kernel_size << "x" << kernel_size << " kernel, stride=" << stride << std::endl;
    }
};

// Activation Layer
class ActivationLayer : public Layer {
private:
    std::string activation_type;
    Matrix last_input;
    
public:
    ActivationLayer(const std::string& type) : activation_type(type) {}
    
    Matrix forward(const Matrix& input) override {
        last_input = input;
        
        if (activation_type == "relu") {
            return input.apply(Activation::relu);
        } else if (activation_type == "sigmoid") {
            return input.apply(Activation::sigmoid);
        } else if (activation_type == "tanh") {
            return input.apply(Activation::tanh_func);
        } else if (activation_type == "softmax") {
            return Activation::softmax(input);
        }
        
        return input; // Linear activation
    }
    
    Matrix backward(const Matrix& gradient) override {
        Matrix result = gradient;
        
        if (activation_type == "relu") {
            for (size_t i = 0; i < result.getRows(); ++i) {
                for (size_t j = 0; j < result.getCols(); ++j) {
                    result(i, j) *= Activation::relu_derivative(last_input(i, j));
                }
            }
        } else if (activation_type == "sigmoid") {
            for (size_t i = 0; i < result.getRows(); ++i) {
                for (size_t j = 0; j < result.getCols(); ++j) {
                    result(i, j) *= Activation::sigmoid_derivative(last_input(i, j));
                }
            }
        } else if (activation_type == "tanh") {
            for (size_t i = 0; i < result.getRows(); ++i) {
                for (size_t j = 0; j < result.getCols(); ++j) {
                    result(i, j) *= Activation::tanh_derivative(last_input(i, j));
                }
            }
        }
        
        return result;
    }
    
    std::string getType() const override {
        return "Activation (" + activation_type + ")";
    }
    
    void print() const override {
        std::cout << "Activation Layer: " << activation_type << std::endl;
    }
};

// Max Pooling Layer
class MaxPoolingLayer : public Layer {
private:
    size_t pool_size;
    size_t stride;
    Matrix last_input;
    Matrix mask; // For backpropagation
    
public:
    MaxPoolingLayer(size_t pool_size, size_t stride = 0) 
        : pool_size(pool_size), stride(stride == 0 ? pool_size : stride) {}
    
    Matrix forward(const Matrix& input) override {
        last_input = input;
        
        size_t input_rows = input.getRows();
        size_t input_cols = input.getCols();
        size_t output_rows = (input_rows - pool_size) / stride + 1;
        size_t output_cols = (input_cols - pool_size) / stride + 1;
        
        Matrix output(output_rows, output_cols);
        mask = Matrix(input_rows, input_cols);
        mask.zero();
        
        for (size_t i = 0; i < output_rows; ++i) {
            for (size_t j = 0; j < output_cols; ++j) {
                double max_val = input(i * stride, j * stride);
                size_t max_i = i * stride;
                size_t max_j = j * stride;
                
                // Find maximum in pool region
                for (size_t pi = 0; pi < pool_size; ++pi) {
                    for (size_t pj = 0; pj < pool_size; ++pj) {
                        size_t row_idx = i * stride + pi;
                        size_t col_idx = j * stride + pj;
                        
                        if (row_idx < input_rows && col_idx < input_cols) {
                            if (input(row_idx, col_idx) > max_val) {
                                max_val = input(row_idx, col_idx);
                                max_i = row_idx;
                                max_j = col_idx;
                            }
                        }
                    }
                }
                
                output(i, j) = max_val;
                mask(max_i, max_j) = 1.0; // Mark the position of maximum
            }
        }
        
        return output;
    }
    
    Matrix backward(const Matrix& gradient) override {
        Matrix input_gradient(last_input.getRows(), last_input.getCols());
        input_gradient.zero();
        
        size_t output_rows = gradient.getRows();
        size_t output_cols = gradient.getCols();
        
        for (size_t i = 0; i < output_rows; ++i) {
            for (size_t j = 0; j < output_cols; ++j) {
                // Distribute gradient only to the max position
                for (size_t pi = 0; pi < pool_size; ++pi) {
                    for (size_t pj = 0; pj < pool_size; ++pj) {
                        size_t row_idx = i * stride + pi;
                        size_t col_idx = j * stride + pj;
                        
                        if (row_idx < input_gradient.getRows() && 
                            col_idx < input_gradient.getCols() && 
                            mask(row_idx, col_idx) > 0) {
                            input_gradient(row_idx, col_idx) = gradient(i, j);
                        }
                    }
                }
            }
        }
        
        return input_gradient;
    }
    
    std::string getType() const override {
        return "MaxPooling";
    }
    
    void print() const override {
        std::cout << "MaxPooling Layer: " << pool_size << "x" << pool_size 
                  << " pool, stride=" << stride << std::endl;
    }
};

// Fully Connected (Dense) Layer
class FullyConnectedLayer : public Layer {
private:
    Matrix weights;
    Matrix biases;
    Matrix last_input;
    Matrix weight_gradients;
    Matrix bias_gradients;
    
public:
    FullyConnectedLayer(size_t input_size, size_t output_size) {
        weights = Matrix(output_size, input_size);
        weights.randomizeXavier();
        
        biases = Matrix(output_size, 1);
        biases.zero();
        
        weight_gradients = Matrix(output_size, input_size);
        bias_gradients = Matrix(output_size, 1);
    }
    
    Matrix forward(const Matrix& input) override {
        last_input = input;
        
        // Ensure input is column vector
        Matrix input_vec = input;
        if (input.getCols() != 1) {
            input_vec = input.flatten();
        }
        
        Matrix output = weights * input_vec + biases;
        return output;
    }
    
    Matrix backward(const Matrix& gradient) override {
        // Compute gradients
        Matrix input_vec = last_input;
        if (last_input.getCols() != 1) {
            input_vec = last_input.flatten();
        }
        
        weight_gradients = gradient * input_vec.transpose();
        bias_gradients = gradient;
        
        // Compute input gradient
        Matrix input_gradient = weights.transpose() * gradient;
        
        // Reshape back if needed
        if (last_input.getCols() != 1) {
            input_gradient = input_gradient.reshape(last_input.getRows(), last_input.getCols());
        }
        
        return input_gradient;
    }
    
    void updateWeights(double learning_rate) override {
        // Update weights and biases
        for (size_t i = 0; i < weights.getRows(); ++i) {
            for (size_t j = 0; j < weights.getCols(); ++j) {
                weights(i, j) -= learning_rate * weight_gradients(i, j);
            }
        }
        
        for (size_t i = 0; i < biases.getRows(); ++i) {
            biases(i, 0) -= learning_rate * bias_gradients(i, 0);
        }
    }
    
    std::string getType() const override {
        return "FullyConnected";
    }
    
    void print() const override {
        std::cout << "Fully Connected Layer: " << weights.getCols() 
                  << " -> " << weights.getRows() << std::endl;
    }
};

// Flatten Layer
class FlattenLayer : public Layer {
private:
    size_t original_rows, original_cols;
    
public:
    Matrix forward(const Matrix& input) override {
        original_rows = input.getRows();
        original_cols = input.getCols();
        return input.flatten();
    }
    
    Matrix backward(const Matrix& gradient) override {
        return gradient.reshape(original_rows, original_cols);
    }
    
    std::string getType() const override {
        return "Flatten";
    }
    
    void print() const override {
        std::cout << "Flatten Layer" << std::endl;
    }
};

// Dropout Layer (for regularization)
class DropoutLayer : public Layer {
private:
    double dropout_rate;
    Matrix mask;
    bool training;
    
public:
    DropoutLayer(double rate) : dropout_rate(rate), training(true) {}
    
    void setTraining(bool is_training) {
        training = is_training;
    }
    
    Matrix forward(const Matrix& input) override {
        if (!training) {
            return input;
        }
        
        mask = Matrix(input.getRows(), input.getCols());
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        for (size_t i = 0; i < input.getRows(); ++i) {
            for (size_t j = 0; j < input.getCols(); ++j) {
                mask(i, j) = (dis(gen) > dropout_rate) ? (1.0 / (1.0 - dropout_rate)) : 0.0;
            }
        }
        
        return input.hadamard(mask);
    }
    
    Matrix backward(const Matrix& gradient) override {
        if (!training) {
            return gradient;
        }
        return gradient.hadamard(mask);
    }
    
    std::string getType() const override {
        return "Dropout";
    }
    
    void print() const override {
        std::cout << "Dropout Layer: rate=" << dropout_rate << std::endl;
    }
};

#endif // CNN_LAYERS_H