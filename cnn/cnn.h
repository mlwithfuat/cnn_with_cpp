#ifndef CNN_H
#define CNN_H

#include "matrix.h"
#include "cnn_layers.h"
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <algorithm> // std::shuffle için
#include <chrono>
#include <iomanip>
#include <numeric>   // std::iota için
#include <random>    // std::random_device ve std::mt19937 için

class CNN {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    double learning_rate;
    std::string loss_function;

    // Training metrics
    std::vector<double> train_losses;
    std::vector<double> train_accuracies;
    std::vector<double> val_losses;
    std::vector<double> val_accuracies;

public:
    // Constructor for the CNN model
    CNN(double lr = 0.001, const std::string& loss = "categorical_crossentropy")
        : learning_rate(lr), loss_function(loss) {}

    // Destructor (defaulted as unique_ptr handles memory)
    ~CNN() = default;

    // Functions to add different types of layers to the network
    void addConvolutionalLayer(size_t num_filters, size_t kernel_size,
                               size_t stride = 1, size_t padding = 0) {
        layers.push_back(std::make_unique<ConvolutionalLayer>(num_filters, kernel_size, stride, padding));
    }

    void addActivationLayer(const std::string& activation_type) {
        layers.push_back(std::make_unique<ActivationLayer>(activation_type));
    }

    void addMaxPoolingLayer(size_t pool_size, size_t stride = 0) {
        layers.push_back(std::make_unique<MaxPoolingLayer>(pool_size, stride));
    }

    void addFullyConnectedLayer(size_t input_size, size_t output_size) {
        layers.push_back(std::make_unique<FullyConnectedLayer>(input_size, output_size));
    }

    void addFlattenLayer() {
        layers.push_back(std::make_unique<FlattenLayer>());
    }

    void addDropoutLayer(double dropout_rate) {
        layers.push_back(std::make_unique<DropoutLayer>(dropout_rate));
    }

    // Performs the forward pass through all layers
    Matrix forward(const Matrix& input, bool training = true) {
        Matrix output = input;

        for (auto& layer : layers) {
            // Set training mode for Dropout layer
            if (auto* dropout_layer = dynamic_cast<DropoutLayer*>(layer.get())) {
                dropout_layer->setTraining(training);
            }

            output = layer->forward(output);
        }

        return output;
    }

    // Performs the backward pass (backpropagation) and updates weights
    void backward(const Matrix& predicted, const Matrix& actual) {
        // Compute loss gradient
        Matrix gradient = computeLossGradient(predicted, actual);

        // Backpropagate through layers in reverse order
        for (int i = layers.size() - 1; i >= 0; --i) {
            gradient = layers[i]->backward(gradient);
        }

        // Update weights of all trainable layers
        for (auto& layer : layers) {
            layer->updateWeights(learning_rate);
        }
    }

    // Computes the loss based on the selected loss function
    double computeLoss(const Matrix& predicted, const Matrix& actual) {
        if (loss_function == "categorical_crossentropy") {
            return categoricalCrossentropy(predicted, actual);
        } else if (loss_function == "mean_squared_error") {
            return meanSquaredError(predicted, actual);
        }
        return 0.0; // Default or unsupported loss
    }

    // Computes the gradient of the loss function
    Matrix computeLossGradient(const Matrix& predicted, const Matrix& actual) {
        if (loss_function == "categorical_crossentropy") {
            return categoricalCrossentropyGradient(predicted, actual);
        } else if (loss_function == "mean_squared_error") {
            return meanSquaredErrorGradient(predicted, actual);
        }
        return Matrix(predicted.getRows(), predicted.getCols()); // Empty gradient
    }

    // Calculates categorical crossentropy loss
    double categoricalCrossentropy(const Matrix& predicted, const Matrix& actual) {
        double loss = 0.0;
        const double epsilon = 1e-15; // Small value to prevent log(0)

        for (size_t i = 0; i < predicted.getRows(); ++i) {
            for (size_t j = 0; j < predicted.getCols(); ++j) {
                // Clip predicted values to avoid log(0) or log(negative)
                double p = std::max(epsilon, std::min(1.0 - epsilon, predicted(i, j)));
                loss -= actual(i, j) * std::log(p);
            }
        }

        return loss / predicted.getRows(); // Average loss per sample
    }

    // Calculates the gradient for categorical crossentropy loss
    Matrix categoricalCrossentropyGradient(const Matrix& predicted, const Matrix& actual) {
        Matrix gradient(predicted.getRows(), predicted.getCols());
        const double epsilon = 1e-15;

        for (size_t i = 0; i < predicted.getRows(); ++i) {
            for (size_t j = 0; j < predicted.getCols(); ++j) {
                double p = std::max(epsilon, std::min(1.0 - epsilon, predicted(i, j)));
                gradient(i, j) = (p - actual(i, j)) / predicted.getRows(); // Simplified for softmax output
            }
        }

        return gradient;
    }

    // Calculates mean squared error loss
    double meanSquaredError(const Matrix& predicted, const Matrix& actual) {
        Matrix diff = predicted - actual;
        double sum_sq_diff = 0.0;

        for (size_t i = 0; i < diff.getRows(); ++i) {
            for (size_t j = 0; j < diff.getCols(); ++j) {
                sum_sq_diff += diff(i, j) * diff(i, j);
            }
        }

        return sum_sq_diff / (2.0 * diff.getRows() * diff.getCols()); // Average MSE
    }

    // Calculates the gradient for mean squared error loss
    Matrix meanSquaredErrorGradient(const Matrix& predicted, const Matrix& actual) {
        Matrix gradient = predicted - actual;
        return gradient * (1.0 / (predicted.getRows() * predicted.getCols())); // Derivative is (y_pred - y_true) / N
    }

    // Calculates accuracy (for classification)
    double calculateAccuracy(const std::vector<Matrix>& predictions,
                             const std::vector<Matrix>& labels) {
        if (predictions.size() != labels.size()) {
            return 0.0; // Mismatch in sizes
        }

        int correct = 0;
        int total = predictions.size();

        for (size_t i = 0; i < predictions.size(); ++i) {
            // Find predicted class (index with highest probability)
            size_t pred_class = 0;
            double max_pred = predictions[i](0, 0);
            for (size_t j = 1; j < predictions[i].getRows(); ++j) {
                if (predictions[i](j, 0) > max_pred) {
                    max_pred = predictions[i](j, 0);
                    pred_class = j;
                }
            }

            // Find actual class (index of the one-hot encoded label)
            size_t actual_class = 0;
            double max_actual = labels[i](0, 0);
            for (size_t j = 1; j < labels[i].getRows(); ++j) {
                if (labels[i](j, 0) > max_actual) {
                    max_actual = labels[i](j, 0);
                    actual_class = j;
                }
            }

            if (pred_class == actual_class) {
                correct++;
            }
        }

        return static_cast<double>(correct) / total;
    }

    // Trains the CNN model
    void train(const std::vector<Matrix>& train_data,
               const std::vector<Matrix>& train_labels,
               const std::vector<Matrix>& val_data = {},
               const std::vector<Matrix>& val_labels = {},
               int epochs = 10,
               int batch_size = 32,
               bool verbose = true) {

        if (train_data.size() != train_labels.size()) {
            throw std::invalid_argument("Training data and labels size mismatch");
        }

        size_t num_samples = train_data.size();
        size_t num_batches = (num_samples + batch_size - 1) / batch_size;

        std::cout << "Training CNN..." << std::endl;
        std::cout << "Samples: " << num_samples << ", Batches: " << num_batches
                  << ", Epochs: " << epochs << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Initialize random device for shuffling
        std::random_device rd;
        std::mt19937 g(rd()); // Mersenne Twister engine

        for (int epoch = 0; epoch < epochs; ++epoch) {
            double epoch_loss = 0.0;
            std::vector<Matrix> epoch_predictions; // Store predictions for overall epoch accuracy

            // Create indices for shuffling
            std::vector<size_t> indices(num_samples);
            std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ... num_samples-1

            // Shuffle training data indices
            std::shuffle(indices.begin(), indices.end(), g); // Use std::shuffle with a random engine

            // Training batches
            for (size_t batch = 0; batch < num_batches; ++batch) {
                double batch_loss = 0.0;

                size_t start_idx = batch * batch_size;
                size_t end_idx = std::min(start_idx + batch_size, num_samples);

                // Process samples in the current batch
                for (size_t i = start_idx; i < end_idx; ++i) {
                    size_t current_sample_idx = indices[i]; // Get shuffled index

                    // Forward pass
                    Matrix prediction = forward(train_data[current_sample_idx], true);

                    // Compute loss
                    double loss = computeLoss(prediction, train_labels[current_sample_idx]);
                    batch_loss += loss;
                    epoch_loss += loss;

                    // Backward pass
                    backward(prediction, train_labels[current_sample_idx]);

                    // Store prediction for accuracy calculation (only for the first batch, to avoid excessive memory usage for large datasets)
                    // For more accurate epoch accuracy, you'd collect all predictions.
                    if (batch == 0 && epoch_predictions.size() < batch_size) {
                        epoch_predictions.push_back(prediction);
                    }
                }

                if (verbose && (batch + 1) % 10 == 0) { // Print batch progress every 10 batches
                    std::cout << "\rEpoch " << epoch + 1 << "/" << epochs
                              << " - Batch " << batch + 1 << "/" << num_batches
                              << " - Loss: " << std::fixed << std::setprecision(4)
                              << batch_loss / (end_idx - start_idx) << std::flush;
                }
            }

            // Calculate training metrics for the epoch
            double avg_train_loss = epoch_loss / num_samples;

            // Calculate accuracy on a subset (first batch or full epoch depending on epoch_predictions logic)
            // For a more complete epoch accuracy, you would collect all predictions during the epoch.
            // Here, we're calculating accuracy on the *first* batch of the *shuffled* data to save memory.
            std::vector<Matrix> first_batch_labels_for_accuracy;
            for (size_t i = 0; i < std::min(batch_size, (int)num_samples); ++i) {
                 first_batch_labels_for_accuracy.push_back(train_labels[indices[i]]);
            }
            double train_accuracy = calculateAccuracy(epoch_predictions, first_batch_labels_for_accuracy);

            train_losses.push_back(avg_train_loss);
            train_accuracies.push_back(train_accuracy);

            // Validation
            double val_loss = 0.0;
            double val_accuracy = 0.0;

            if (!val_data.empty() && !val_labels.empty()) {
                std::vector<Matrix> val_predictions;
                val_predictions.reserve(val_data.size()); // Pre-allocate memory

                for (size_t i = 0; i < val_data.size(); ++i) {
                    Matrix prediction = forward(val_data[i], false); // No dropout during validation/test
                    val_loss += computeLoss(prediction, val_labels[i]);
                    val_predictions.push_back(prediction);
                }

                val_loss /= val_data.size();
                val_accuracy = calculateAccuracy(val_predictions, val_labels);

                val_losses.push_back(val_loss);
                val_accuracies.push_back(val_accuracy);
            }

            // Print epoch summary
            if (verbose) {
                std::cout << "\rEpoch " << epoch + 1 << "/" << epochs
                          << " - Loss: " << std::fixed << std::setprecision(4) << avg_train_loss
                          << " - Accuracy: " << std::setprecision(4) << train_accuracy;

                if (!val_data.empty()) {
                    std::cout << " - Val Loss: " << std::setprecision(4) << val_loss
                              << " - Val Accuracy: " << std::setprecision(4) << val_accuracy;
                }
                std::cout << std::endl;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

        std::cout << std::string(60, '=') << std::endl;
        std::cout << "Training completed in " << duration.count() << " seconds" << std::endl;
    }

    // Makes a prediction for a single input matrix
    Matrix predict(const Matrix& input) {
        return forward(input, false); // No dropout during prediction
    }

    // Makes predictions for a batch of input matrices
    std::vector<Matrix> predict(const std::vector<Matrix>& inputs) {
        std::vector<Matrix> predictions;
        predictions.reserve(inputs.size()); // Pre-allocate memory

        for (const auto& input : inputs) {
            predictions.push_back(predict(input));
        }

        return predictions;
    }

    // Evaluates the model on a test dataset
    void evaluate(const std::vector<Matrix>& test_data,
                  const std::vector<Matrix>& test_labels,
                  bool verbose = true) {

        if (test_data.size() != test_labels.size()) {
            throw std::invalid_argument("Test data and labels size mismatch");
        }

        std::cout << "Evaluating model..." << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<Matrix> predictions = predict(test_data); // Get predictions for all test data

        double test_loss = 0.0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            test_loss += computeLoss(predictions[i], test_labels[i]);
        }
        test_loss /= predictions.size(); // Average test loss

        double test_accuracy = calculateAccuracy(predictions, test_labels);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        if (verbose) {
            std::cout << "Test Results:" << std::endl;
            std::cout << "  Loss: " << std::fixed << std::setprecision(4) << test_loss << std::endl;
            std::cout << "  Accuracy: " << std::setprecision(4) << test_accuracy << std::endl;
            std::cout << "  Evaluation time: " << duration.count() << " ms" << std::endl;
        }
    }

    // Prints a summary of the model's architecture
    void summary() const {
        std::cout << "\nModel Summary:" << std::endl;
        std::cout << std::string(50, '=') << std::endl;

        for (size_t i = 0; i < layers.size(); ++i) {
            std::cout << "Layer " << i + 1 << ": ";
            layers[i]->print(); // Call virtual print method for each layer
        }

        std::cout << std::string(50, '=') << std::endl;
        std::cout << "Learning Rate: " << learning_rate << std::endl;
        std::cout << "Loss Function: " << loss_function << std::endl;
        std::cout << "Total Layers: " << layers.size() << std::endl;
    }

    // Getters for training history metrics
    const std::vector<double>& getTrainLosses() const { return train_losses; }
    const std::vector<double>& getTrainAccuracies() const { return train_accuracies; }
    const std::vector<double>& getValLosses() const { return val_losses; }
    const std::vector<double>& getValAccuracies() const { return val_accuracies; }

    // Simplified model saving (only architecture and basic params, not weights)
    void saveModel(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for saving: " + filename);
        }

        // Save basic parameters
        file.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));

        size_t loss_size = loss_function.size();
        file.write(reinterpret_cast<const char*>(&loss_size), sizeof(loss_size));
        file.write(loss_function.c_str(), loss_size);

        // Save number of layers
        size_t num_layers = layers.size();
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

        std::cout << "Model architecture saved to " << filename << std::endl;
        std::cout << "Note: Weight saving not implemented in this version" << std::endl;

        file.close();
    }

    // Setter for learning rate
    void setLearningRate(double lr) {
        learning_rate = lr;
    }

    // Getter for learning rate
    double getLearningRate() const {
        return learning_rate;
    }
};

#endif // CNN_H
