#include <fashion_mnist/csv_reader.hpp>

#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <print>

namespace {

constexpr std::size_t kBatchSize = 256;

int run(const char* dataset_path, [[maybe_unused]]const char* model_path) {
    std::ifstream dataset{dataset_path};
    if (!dataset.is_open()) {
        throw std::runtime_error{std::string{"Unable to open test data file: "} + dataset_path};
    }

    fashion_mnist::CsvReader reader{dataset};

    std::size_t correct = 0;
    std::size_t total = 0;

    while (auto batch = reader.read_batch(kBatchSize)) {
        const auto predictions = std::vector<float>(); //classifier.predict_batch(batch->pixels);
        for (std::size_t index = 0; index < predictions.size(); ++index) {
            correct += predictions[index] == batch->labels[index] ? 1U : 0U;
        }
        total += batch->labels.size();
    }

    if (total == 0) {
        throw std::runtime_error{"The test data file is empty"};
    }

    const double accuracy = static_cast<double>(correct) / static_cast<double>(total);
    std::println("{:.6f}", accuracy);
    return EXIT_SUCCESS;
}

}  // namespace

int main(const int argc, char* argv[]) {
    if (argc != 3) {
        std::println("Usage: {} <test.csv> <saved_model_directory>", argv[0]);
        return 2;
    }

    try {
        return run(argv[1], argv[2]);
    } catch (const std::exception& error) {
        std::println(std::cerr, "Error: {}", error.what());
        return EXIT_FAILURE;
    }
}
