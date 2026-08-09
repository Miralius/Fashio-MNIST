#pragma once

#include <experimental/propagate_const>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace fashion_mnist {

class TensorFlowClassifier {
public:
    explicit TensorFlowClassifier(const std::string& model_directory);
    ~TensorFlowClassifier();

    TensorFlowClassifier(const TensorFlowClassifier&) = delete;
    TensorFlowClassifier& operator=(const TensorFlowClassifier&) = delete;
    TensorFlowClassifier(TensorFlowClassifier&&) noexcept;
    TensorFlowClassifier& operator=(TensorFlowClassifier&&) noexcept;

    // normalized_pixels contains batch_size consecutive 28x28 images.
    [[nodiscard]] std::vector<unsigned int> predict_batch(
            std::span<const float> normalized_pixels) const;

private:
    class PImpl;
    std::experimental::propagate_const<std::unique_ptr<PImpl>> pImpl_;
};

}  // namespace fashion_mnist
