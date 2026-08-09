#include <tensorflow_classifier.hpp>

#include <../include/csv_reader.hpp>

#include <tensorflow/c/c_api.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace fashion_mnist {
    namespace {
        void require_ok(const TF_Status *status, const std::string &action) {
            if (TF_GetCode(status) != TF_OK) {
                throw std::runtime_error{action + ": " + TF_Message(status)};
            }
        }

        TF_Operation *find_operation(TF_Graph *graph, const char *name, const char *kind) {
            if (TF_Operation *operation = TF_GraphOperationByName(graph, name); operation != nullptr) {
                return operation;
            }
            throw std::runtime_error{
                std::string{"SavedModel "} + kind + " operation '" + name + "' was not found"
            };
        }
    } // namespace

    class TensorFlowClassifier::PImpl {
        using GraphPtr = std::unique_ptr<
            TF_Graph,
            decltype([](TF_Graph *graph) noexcept { TF_DeleteGraph(graph); })>;

        using SessionOptionsPtr = std::unique_ptr<
            TF_SessionOptions,
            decltype([](TF_SessionOptions *options) noexcept { TF_DeleteSessionOptions(options); })>;

        using StatusPtr = std::unique_ptr<
            TF_Status,
            decltype([](TF_Status *status) noexcept { TF_DeleteStatus(status); })>;

        using TensorPtr = std::unique_ptr<
            TF_Tensor,
            decltype([](TF_Tensor *tensor) noexcept { TF_DeleteTensor(tensor); })>;

        using SessionPtr = std::unique_ptr<
            TF_Session,
            decltype([](TF_Session *session) noexcept {
                if (session == nullptr) {
                    return;
                }
                const StatusPtr status{TF_NewStatus()};
                TF_CloseSession(session, status.get());
                TF_DeleteSession(session, status.get());
            })>;

    public:
        explicit PImpl(const std::string &model_directory)
            : graph_{TF_NewGraph()},
              session_options_{TF_NewSessionOptions()},
              session_{load_session(session_options_.get(), model_directory, graph_.get())},
              input_operation_{
                  find_operation(graph_.get(), "serving_default_input", "input")
              },
              output_operation_{
                  find_operation(graph_.get(), "StatefulPartitionedCall", "output")
              } {
            if (TF_OperationOutputType({.oper = input_operation_, .index = 0}) != TF_FLOAT ||
                TF_OperationOutputType({.oper = output_operation_, .index = 0}) != TF_FLOAT) {
                throw std::runtime_error{"SavedModel input and output tensors must use float values"};
            }
        }

        ~PImpl() = default;

        PImpl(const PImpl &) = delete;

        PImpl &operator=(const PImpl &) = delete;

        PImpl(PImpl &&) = delete;

        PImpl &operator=(PImpl &&) = delete;

        [[nodiscard]] std::vector<unsigned int> predict_batch(const std::span<const float> normalized_pixels) const {
            if (normalized_pixels.empty() || normalized_pixels.size() % kPixelCount != 0) {
                throw std::invalid_argument{"Input must contain one or more complete 28x28 images"};
            }

            const std::size_t batch_size = normalized_pixels.size() / kPixelCount;
            const std::array<int64_t, 4> dimensions = {
                static_cast<int64_t>(batch_size),
                static_cast<int64_t>(kImageHeight),
                static_cast<int64_t>(kImageWidth),
                1
            };
            const std::size_t byte_count = normalized_pixels.size_bytes();
            const TensorPtr input{TF_AllocateTensor(TF_FLOAT, dimensions.data(), 4, byte_count)};
            if (!input) {
                throw std::runtime_error{"Unable to allocate the TensorFlow input tensor"};
            }

            if (void *const input_data = TF_TensorData(input.get()); input_data) {
                std::ranges::copy(normalized_pixels, static_cast<float *>(input_data));
            } else {
                throw std::runtime_error{"TensorFlow input tensor has no data buffer"};
            }

            const TF_Output input_port{.oper = input_operation_, .index = 0};
            const TF_Output output_port{.oper = output_operation_, .index = 0};
            TF_Tensor *input_value = input.get();
            TensorPtr output;
            const StatusPtr status{TF_NewStatus()};

            TF_SessionRun(session_.get(),
                          nullptr,
                          &input_port,
                          &input_value,
                          1,
                          &output_port,
                          std::out_ptr(output),
                          1,
                          nullptr,
                          0,
                          nullptr,
                          status.get());
            require_ok(status.get(), "Unable to run TensorFlow inference");

            if (!output || TF_TensorType(output.get()) != TF_FLOAT ||
                TF_NumDims(output.get()) != 2 ||
                TF_Dim(output.get(), 0) != static_cast<int64_t>(batch_size) ||
                TF_Dim(output.get(), 1) != static_cast<int64_t>(kClassCount) ||
                TF_TensorByteSize(output.get()) < batch_size * kClassCount * sizeof(float)) {
                throw std::runtime_error{"SavedModel returned an unexpected output tensor shape"};
            }

            const std::span probabilities{
                static_cast<const float *>(TF_TensorData(output.get())), batch_size * kClassCount
            };
            std::vector<unsigned int> predictions;
            predictions.reserve(batch_size);
            for (std::size_t image = 0; image < batch_size; ++image) {
                const auto scores = probabilities.subspan(image * kClassCount, kClassCount);
                const auto best_score = std::ranges::max_element(scores);
                const auto predicted_class = std::ranges::distance(scores.begin(), best_score);
                predictions.push_back(predicted_class);
            }
            return predictions;
        }

    private:
        static SessionPtr load_session(const TF_SessionOptions *session_options,
                                       const std::string &model_directory,
                                       TF_Graph *graph) {
            if (session_options == nullptr || graph == nullptr) {
                throw std::runtime_error{"Unable to allocate TensorFlow objects"};
            }

            const StatusPtr status{TF_NewStatus()};
            const char *tags[] = {"serve"};
            SessionPtr session{
                TF_LoadSessionFromSavedModel(session_options,
                                             nullptr,
                                             model_directory.c_str(),
                                             tags,
                                             1,
                                             graph,
                                             nullptr,
                                             status.get())
            };
            require_ok(status.get(), std::format("Unable to load SavedModel from '{}", model_directory));
            if (!session) {
                throw std::runtime_error{"TensorFlow returned an empty session"};
            }
            return session;
        }

        const GraphPtr graph_;
        const SessionOptionsPtr session_options_;
        const SessionPtr session_;
        TF_Operation *const input_operation_;
        TF_Operation *const output_operation_;
    };

    TensorFlowClassifier::TensorFlowClassifier(const std::string &model_directory)
        : pImpl_{std::make_unique<PImpl>(model_directory)} {
    }

    TensorFlowClassifier::~TensorFlowClassifier() = default;

    TensorFlowClassifier::TensorFlowClassifier(TensorFlowClassifier &&) noexcept = default;

    TensorFlowClassifier &TensorFlowClassifier::operator=(TensorFlowClassifier &&) noexcept = default;

    std::vector<unsigned int>
    TensorFlowClassifier::predict_batch(const std::span<const float> normalized_pixels) const {
        return pImpl_->predict_batch(normalized_pixels);
    }
} // namespace fashion_mnist
