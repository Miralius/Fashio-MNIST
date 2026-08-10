#pragma once

/**
 * @file tensorflow_classifier.hpp
 * @brief Классификация изображений Fashion-MNIST с помощью TensorFlow SavedModel.
 */

#include <experimental/propagate_const>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace fashion_mnist {
    /**
     * @brief Классификатор изображений Fashion-MNIST на основе TensorFlow SavedModel.
     *
     * При создании объекта модель загружается из заданного каталога и остаётся
     * доступной для инференса в течение всего времени жизни классификатора.
     * Классификатор не копируется, но поддерживает перемещение.
     *
     * Ожидается, что модель содержит вход `serving_default_input` типа `TF_FLOAT`
     * и выход `StatefulPartitionedCall` типа `TF_FLOAT`. Вход модели должен иметь
     * форму `[batch_size, 28, 28, 1]`, а выход — `[batch_size, 10]`.
     */
    class TensorFlowClassifier {
    public:
        /**
         * @brief Загружает TensorFlow SavedModel.
         * @param model_directory Путь к каталогу с сохранённой моделью.
         * @throws std::runtime_error Если объекты TensorFlow не удалось создать,
         * модель не удалось загрузить или её входы и выходы не соответствуют
         * ожидаемому интерфейсу.
         */
        explicit TensorFlowClassifier(const std::string &model_directory);

        /** @brief Освобождает ресурсы TensorFlow, связанные с моделью. */
        ~TensorFlowClassifier();

        TensorFlowClassifier(const TensorFlowClassifier &) = delete;

        TensorFlowClassifier &operator=(const TensorFlowClassifier &) = delete;

        TensorFlowClassifier(TensorFlowClassifier &&) noexcept;

        TensorFlowClassifier &operator=(TensorFlowClassifier &&) noexcept;

        /**
         * @brief Определяет классы для пакета изображений.
         *
         * Пиксели изображений должны быть нормализованы и расположены подряд:
         * каждые 784 последовательных значения образуют одно изображение 28×28.
         *
         * @param normalized_pixels Пиксели одного или нескольких изображений.
         * @return Индексы наиболее вероятных классов в диапазоне `[0, 9]` в том
         * же порядке, в котором изображения переданы во входном диапазоне.
         * @throws std::invalid_argument Если диапазон пуст или его размер не
         * кратен 784.
         * @throws std::runtime_error Если не удалось выполнить инференс либо
         * модель вернула тензор неожиданного типа или размера.
         */
        [[nodiscard]] std::vector<unsigned int> predict_batch(
            std::span<const float> normalized_pixels) const;

    private:
        class PImpl;
        std::experimental::propagate_const<std::unique_ptr<PImpl> > pImpl_;
    };
} // namespace fashion_mnist
