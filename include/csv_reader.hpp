#pragma once

/**
 * @file csv_reader.hpp
 * @brief Чтение размеченных изображений Fashion-MNIST из CSV-файла.
 */

#include <cstddef>
#include <istream>
#include <optional>
#include <vector>

/**
 * @namespace fashion_mnist
 * @brief Компоненты приложения для классификации изображений Fashion-MNIST.
 */
namespace fashion_mnist {

/** @brief Ширина изображения Fashion-MNIST в пикселях. */
inline constexpr std::size_t kImageWidth = 28;

/** @brief Высота изображения Fashion-MNIST в пикселях. */
inline constexpr std::size_t kImageHeight = 28;

/** @brief Количество пикселей в одном изображении Fashion-MNIST. */
inline constexpr std::size_t kPixelCount = kImageWidth * kImageHeight;

/** @brief Количество классов в наборе данных Fashion-MNIST. */
inline constexpr std::size_t kClassCount = 10;

/**
 * @brief Пакет размеченных изображений, подготовленный для инференса.
 *
 * Метка с индексом `i` относится к изображению `i`. Пиксели изображений
 * расположены последовательно, поэтому пакет всегда удовлетворяет условию
 * `pixels.size() == labels.size() * kPixelCount`.
 */
struct Batch {
    /** @brief Эталонные метки классов изображений. */
    std::vector<unsigned int> labels;

    /**
     * @brief Нормализованные пиксели изображений.
     *
     * Значения лежат в диапазоне `[0, 1]` для соответствия TF_FLOAT. Каждое изображение хранится
     * построчно в виде `kPixelCount` последовательных значений.
     */
    std::vector<float> pixels;
};

/**
 * @brief Потоковый читатель размеченного набора Fashion-MNIST в формате CSV.
 *
 * В каждой строке ожидается 785 целых чисел без заголовка: метка класса и
 * 784 значения пикселей. Читатель не владеет входным потоком, поэтому поток
 * должен существовать в течение всего времени использования объекта.
 */
class CsvReader {
public:
    /**
     * @brief Создаёт читатель поверх заданного входного потока.
     * @param input Поток с данными CSV без строки заголовка.
     */
    explicit CsvReader(std::istream& input);

    /**
     * @brief Читает и нормализует очередной пакет изображений.
     *
     * Последний пакет может содержать меньше `max_samples` изображений.
     * Чистое достижение конца потока до чтения новой строки не считается
     * ошибкой.
     *
     * @param max_samples Максимальное количество изображений в пакете.
     * @return Прочитанный непустой пакет либо `std::nullopt`, если данные
     * закончились.
     * @throws std::invalid_argument Если `max_samples` равен нулю.
     * @throws std::runtime_error Если поток повреждён или строка CSV имеет
     * неверное количество столбцов, некорректные числа либо значения вне
     * допустимого диапазона.
     */
    [[nodiscard]] std::optional<Batch> read_batch(std::size_t max_samples);

    /**
     * @brief Возвращает номер последней строки, извлеченной из потока.
     * @return Номер строки, начиная с единицы, либо `0` до начала чтения.
     */
    [[nodiscard]] std::size_t line_number() const noexcept;

private:
    std::istream& input_;
    std::size_t line_number_ = 0;
};

}  // namespace fashion_mnist
