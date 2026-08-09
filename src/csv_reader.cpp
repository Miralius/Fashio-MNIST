#include <../include/csv_reader.hpp>

#include <array>
#include <charconv>
#include <expected>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

namespace fashion_mnist {
    namespace {
        struct Sample {
            unsigned int label;
            std::array<float, kPixelCount> pixels;
        };

        constexpr bool is_ascii_space(const char ch) noexcept {
            return std::string_view{" \t\n\r\f\v"}.contains(ch);
        }

        std::string_view trim(std::string_view value) noexcept {
            while (!value.empty() && is_ascii_space(value.front()) != 0) {
                value.remove_prefix(1);
            }
            while (!value.empty() && is_ascii_space(value.back()) != 0) {
                value.remove_suffix(1);
            }
            return value;
        }

        std::expected<unsigned int, std::string> parse_integer(
            std::string_view field,
            const std::size_t line,
            const std::size_t column) {
            field = trim(field);
            unsigned int value = 0;
            if (const auto [ptr, ec] = std::from_chars(
                    field.data(), field.data() + field.size(), value);
                field.empty() || ec != std::errc{} || ptr != field.data() + field.size()) {
                return std::unexpected{std::format("Invalid integer at line {}, column {}", line, column)};
            }
            return value;
        }

        Sample parse_line(std::string_view line, const std::size_t line_number) {
            Sample sample{};
            for (std::size_t column = 1; column <= kPixelCount + 1; ++column) {
                const std::size_t separator = line.find(',');

                if (const bool separator_expected = column <= kPixelCount;
                    (separator != std::string_view::npos) != separator_expected) {
                    throw std::runtime_error{
                        "Expected 785 columns at line " + std::to_string(line_number)
                    };
                }

                const std::string_view field = separator == std::string_view::npos
                                                   ? line
                                                   : line.substr(0, separator);
                const auto value_result = parse_integer(field, line_number, column);
                if (!value_result) {
                    throw std::runtime_error{value_result.error()};
                }

                if (column == 1) {
                    if (value_result.value() >= kClassCount) {
                        throw std::runtime_error{
                            "Class label is outside [0, 9] at line " +
                            std::to_string(line_number)
                        };
                    }
                    sample.label = value_result.value();
                } else {
                    if (value_result.value() > 255) {
                        throw std::runtime_error{
                            "Pixel value is outside [0, 255] at line " +
                            std::to_string(line_number) + ", column " +
                            std::to_string(column)
                        };
                    }
                    sample.pixels[column - 2] =
                            static_cast<float>(value_result.value()) / 255.0F;
                }

                if (separator == std::string_view::npos) {
                    line = {};
                } else {
                    line.remove_prefix(separator + 1);
                }
            }
            return sample;
        }
    } // namespace

    CsvReader::CsvReader(std::istream &input) : input_{input} {
    }

    std::optional<Batch> CsvReader::read_batch(const std::size_t max_samples) {
        if (max_samples == 0) {
            throw std::invalid_argument{"Batch size must be greater than zero"};
        }

        Batch batch;
        batch.labels.reserve(max_samples);
        batch.pixels.reserve(max_samples * kPixelCount);

        std::string line;
        while (batch.labels.size() < max_samples && std::getline(input_, line)) {
            ++line_number_;
            if (line.empty()) {
                throw std::runtime_error{"Empty row at line " + std::to_string(line_number_)};
            }
            const auto [label, pixels] = parse_line(line, line_number_);
            batch.labels.push_back(label);
            batch.pixels.insert(batch.pixels.end(), pixels.begin(), pixels.end());
        }

        if (input_.bad()) {
            throw std::runtime_error{"Unable to read the test data stream"};
        }
        if (batch.labels.empty()) {
            return std::nullopt;
        }
        return batch;
    }

    std::size_t CsvReader::line_number() const noexcept {
        return line_number_;
    }
} // namespace fashion_mnist
