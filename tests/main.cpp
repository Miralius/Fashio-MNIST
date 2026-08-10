#include <gtest/gtest.h>

#include "csv_reader.hpp"
#include "tensorflow_classifier.hpp"

#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

static_assert(!std::is_copy_constructible_v<fashion_mnist::TensorFlowClassifier>);
static_assert(!std::is_copy_assignable_v<fashion_mnist::TensorFlowClassifier>);
static_assert(std::is_nothrow_move_constructible_v<fashion_mnist::TensorFlowClassifier>);
static_assert(std::is_nothrow_move_assignable_v<fashion_mnist::TensorFlowClassifier>);

std::string make_row(const int label, const int pixel = 0) {
    std::ostringstream row;
    row << label;
    for (std::size_t index = 0; index < fashion_mnist::kPixelCount; ++index) {
        row << ',' << pixel;
    }
    return row.str();
}

TEST(CsvReader, ReadsBatchesAndNormalizesPixels) {
    std::istringstream input{make_row(3, 255) + "\n" + make_row(7, 0) + "\n"};
    fashion_mnist::CsvReader reader{input};

    const auto first_batch = reader.read_batch(1);
    ASSERT_TRUE(first_batch.has_value());
    ASSERT_EQ(first_batch->labels, std::vector{3U});
    ASSERT_EQ(first_batch->pixels.size(), fashion_mnist::kPixelCount);
    EXPECT_FLOAT_EQ(first_batch->pixels.front(), 1.0F);

    const auto second_batch = reader.read_batch(4);
    ASSERT_TRUE(second_batch.has_value());
    ASSERT_EQ(second_batch->labels, std::vector{7U});
    EXPECT_FLOAT_EQ(second_batch->pixels.front(), 0.0F);
    EXPECT_FALSE(reader.read_batch(4).has_value());
    EXPECT_EQ(reader.line_number(), 2U);
}

TEST(CsvReader, EmptyInputReturnsNullopt) {
    std::istringstream input;
    fashion_mnist::CsvReader reader{input};

    EXPECT_FALSE(reader.read_batch(1).has_value());
}

TEST(CsvReader, RejectsMalformedRows) {
    std::istringstream input{"3,0,0\n"};
    fashion_mnist::CsvReader reader{input};

    EXPECT_THROW(static_cast<void>(reader.read_batch(1)), std::runtime_error);
}

TEST(CsvReader, RejectsExtraColumns) {
    std::istringstream input{make_row(3) + ",0\n"};
    fashion_mnist::CsvReader reader{input};

    EXPECT_THROW(static_cast<void>(reader.read_batch(1)), std::runtime_error);
}

TEST(CsvReader, RejectsOutOfRangeValues) {
    std::istringstream input{make_row(3, 256) + "\n"};
    fashion_mnist::CsvReader reader{input};

    EXPECT_THROW(static_cast<void>(reader.read_batch(1)), std::runtime_error);
}

TEST(CsvReader, RejectsNegativeValues) {
    std::istringstream input{make_row(3, -1) + "\n"};
    fashion_mnist::CsvReader reader{input};

    EXPECT_THROW(static_cast<void>(reader.read_batch(1)), std::runtime_error);
}

}  // namespace

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
