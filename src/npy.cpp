#include "npy.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <utility>
#include <vector>

namespace {

constexpr unsigned char kMagic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};

std::uint32_t read_u16_le(const unsigned char* bytes) {
    return static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8);
}

std::uint32_t read_u32_le(const unsigned char* bytes) {
    return static_cast<std::uint32_t>(bytes[0])
        | (static_cast<std::uint32_t>(bytes[1]) << 8)
        | (static_cast<std::uint32_t>(bytes[2]) << 16)
        | (static_cast<std::uint32_t>(bytes[3]) << 24);
}

bool header_span(const std::vector<unsigned char>& file, std::size_t& begin, std::size_t& end, std::string& error) {
    if (file.size() < 10 || std::memcmp(file.data(), kMagic, 6) != 0) {
        error = "файл не похож на npy";
        return false;
    }
    const int major = file[6];
    std::size_t header_len = 0;
    if (major == 1) {
        header_len = read_u16_le(file.data() + 8);
        begin = 10;
    } else if (major == 2 || major == 3) {
        if (file.size() < 12) {
            error = "заголовок npy обрезан";
            return false;
        }
        header_len = read_u32_le(file.data() + 8);
        begin = 12;
    } else {
        error = "неподдерживаемая версия npy";
        return false;
    }
    end = begin + header_len;
    if (header_len == 0 || end > file.size()) {
        error = "заголовок npy обрезан";
        return false;
    }
    return true;
}

std::string quoted_after(const std::string& header, const std::string& key) {
    const auto key_at = header.find(key);
    if (key_at == std::string::npos) {
        return {};
    }
    const auto colon = header.find(':', key_at + key.size());
    if (colon == std::string::npos) {
        return {};
    }
    const auto quote = header.find_first_of("\'\"", colon + 1);
    if (quote == std::string::npos) {
        return {};
    }
    const char mark = header[quote];
    const auto close = header.find(mark, quote + 1);
    if (close == std::string::npos) {
        return {};
    }
    return header.substr(quote + 1, close - quote - 1);
}

bool parse_shape(const std::string& header, std::vector<int>& dims, std::string& error) {
    const auto shape_at = header.find("shape");
    const auto open = shape_at == std::string::npos ? std::string::npos : header.find('(', shape_at);
    const auto close = open == std::string::npos ? std::string::npos : header.find(')', open);
    if (close == std::string::npos) {
        error = "в заголовке npy нет shape";
        return false;
    }
    const char* cursor = header.c_str() + open + 1;
    const char* stop = header.c_str() + close;
    while (cursor < stop) {
        while (cursor < stop && (*cursor < '0' || *cursor > '9') && *cursor != '-') {
            ++cursor;
        }
        if (cursor >= stop) {
            break;
        }
        char* next = nullptr;
        const long value = std::strtol(cursor, &next, 10);
        if (next == cursor) {
            break;
        }
        dims.push_back(static_cast<int>(value));
        cursor = next;
    }
    return true;
}

}  // namespace

bool load_npy(const std::string& path, Volume& volume, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "не удалось открыть " + path;
        return false;
    }
    std::vector<unsigned char> file(std::istreambuf_iterator<char>(in), {});
    if (!in && !in.eof()) {
        error = "не удалось прочитать " + path;
        return false;
    }

    std::size_t begin = 0;
    std::size_t end = 0;
    if (!header_span(file, begin, end, error)) {
        return false;
    }
    const std::string header(file.begin() + static_cast<std::ptrdiff_t>(begin),
                             file.begin() + static_cast<std::ptrdiff_t>(end));

    const std::string descr = quoted_after(header, "descr");
    if (descr != "<f4" && descr != "|f4") {
        error = "нужен float32 ('<f4'), в файле '" + descr + "'";
        return false;
    }

    const auto fortran_at = header.find("fortran_order");
    if (fortran_at == std::string::npos) {
        error = "в заголовке npy нет fortran_order";
        return false;
    }
    if (header.substr(fortran_at, 40).find("True") != std::string::npos) {
        error = "порядок fortran в npy не поддерживается";
        return false;
    }

    std::vector<int> dims;
    if (!parse_shape(header, dims, error)) {
        return false;
    }
    if (dims.size() != 3) {
        error = "нужен куб из трёх измерений, в файле " + std::to_string(dims.size());
        return false;
    }
    if (dims[0] != dims[1] || dims[1] != dims[2] || dims[0] < 1 || dims[0] > Volume::kMaxSize) {
        error = "нужен куб N×N×N со стороной от 1 до 512";
        return false;
    }

    const int n = dims[0];
    const std::uint64_t voxels = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(n);
    const std::uint64_t nbytes = voxels * sizeof(float);
    if (file.size() < end + nbytes) {
        error = "файл npy короче, чем обещает заголовок";
        return false;
    }

    Volume loaded(n);
    std::memcpy(loaded.data().data(), file.data() + end, static_cast<std::size_t>(nbytes));
    volume = std::move(loaded);
    return true;
}
