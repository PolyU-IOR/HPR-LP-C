#include "cpu_presolve/mpsreader.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cpu_presolve {
namespace {

enum class Section {
  kNone,
  kObjSense,
  kRows,
  kColumns,
  kRhs,
  kBounds,
  kRanges,
  kUnsupportedQuadratic,
};

enum class Format {
  kFree,
  kFixed,
};

struct Card {
  int line = 0;
  bool is_comment = false;
  bool is_header = false;
  bool missing_name_field = false;
  std::vector<std::string> fields;
};

struct RowInfo {
  char type = 'N';
  int index = -1;
};

enum class VarType {
  kContinuous,
  kMarkedInteger,
  kInteger,
  kBinary,
};

struct RawModel {
  std::string name;
  std::string objective_row;
  std::unordered_map<std::string, RowInfo> rows;
  std::unordered_map<std::string, int> cols;
  std::vector<char> row_types;
  std::vector<std::vector<std::pair<int, double>>> col_entries;
  std::vector<double> objective;
  std::vector<double> row_lower;
  std::vector<double> row_upper;
  std::vector<double> col_lower;
  std::vector<double> col_upper;
  std::vector<VarType> var_types;
  double obj_constant = 0.0;
};

std::string trim_copy(std::string value) {
  const auto begin = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) {
    return std::isspace(ch) != 0;
  });
  const auto end = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
    return std::isspace(ch) != 0;
  }).base();
  if (begin >= end) {
    return {};
  }
  return std::string(begin, end);
}

std::string field_slice(const std::string& line, int begin, int end) {
  if (static_cast<int>(line.size()) <= begin) {
    return {};
  }
  const int stop = std::min(static_cast<int>(line.size()), end);
  return trim_copy(line.substr(static_cast<std::size_t>(begin),
                               static_cast<std::size_t>(stop - begin)));
}

std::vector<std::string> split_line(const std::string& line) {
  std::istringstream stream(line);
  std::vector<std::string> tokens;
  std::string token;
  while (stream >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

bool is_blank_or_comment(const std::string& line) {
  for (char ch : line) {
    if (std::isspace(static_cast<unsigned char>(ch)) != 0) {
      continue;
    }
    return ch == '*' || ch == '&';
  }
  return true;
}

bool is_section_header(const std::string& token) {
  return token == "NAME" || token == "OBJSENSE" || token == "ROWS" ||
         token == "COLUMNS" || token == "RHS" || token == "BOUNDS" ||
         token == "RANGES" || token == "QUADOBJ" || token == "QMATRIX" ||
         token == "ENDATA";
}

Card read_card_free(const std::string& line, int line_number) {
  Card card;
  card.line = line_number;
  if (is_blank_or_comment(line)) {
    card.is_comment = true;
    return card;
  }

  card.fields = split_line(line);
  if (!line.empty() && std::isspace(static_cast<unsigned char>(line[0])) == 0) {
    card.is_header = true;
    if (card.fields.size() >= 2 && card.fields[0] == "OBJECT" && card.fields[1] == "BOUND") {
      card.fields = {"OBJECT BOUND"};
    }
  }
  return card;
}

Card read_card_fixed(const std::string& line, int line_number) {
  Card card;
  card.line = line_number;
  if (is_blank_or_comment(line)) {
    card.is_comment = true;
    return card;
  }

  if (!line.empty() && std::isspace(static_cast<unsigned char>(line[0])) == 0) {
    card.is_header = true;
    card.fields = split_line(line);
    if (!card.fields.empty() && card.fields[0] == "NAME") {
      const std::string fixed_name = field_slice(line, 14, static_cast<int>(line.size()));
      if (!fixed_name.empty()) {
        card.fields = {"NAME", fixed_name};
      }
    } else if (card.fields.size() >= 2 && card.fields[0] == "OBJECT" &&
               card.fields[1] == "BOUND") {
      card.fields = {"OBJECT BOUND"};
    }
    return card;
  }

  const std::string name_field = field_slice(line, 4, 12);
  card.missing_name_field = name_field.empty();
  std::vector<std::string> fields;
  for (const auto& field : {
           field_slice(line, 1, 3),
           name_field,
           field_slice(line, 14, 22),
           field_slice(line, 24, 36),
           field_slice(line, 39, 47),
           field_slice(line, 49, 61),
       }) {
    if (!field.empty()) {
      fields.push_back(field);
    }
  }
  card.fields = std::move(fields);
  return card;
}

double parse_double_fallback(const std::string& token) {
  std::string normalized = token;
  for (char& ch : normalized) {
    if (ch == 'd' || ch == 'D') {
      ch = 'E';
    }
  }
  std::size_t consumed = 0;
  double value = 0.0;
  try {
    value = std::stod(normalized, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument("invalid numeric token in MPS: " + token);
  }
  if (consumed != normalized.size()) {
    throw std::invalid_argument("invalid numeric token in MPS: " + token);
  }
  return value;
}

double parse_double(const std::string& token) {
  const std::size_t n = token.size();
  std::size_t i = 0;
  bool negative = false;

  if (i < n) {
    const char ch = token[i];
    if (ch == '-') {
      negative = true;
      ++i;
    } else if (ch == '+') {
      ++i;
    }
  }

  double value = 0.0;
  double frac_scale = 0.1;
  int ndigits = 0;
  bool seen_digit = false;

  auto is_digit = [](char ch) {
    return ch >= '0' && ch <= '9';
  };

  while (i < n) {
    const char ch = token[i];
    if (!is_digit(ch)) {
      break;
    }
    value = value * 10.0 + static_cast<double>(ch - '0');
    ++ndigits;
    seen_digit = true;
    if (ndigits > 18) {
      return parse_double_fallback(token);
    }
    ++i;
  }

  if (i < n && token[i] == '.') {
    ++i;
    while (i < n) {
      const char ch = token[i];
      if (!is_digit(ch)) {
        break;
      }
      value += static_cast<double>(ch - '0') * frac_scale;
      frac_scale *= 0.1;
      ++ndigits;
      seen_digit = true;
      if (ndigits > 18) {
        return parse_double_fallback(token);
      }
      ++i;
    }
  }

  if (!seen_digit) {
    return parse_double_fallback(token);
  }

  if (i < n) {
    const char ch = token[i];
    if (ch == 'e' || ch == 'E' || ch == 'd' || ch == 'D') {
      ++i;
      bool exp_negative = false;
      if (i < n) {
        const char sign = token[i];
        if (sign == '-') {
          exp_negative = true;
          ++i;
        } else if (sign == '+') {
          ++i;
        }
      }

      int exp_value = 0;
      int exp_digits = 0;
      while (i < n) {
        const char digit = token[i];
        if (!is_digit(digit)) {
          return parse_double_fallback(token);
        }
        exp_value = 10 * exp_value + static_cast<int>(digit - '0');
        ++exp_digits;
        ++i;
      }
      if (exp_digits == 0) {
        return parse_double_fallback(token);
      }
      const double scale = std::pow(10.0, exp_value);
      value = exp_negative ? value / scale : value * scale;
    } else {
      return parse_double_fallback(token);
    }
  }

  return negative ? -value : value;
}

int ensure_col(RawModel& model, const std::string& col_name, bool integer_section) {
  const auto found = model.cols.find(col_name);
  if (found != model.cols.end()) {
    return found->second;
  }

  const int index = static_cast<int>(model.cols.size());
  model.cols.emplace(col_name, index);
  model.col_entries.emplace_back();
  model.objective.push_back(0.0);
  model.col_lower.push_back(std::numeric_limits<double>::quiet_NaN());
  model.col_upper.push_back(std::numeric_limits<double>::quiet_NaN());
  model.var_types.push_back(integer_section ? VarType::kMarkedInteger : VarType::kContinuous);
  return index;
}

Section header_to_section(const std::string& header) {
  if (header == "OBJSENSE") {
    return Section::kObjSense;
  }
  if (header == "ROWS") {
    return Section::kRows;
  }
  if (header == "COLUMNS") {
    return Section::kColumns;
  }
  if (header == "RHS") {
    return Section::kRhs;
  }
  if (header == "BOUNDS") {
    return Section::kBounds;
  }
  if (header == "RANGES") {
    return Section::kRanges;
  }
  if (header == "QUADOBJ" || header == "QMATRIX") {
    return Section::kUnsupportedQuadratic;
  }
  return Section::kNone;
}

void read_rows_line(RawModel& model, const Card& card) {
  if (card.fields.size() < 2) {
    throw std::invalid_argument("invalid ROWS entry at line " + std::to_string(card.line));
  }
  const char type = card.fields[0].empty() ? '\0' : card.fields[0][0];
  const std::string& row_name = card.fields[1];
  if (type == 'N') {
    if (model.objective_row.empty()) {
      model.objective_row = row_name;
      model.rows.emplace(row_name, RowInfo{'N', -1});
    } else {
      model.rows.emplace(row_name, RowInfo{'N', -1});
    }
    return;
  }
  if (type != 'E' && type != 'G' && type != 'L') {
    throw std::invalid_argument("unsupported MPS row type: " + std::string(1, type));
  }
  const int index = static_cast<int>(model.row_types.size());
  model.rows.emplace(row_name, RowInfo{type, index});
  model.row_types.push_back(type);
  if (type == 'E') {
    model.row_lower.push_back(0.0);
    model.row_upper.push_back(0.0);
  } else if (type == 'G') {
    model.row_lower.push_back(0.0);
    model.row_upper.push_back(std::numeric_limits<double>::infinity());
  } else {
    model.row_lower.push_back(-std::numeric_limits<double>::infinity());
    model.row_upper.push_back(0.0);
  }
}

void read_columns_line(RawModel& model, const Card& card, bool integer_section) {
  if (card.fields.size() < 3) {
    throw std::invalid_argument("invalid COLUMNS entry at line " + std::to_string(card.line));
  }
  const int col = ensure_col(model, card.fields[0], integer_section);
  for (std::size_t k = 1; k + 1 < card.fields.size(); k += 2) {
    const std::string& row_name = card.fields[k];
    const double value = parse_double(card.fields[k + 1]);
    const auto found_row = model.rows.find(row_name);
    if (found_row == model.rows.end()) {
      throw std::invalid_argument("unknown MPS row in COLUMNS: " + row_name);
    }
    if (found_row->second.index < 0) {
      if (row_name == model.objective_row) {
        model.objective[static_cast<std::size_t>(col)] += value;
      }
      continue;
    }
    model.col_entries[static_cast<std::size_t>(col)].push_back({found_row->second.index, value});
  }
}

void read_rhs_line(RawModel& model, const Card& card, std::string& rhs_name) {
  if (card.fields.size() < 3) {
    throw std::invalid_argument("invalid RHS entry at line " + std::to_string(card.line));
  }
  if (rhs_name.empty()) {
    rhs_name = card.fields[0];
  }
  if (card.fields[0] != rhs_name) {
    return;
  }

  for (std::size_t k = 1; k + 1 < card.fields.size(); k += 2) {
    const auto found_row = model.rows.find(card.fields[k]);
    if (found_row == model.rows.end()) {
      throw std::invalid_argument("unknown MPS row in RHS: " + card.fields[k]);
    }
    if (found_row->second.index < 0) {
      if (card.fields[k] == model.objective_row) {
        model.obj_constant = -parse_double(card.fields[k + 1]);
      }
      continue;
    }

    const int row = found_row->second.index;
    const double value = parse_double(card.fields[k + 1]);
    if (found_row->second.type == 'E') {
      model.row_lower[static_cast<std::size_t>(row)] = value;
      model.row_upper[static_cast<std::size_t>(row)] = value;
    } else if (found_row->second.type == 'G') {
      model.row_lower[static_cast<std::size_t>(row)] = value;
    } else {
      model.row_upper[static_cast<std::size_t>(row)] = value;
    }
  }
}

void apply_range(RawModel& model, const std::string& row_name, double value) {
  const auto found_row = model.rows.find(row_name);
  if (found_row == model.rows.end()) {
    throw std::invalid_argument("unknown MPS row in RANGES: " + row_name);
  }
  if (found_row->second.index < 0) {
    return;
  }

  const int row = found_row->second.index;
  const std::size_t idx = static_cast<std::size_t>(row);
  if (found_row->second.type == 'E') {
    if (value >= 0.0) {
      model.row_upper[idx] += value;
    } else {
      model.row_lower[idx] += value;
    }
  } else if (found_row->second.type == 'L') {
    model.row_lower[idx] = model.row_upper[idx] - std::fabs(value);
  } else if (found_row->second.type == 'G') {
    model.row_upper[idx] = model.row_lower[idx] + std::fabs(value);
  }
}

void read_ranges_line(RawModel& model, const Card& card, std::string& ranges_name) {
  if (card.fields.size() < 3) {
    throw std::invalid_argument("invalid RANGES entry at line " + std::to_string(card.line));
  }
  if (ranges_name.empty()) {
    ranges_name = card.fields[0];
  }
  if (card.fields[0] != ranges_name) {
    return;
  }

  for (std::size_t k = 1; k + 1 < card.fields.size(); k += 2) {
    apply_range(model, card.fields[k], parse_double(card.fields[k + 1]));
  }
}

void read_bounds_line(RawModel& model, const Card& card, std::string& bounds_name) {
  const double inf = std::numeric_limits<double>::infinity();
  if (card.fields.size() < 3) {
    throw std::invalid_argument("invalid BOUNDS entry at line " + std::to_string(card.line));
  }
  if (bounds_name.empty()) {
    bounds_name = card.fields[1];
  }
  if (card.fields[1] != bounds_name) {
    return;
  }

  const std::string& bound_type = card.fields[0];
  const auto found_col = model.cols.find(card.fields[2]);
  if (found_col == model.cols.end()) {
    throw std::invalid_argument("unknown MPS column in BOUNDS: " + card.fields[2]);
  }
  const std::size_t col = static_cast<std::size_t>(found_col->second);
  if (bound_type == "FR") {
    model.col_lower[col] = -inf;
    model.col_upper[col] = inf;
    return;
  }
  if (bound_type == "MI") {
    model.col_lower[col] = -inf;
    return;
  }
  if (bound_type == "PL") {
    model.col_upper[col] = inf;
    return;
  }
  if (bound_type == "BV") {
    model.var_types[col] = VarType::kBinary;
    model.col_lower[col] = 0.0;
    model.col_upper[col] = 1.0;
    return;
  }

  if (card.fields.size() < 4) {
    throw std::invalid_argument("missing value for MPS bound type: " + bound_type);
  }
  const double value = parse_double(card.fields[3]);
  if (bound_type == "LO") {
    model.col_lower[col] = value;
  } else if (bound_type == "UP") {
    model.col_upper[col] = value;
  } else if (bound_type == "FX") {
    model.col_lower[col] = value;
    model.col_upper[col] = value;
  } else if (bound_type == "LI") {
    model.var_types[col] = VarType::kInteger;
    model.col_lower[col] = value;
  } else if (bound_type == "UI") {
    model.var_types[col] = VarType::kInteger;
    model.col_upper[col] = value;
  } else {
    throw std::invalid_argument("unsupported MPS bound type: " + bound_type);
  }
}

void finalize_variable_bounds(RawModel& model) {
  const double inf = std::numeric_limits<double>::infinity();
  for (std::size_t col = 0; col < model.col_lower.size(); ++col) {
    const bool lower_missing = std::isnan(model.col_lower[col]);
    const bool upper_missing = std::isnan(model.col_upper[col]);
    const VarType type = model.var_types[col];
    if (lower_missing && upper_missing) {
      model.col_lower[col] = 0.0;
      model.col_upper[col] = type == VarType::kMarkedInteger ? 1.0 : inf;
    } else if (lower_missing) {
      model.col_lower[col] = model.col_upper[col] < 0.0 ? -inf : 0.0;
    } else if (upper_missing) {
      model.col_upper[col] = inf;
    }
    if (type == VarType::kMarkedInteger) {
      model.var_types[col] = VarType::kInteger;
    }
  }
}

RawModel parse_lines(const std::vector<std::string>& lines, Format format) {
  RawModel model;
  Section section = Section::kNone;
  bool integer_section = false;
  std::string rhs_name;
  std::string bounds_name;
  std::string ranges_name;
  std::string column_name;

  for (std::size_t i = 0; i < lines.size(); ++i) {
    Card card = format == Format::kFree
                    ? read_card_free(lines[i], static_cast<int>(i + 1))
                    : read_card_fixed(lines[i], static_cast<int>(i + 1));
    if (card.is_comment || card.fields.empty()) {
      continue;
    }

    if (card.is_header) {
      const std::string& header = card.fields[0];
      if (header == "NAME") {
        if (card.fields.size() >= 2) {
          model.name = card.fields[1];
        }
        continue;
      }
      if (header == "ENDATA") {
        break;
      }
      if (!is_section_header(header)) {
        throw std::invalid_argument("unknown MPS section header: " + header);
      }
      section = header_to_section(header);
      if (section == Section::kUnsupportedQuadratic) {
        throw std::invalid_argument("quadratic MPS sections are not supported by cpu_presolve reader");
      }
      continue;
    }

    // In fixed-format MPS, a blank name field continues the previous column,
    // RHS, range, or bound vector. Preserve that information before dispatching
    // to the section-specific readers, which consume normalized field lists.
    if (format == Format::kFixed && card.missing_name_field) {
      if (section == Section::kColumns) {
        if (column_name.empty()) {
          throw std::invalid_argument("COLUMNS continuation without a column name at line " +
                                      std::to_string(card.line));
        }
        card.fields.insert(card.fields.begin(), column_name);
      } else if (section == Section::kRhs) {
        if (rhs_name.empty()) {
          throw std::invalid_argument("RHS continuation without a vector name at line " +
                                      std::to_string(card.line));
        }
        card.fields.insert(card.fields.begin(), rhs_name);
      } else if (section == Section::kRanges) {
        if (ranges_name.empty()) {
          throw std::invalid_argument("RANGES continuation without a vector name at line " +
                                      std::to_string(card.line));
        }
        card.fields.insert(card.fields.begin(), ranges_name);
      } else if (section == Section::kBounds) {
        if (bounds_name.empty()) {
          throw std::invalid_argument("BOUNDS continuation without a vector name at line " +
                                      std::to_string(card.line));
        }
        card.fields.insert(card.fields.begin() + 1, bounds_name);
      }
    }

    switch (section) {
      case Section::kObjSense:
        if (!card.fields.empty() && card.fields[0] == "MAX") {
          throw std::invalid_argument("MPS OBJSENSE MAX is not supported by cpu_presolve reader");
        }
        break;
      case Section::kRows:
        read_rows_line(model, card);
        break;
      case Section::kColumns:
        if (format == Format::kFixed && !card.missing_name_field && !card.fields.empty()) {
          column_name = card.fields[0];
        }
        if (card.fields.size() >= 3 && card.fields[1] == "'MARKER'") {
          if (card.fields[2] == "'INTORG'") {
            integer_section = true;
          } else if (card.fields[2] == "'INTEND'") {
            integer_section = false;
          }
          break;
        }
        read_columns_line(model, card, integer_section);
        break;
      case Section::kRhs:
        read_rhs_line(model, card, rhs_name);
        break;
      case Section::kBounds:
        read_bounds_line(model, card, bounds_name);
        break;
      case Section::kRanges:
        read_ranges_line(model, card, ranges_name);
        break;
      case Section::kNone:
      case Section::kUnsupportedQuadratic:
        throw std::invalid_argument("MPS data entry outside a supported section");
    }
  }

  if (model.row_types.empty() && model.cols.empty()) {
    throw std::invalid_argument("empty MPS model");
  }
  finalize_variable_bounds(model);
  return model;
}

MpsModel build_model(RawModel model) {
  std::vector<int> col_ptr;
  std::vector<int> row_idx;
  std::vector<double> values;
  col_ptr.reserve(model.col_entries.size() + 1);
  col_ptr.push_back(0);

  for (const auto& entries : model.col_entries) {
    std::map<int, double> merged;
    for (const auto& entry : entries) {
      merged[entry.first] += entry.second;
    }
    for (const auto& entry : merged) {
      row_idx.push_back(entry.first);
      values.push_back(entry.second);
    }
    col_ptr.push_back(static_cast<int>(values.size()));
  }

  LpModel lp(
      CscMatrix(
          static_cast<int>(model.row_types.size()),
          static_cast<int>(model.col_entries.size()),
          std::move(col_ptr),
          std::move(row_idx),
          std::move(values)),
      std::move(model.objective),
      std::move(model.row_lower),
      std::move(model.row_upper),
      std::move(model.col_lower),
      std::move(model.col_upper),
      model.obj_constant);

  return MpsModel{std::move(model.name), std::move(lp)};
}

}  // namespace

MpsModel read_mps(std::istream& input) {
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(input, line)) {
    lines.push_back(line);
  }

  try {
    return build_model(parse_lines(lines, Format::kFree));
  } catch (const std::exception& free_error) {
    try {
      return build_model(parse_lines(lines, Format::kFixed));
    } catch (const std::exception&) {
      throw std::invalid_argument(free_error.what());
    }
  }
}

}  // namespace cpu_presolve
