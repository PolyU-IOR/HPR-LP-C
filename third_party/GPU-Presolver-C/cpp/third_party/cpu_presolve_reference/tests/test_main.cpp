#include "test_common.hpp"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace cpu_presolve_tests {
namespace {

std::vector<std::pair<std::string, TestFn>>& registry() {
  static std::vector<std::pair<std::string, TestFn>> tests;
  return tests;
}

}  // namespace

void register_test(const char* name, TestFn fn) {
  registry().push_back({name, fn});
}

int run_all_tests() {
  int failed = 0;
  for (const auto& entry : registry()) {
    try {
      entry.second();
      std::cout << "[PASS] " << entry.first << '\n';
    } catch (const std::exception& err) {
      ++failed;
      std::cerr << "[FAIL] " << entry.first << ": " << err.what() << '\n';
    }
  }
  std::cout << registry().size() << " tests, " << failed << " failures\n";
  return failed == 0 ? 0 : 1;
}

}  // namespace cpu_presolve_tests

int main() {
  return cpu_presolve_tests::run_all_tests();
}

