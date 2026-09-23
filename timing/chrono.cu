#include <chrono>

int main(){
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);
    auto duration2 = std::chrono::duration_cast<std::chrono::seconds> (end - start);
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds> (end - start);
}