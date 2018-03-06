#include <iostream>
#include "AffinityPropagation.h"
#include <fstream>
#include <vector>

using namespace std;

std::vector<std::vector<float>>
load_feature(
             const char * filename,
             const int feature_num,
             const int feature_len) {
    using std::vector;
    std::ifstream is(filename, std::ifstream::binary);
    if (!is.is_open()) throw "file cannot be opened";
    vector<vector<float>> feature_arr(feature_num, vector<float>(feature_len, 0.));

    for (int fid = 0; fid < feature_num; ++ fid) {
        for (int idx = 0; idx < feature_len; ++ idx) {
            is >> feature_arr[fid][idx];
        }
    }

    return feature_arr;
}

int main() {
    auto feature_arr = load_feature("demo.txt", 6, 5);

    AP::AffinityPropagation ap;
    ap.fit(feature_arr);

    for (auto e: ap.m_labels) {
        std::cout << e << std::endl;
    }
	return 0;
}
