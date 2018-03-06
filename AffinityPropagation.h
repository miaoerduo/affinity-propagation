//
// Created by zhaoyu on 2017/3/23.
//

#ifndef AP_AFFINITYPROPAGATION_H
#define AP_AFFINITYPROPAGATION_H

#include <functional>
#include <vector>

namespace AP {

    class AffinityPropagation {

    public:
        static float l2(const void *f1, const void *f2, const long len) {
            float ret(0.);
            const float *fea1 = static_cast<const float *>(f1);
            const float *fea2 = static_cast<const float *>(f2);
            const long length = len / sizeof(float);
            for (int i = 0; i < length; ++ i) {
                float d = fea1[i] - fea2[i];
                ret += d * d;
            }
            return - ret;
        }

        static void affinity_propagation(
                std::vector<int> &cluster_centers_indices,
                std::vector<int> &labels,
                std::vector< std::vector<float> > &S,
                int convergence_iter = 15,
                int max_iter = 200,
                float damping = 0.5
        );

    public:
        AffinityPropagation(float damping=0.5, int max_iter=200, int convergence_iter=15, std::function<float(const void *, const void *, const long)> comparer=AffinityPropagation::l2);

        void fit(const float *feature_arr, const long feature_num, const long feature_len);
        void fit(const std::vector< std::vector<float> > &feature_arr);

    public:
        void cal_affinity_matrix(const float *feature_arr, const long feature_num, const long feature_len);
        void cal_affinity_matrix(const std::vector<std::vector<float>> &feature_arr);

    public:

        float m_damping;
        int m_max_iter;
        int m_convergence_iter;
        std::function<float(const void *, const void *, const long)> m_comp;
        std::vector<std::vector<float>> m_affinity_matrix;
        std::vector<int> m_cluster_centers_indices;
        std::vector<int> m_labels;
    };

}

#endif //AP_AFFINITYPROPAGATION_H
