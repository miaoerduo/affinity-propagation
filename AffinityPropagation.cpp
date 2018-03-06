//
// Created by zhaoyu on 2017/3/23.
//

#include "AffinityPropagation.h"
#include <assert.h>
#include <algorithm>
#include <limits>
#include <set>
#include <unordered_map>
#include <iostream>

namespace AP {

    AffinityPropagation::AffinityPropagation(
            float damping,
            int max_iter,
            int convergence_iter,
            std::function<float(const void *, const void *, const long)> comp
    ) :
            m_damping(damping),
            m_max_iter(max_iter),
            m_convergence_iter(convergence_iter),
            m_comp(comp) {
        assert(m_damping >= 0.5 && m_damping <= 1.);
        assert(m_max_iter >= convergence_iter);
        assert(convergence_iter > 0);
    }

    void AffinityPropagation::fit(const float *feature_arr, const long feature_num, const long feature_len) {
        this->cal_affinity_matrix(feature_arr, feature_num, feature_len);
        this->m_cluster_centers_indices.clear();
        this->m_labels.clear();
        affinity_propagation(
                this->m_cluster_centers_indices,
                this->m_labels,
                this->m_affinity_matrix,
                this->m_convergence_iter,
                this->m_max_iter,
                this->m_damping
                );
    }

    void AffinityPropagation::fit(const std::vector< std::vector<float> > &feature_arr) {
        this->cal_affinity_matrix(feature_arr);
        this->m_cluster_centers_indices.clear();
        this->m_labels.clear();
        affinity_propagation(
                this->m_cluster_centers_indices,
                this->m_labels,
                this->m_affinity_matrix,
                this->m_convergence_iter,
                this->m_max_iter,
                this->m_damping
                );
    }

    void AffinityPropagation::cal_affinity_matrix(const std::vector<std::vector<float>> &feature_arr) {
        const unsigned long feature_num = feature_arr.size();
        assert(feature_num > 0);
        const unsigned long feature_length = feature_arr[0].size();
        assert(feature_length > 0);
        std::vector<std::vector<float> > tmp = std::vector<std::vector<float> >(feature_num,
                                                                                std::vector<float>(feature_num, 0.));

        this->m_affinity_matrix.swap(tmp);

        for (int i = 0; i < feature_num; ++i) {
            for (int j = i + 1; j < feature_num; ++j) {

                this->m_affinity_matrix[i][j]
                        = this->m_affinity_matrix[j][i]
                        = this->m_comp(
                                reinterpret_cast<const void *>(&feature_arr[i][0]),
                                reinterpret_cast<const void *>(&feature_arr[j][0]),
                                feature_length * sizeof(float));
            }
        }
    }

    void AffinityPropagation::cal_affinity_matrix(const float *feature_arr,
                                                  const long feature_num,
                                                  const long feature_len) {

        assert(feature_num > 0);
        assert(feature_len > 0);

        std::vector<std::vector<float> > tmp = std::vector<std::vector<float> >(feature_num,
                                                                                std::vector<float>(feature_num, 0.f));

        this->m_affinity_matrix.swap(tmp);

        for (int i = 0; i < feature_num; ++i) {
            for (int j = i + 1; j < feature_num; ++j) {

                this->m_affinity_matrix[i][j]
                        = this->m_affinity_matrix[j][i]
                        = this->m_comp(
                                reinterpret_cast<const void *>(feature_arr + i * feature_len),
                                reinterpret_cast<const void *>(feature_arr + j * feature_len),
                                feature_len * sizeof(float));
            }
        }
    }

    void AffinityPropagation::affinity_propagation(
            std::vector<int> &cluster_centers_indices, std::vector<int> &labels,
            std::vector<std::vector<float> > &S, int convergence_iter,
            int max_iter, float damping) {

        assert(convergence_iter > 0);
        assert(max_iter > 0);
        assert(damping >= 0.5 && damping <=1.);
        assert(S.size() > 0 && S.size() == S[0].size());

        const float FLOAT_MIN(-1e30f);
        const unsigned long n_samples = S.size();
        const unsigned long total_ = n_samples * n_samples;

        std::vector<float> S_elem_arr;
        for (int i = 0; i < n_samples; ++ i) {
            S_elem_arr.insert(S_elem_arr.end(), S[i].begin(), S[i].end());
        }

        // preference
        // 取S的中值
        float preference(0.f);
        std::nth_element(S_elem_arr.begin(), S_elem_arr.begin() + total_ / 2, S_elem_arr.end());
        preference = S_elem_arr[total_ / 2];
        if (total_ % 2 == 0) {
            std::nth_element(S_elem_arr.begin(), S_elem_arr.begin() + total_ / 2 - 1, S_elem_arr.end());
            preference = S_elem_arr[total_ / 2 - 1] + (preference - S_elem_arr[total_ / 2 - 1]) / 2;
        }

        std::vector<float>().swap(S_elem_arr);

        for (int i = 0; i < S.size(); ++ i) {
            S[i][i] = preference;
        }

        std::vector< std::vector<float> > A(n_samples, std::vector<float>(n_samples, 0.f));
        std::vector< std::vector<float> > R(n_samples, std::vector<float>(n_samples, 0.f));
        std::vector< std::vector<float> > tmp(n_samples, std::vector<float>(n_samples, 0.f));
        std::vector<int> I(n_samples, 0);
        std::vector<float> Y(n_samples, 0.f);
        std::vector<float> Y2(n_samples, 0.f);
        std::vector<float> dA(n_samples, 0.f);
        std::vector< std::vector<int> > e(n_samples, std::vector<int>(static_cast<unsigned int>(convergence_iter), 0));
        std::vector<int> se(n_samples, 0);

        for (int it = 0; it < max_iter; ++ it) {
            // np.add(A, S, tmp)
            for (int i = 0; i < n_samples; ++ i) {
                for (int j = 0; j < n_samples; ++ j) {
                    tmp[i][j] = A[i][j] + S[i][j];
                }
            }

            // I = np.argmax(tmp, axis=1)
            for (int i = 0; i < n_samples; ++ i) {
                int max_idx = 0;
                for (int j = 0; j < n_samples; ++ j) {
                    if (tmp[i][max_idx] < tmp[i][j]) {
                        max_idx = j;
                    }
                }
                I[i] = max_idx;
            }

            // Y = tmp[ind, I]  # np.max(A + S, axis=1)
            for (int i = 0; i < n_samples; ++ i) {
                Y[i] = tmp[i][I[i]];
            }

            // tmp[ind, I] = -np.inf
            for (int i = 0; i < n_samples; ++ i) {
                tmp[i][I[i]] = FLOAT_MIN;
            }

            // Y2 = np.max(tmp, axis=1)
            for (int i = 0; i < n_samples; ++ i) {
                float _max = tmp[i][0];
                for (int j = 0; j < n_samples; ++ j) {
                    if (_max < tmp[i][j]) {
                        _max = tmp[i][j];
                    }
                }
                Y2[i] = _max;
            }

            // # tmp = Rnew
            //np.subtract(S, Y[:, None], tmp)
            //tmp[ind, I] = S[ind, I] - Y2

            for (int i = 0; i < n_samples; ++ i) {
                for (int j = 0; j < n_samples; ++ j) {
                    tmp[i][j] = S[i][j] - Y[i];
                }
            }

            for (int i = 0; i < n_samples; ++ i) {
                tmp[i][I[i]] = S[i][I[i]] - Y2[i];
            }

            // # Damping
            // tmp *= 1 - damping
            // R *= damping
            // R += tmp
            for (int i = 0; i < n_samples; ++ i) {
                for (int j = 0; j < n_samples; ++j) {
                    R[i][j] = (1 - damping) * tmp[i][j] + damping * R[i][j];
                }
            }

            // np.maximum(R, 0, tmp)
            // tmp.flat[::n_samples + 1] = R.flat[::n_samples + 1]

            for (int i = 0; i < n_samples; ++ i) {
                for (int j = 0; j < n_samples; ++ j) {
                    tmp[i][j] = std::max(float(0.), R[i][j]);
                }
            }

            for (int i = 0; i < n_samples; ++ i) {
                tmp[i][i] = R[i][i];
            }

            // tmp -= np.sum(tmp, axis=0)

            for (int j = 0; j < n_samples; ++ j) {
                float _sum(0.f);
                for (int i = 0; i < n_samples; ++ i) {
                    _sum += tmp[i][j];
                }
                for (int i = 0; i < n_samples; ++ i) {
                    tmp[i][j] -= _sum;
                }
            }
            // dA = np.diag(tmp).copy()
            for (int i = 0; i < n_samples; ++ i) {
                dA[i] = tmp[i][i];
            }

            // tmp.clip(0, np.inf, tmp)
            for (int i = 0; i < n_samples; ++ i) {
                for (int j = 0; j < n_samples; ++ j) {
                    tmp[i][j] = std::max(float(0.), tmp[i][j]);
                }
            }

            // tmp.flat[::n_samples + 1] = dA
            for (int i = 0; i < n_samples; ++ i) {
                tmp[i][i] = dA[i];
            }

            // # Damping

            for (int i = 0; i < n_samples; ++ i) {
                for (int j = 0; j < n_samples; ++ j) {
                    A[i][j] = - (1 - damping) * tmp[i][j] + damping * A[i][j];
                }
            }

            // # Check for convergence
            // E = (np.diag(A) + np.diag(R)) > 0
            // e[:, it % convergence_iter] = E
            // K = np.sum(E, axis=0)
            int _it = it % convergence_iter;
            int K(0);
            for (int i = 0; i < n_samples; ++ i) {
                e[i][_it] = (A[i][i] + R[i][i]) > 0 ? 1: 0;
                K += e[i][_it];
            }

            if (it >= convergence_iter) {
                // se = np.sum(e, axis=1)
                se.assign(n_samples, 0);
                for (int i = 0; i < n_samples; ++ i) {
                    for (int t = 0; t < convergence_iter; ++ t) {
                        se[i] += e[i][t];
                    }
                }
                // unconverged = (np.sum((se == convergence_iter) + (se == 0))
                //               != n_samples)

                int _c(0);
                for (int i = 0; i < n_samples; ++ i) {
                    _c += (se[i] == convergence_iter || se[i] == 0);
                }

                bool unconverged = _c != n_samples;

                if ( (!unconverged && (K > 0)) || (it == max_iter) ) {
                    break;
                }
            }
        }

        // I = np.where(np.diag(A + R) > 0)[0]
        // K = I.size  # Identify exemplars

        int K(0);
        for (int i = 0; i < n_samples; ++ i) {
            if (A[i][i] + R[i][i] > 0) {
                I[K ++] = i;
            }
        }

        if (K > 0) {
            // c = np.argmax(S[:, I], axis=1)
            std::vector<int> c(n_samples, 0);
            for (int i = 0; i < n_samples; ++ i) {
                int _max_idx = 0;
                for (int k = 0; k < K; ++k) {
                    if (S[i][I[_max_idx]] < S[i][I[k]]) {
                        _max_idx = k;
                    }
                }
                c[i] = _max_idx;
            }

            // c[I] = np.arange(K) # Identify clusters
            for (int k = 0; k < K; ++ k) {
                c[I[k]] = k;
            }

            // # Refine the final set of exemplars and clusters and return results
            // for k in range(K):
            //     ii = np.where(c == k)[0]
            //     j = np.argmax(np.sum(S[ii[:, np.newaxis], ii], axis=0))
            //     I[k] = ii[j]

            for (int k = 0; k < K; ++ k) {
                std::vector<int> ii;
                for (int i = 0; i < n_samples; ++ i) {
                    if (c[i] == k) ii.push_back(i);
                }

                int _j = -1;
                float _j_max = -1e30f;

                for (int j = 0; j < ii.size(); ++ j) {
                    float _sum(0.f);
                    for (int i = 0; i < ii.size(); ++ i) {
                        _sum += S[ii[i]][ii[j]];
                    }
                    if (_sum > _j_max) {
                        _j_max = _sum;
                        _j = j;
                    }
                }

                I[k] = ii[_j];
            }

            // c = np.argmax(S[:, I], axis=1)
            // c[I] = np.arange(K)
            for (int i = 0; i < n_samples; ++ i) {
                int _max_idx = I[0];
                for (int k = 0; k < K; ++ k) {
                    if (S[i][_max_idx] < S[i][I[k]]) {
                        _max_idx = I[k];
                    }
                }
                c[i] = _max_idx;
            }

            // c[I] = np.arange(K) # Identify clusters
            for (int k = 0; k < K; ++ k) {
                c[I[k]] = k;
            }

            for (int i = 0; i < n_samples; ++ i) {
                labels.push_back(I[c[i]]);
            }

            std::set<int> s(labels.begin(), labels.end());

            std::vector<int>(s.begin(), s.end()).swap(cluster_centers_indices);
            std::unordered_map<int, int> c2l;
            for (int i = 0; i < cluster_centers_indices.size(); ++ i) {
                c2l[cluster_centers_indices[i]] = i;
            }
            for (auto &v: labels) {
                v = c2l[v];
            }

        } else {
            labels.empty();
            cluster_centers_indices.empty();
        }
    }
}
