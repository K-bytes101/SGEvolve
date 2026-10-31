#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>

using namespace std;

/**
 * Heuristic approach for Weight Partitioning with unknown weights:
 * 
 * 1. Initial Sorting: Uses a MergeSort-based approach to establish a relative order 
 *    of item weights using a portion of the query budget.
 * 2. Initial Weight Estimation: Initializes weights based on the expected values 
 *    of order statistics for an exponential distribution (harmonic mean sequence).
 * 3. Feedback Loop: 
 *    a. Generates a balanced partition using current estimates via local Hill Climbing.
 *    b. Compares two sets that are estimated to have close total weights.
 *    c. Refines weight estimates using an iterative relaxation method that satisfies 
 *       set-sum constraints multiplicatively.
 *    d. Applies Isotonic Regression (Pool Adjacent Violators Algorithm) to ensure 
 *       estimates remain consistent with the item-level sorting.
 * 4. Final Optimization: Performs a large-scale Hill Climbing search using the 
 *    final refined weight estimates to produce the output partition.
 */

int N, D, Q;
int q_count = 0;

struct Constraint {
    vector<int> L, R;
    int res; // 1: L > R, -1: L < R, 0: L == R
};

vector<Constraint> all_constraints;
vector<int> sorted_items;

// Function to perform a balance query
char query(const vector<int>& L, const vector<int>& R) {
    if (q_count >= Q) return '=';
    q_count++;
    cout << L.size() << " " << R.size();
    for (int x : L) cout << " " << x;
    for (int x : R) cout << " " << x;
    cout << endl;
    char res;
    if (!(cin >> res)) exit(0);
    return res;
}

// Pool Adjacent Violators Algorithm (Isotonic Regression)
// Ensures estimated weights w[sorted_items[i]] follow the monotonic order.
void isotonic_regression(vector<double>& w) {
    struct Block { double sum; int count; };
    vector<Block> blocks;
    blocks.reserve(N);
    for (int i = 0; i < N; ++i) {
        blocks.push_back({w[sorted_items[i]], 1});
        while (blocks.size() >= 2) {
            auto& b_last = blocks.back();
            auto& b_prev = blocks[blocks.size() - 2];
            if (b_prev.sum / b_prev.count > b_last.sum / b_last.count) {
                b_prev.sum += b_last.sum;
                b_prev.count += b_last.count;
                blocks.pop_back();
            } else break;
        }
    }
    int idx = 0;
    for (const auto& b : blocks) {
        double avg = b.sum / b.count;
        for (int i = 0; i < b.count; ++i) w[sorted_items[idx++]] = avg;
    }
}

// Relaxation-based weight adjustment loop
void adjust_weights(vector<double>& w, int iters, chrono::steady_clock::time_point deadline) {
    for (int it = 0; it < iters; ++it) {
        if ((it & 7) == 0 && chrono::steady_clock::now() > deadline) break;
        bool changed = false;
        for (const auto& c : all_constraints) {
            double sL = 0, sR = 0;
            for (int x : c.L) sL += w[x];
            for (int x : c.R) sR += w[x];

            // Adjust weights multiplicatively if the current estimates contradict balance feedback
            if (c.res == 1 && sL <= sR) {
                double m = sqrt((sR + 1.0) / (sL + 1.0)) * 1.01;
                for (int x : c.L) w[x] *= m;
                for (int x : c.R) w[x] /= m;
                changed = true;
            } else if (c.res == -1 && sL >= sR) {
                double m = sqrt((sL + 1.0) / (sR + 1.0)) * 1.01;
                for (int x : c.R) w[x] *= m;
                for (int x : c.L) w[x] /= m;
                changed = true;
            }
        }
        if (!changed && it > 5) break;
        isotonic_regression(w);
    }
    // Normalize weights to keep mean stable
    double sum_w = accumulate(w.begin(), w.end(), 0.0);
    double scale = (100000.0 * N) / (sum_w + 1e-9);
    for (double& val : w) val *= scale;
    isotonic_regression(w);
}

// Hill Climbing algorithm for multi-way number partitioning
void optimize_partition(vector<vector<int>>& sets, vector<double>& set_sums, const vector<double>& w, int iters, chrono::steady_clock::time_point deadline) {
    double total_w = accumulate(w.begin(), w.end(), 0.0);
    double target = total_w / D;
    mt19937 rng(42);

    for (int i = 0; i < iters; ++i) {
        if ((i & 1023) == 0 && chrono::steady_clock::now() > deadline) break;
        int a = rng() % D;
        int b = (a + 1 + rng() % (D - 1)) % D;

        double current_error = pow(set_sums[a] - target, 2) + pow(set_sums[b] - target, 2);

        // Greedy local move
        if (!sets[a].empty()) {
            int ia = rng() % sets[a].size();
            int item = sets[a][ia];
            double ns_a = set_sums[a] - w[item];
            double ns_b = set_sums[b] + w[item];
            if (pow(ns_a - target, 2) + pow(ns_b - target, 2) < current_error) {
                set_sums[a] = ns_a; set_sums[b] = ns_b;
                sets[b].push_back(item);
                sets[a][ia] = sets[a].back(); sets[a].pop_back();
                current_error = pow(ns_a - target, 2) + pow(ns_b - target, 2);
            }
        }
        // Greedy local swap
        if (!sets[a].empty() && !sets[b].empty()) {
            int ia = rng() % sets[a].size(), ib = rng() % sets[b].size();
            int item_a = sets[a][ia], item_b = sets[b][ib];
            double ns_a = set_sums[a] - w[item_a] + w[item_b], ns_b = set_sums[b] - w[item_b] + w[item_a];
            if (pow(ns_a - target, 2) + pow(ns_b - target, 2) < current_error) {
                set_sums[a] = ns_a; set_sums[b] = ns_b;
                swap(sets[a][ia], sets[b][ib]);
            }
        }
    }
}

// Sorting phase using MergeSort with query limits
void merge_parts(vector<int>& arr, int l, int m, int r, int Q_limit) {
    if (q_count >= Q_limit) return;
    vector<int> L_seg(arr.begin() + l, arr.begin() + m), R_seg(arr.begin() + m, arr.begin() + r);
    int i = 0, j = 0, k = l;
    while (i < (int)L_seg.size() && j < (int)R_seg.size()) {
        if (q_count < Q_limit) {
            char res = query({L_seg[i]}, {R_seg[j]});
            if (res == '<') arr[k++] = L_seg[i++];
            else arr[k++] = R_seg[j++];
        } else arr[k++] = L_seg[i++];
    }
    while (i < (int)L_seg.size()) arr[k++] = L_seg[i++];
    while (j < (int)R_seg.size()) arr[k++] = R_seg[j++];
}

void merge_sort(vector<int>& arr, int l, int r, int Q_limit) {
    if (r - l <= 1 || q_count >= Q_limit) return;
    int m = l + (r - l) / 2;
    merge_sort(arr, l, m, Q_limit);
    merge_sort(arr, m, r, Q_limit);
    merge_parts(arr, l, m, r, Q_limit);
}

int main() {
    auto start_time = chrono::steady_clock::now();
    ios_base::sync_with_stdio(false); cin.tie(NULL);

    if (!(cin >> N >> D >> Q)) return 0;
    sorted_items.resize(N);
    iota(sorted_items.begin(), sorted_items.end(), 0);

    // Initial item-sorting
    int Q_sort_limit = min(Q - D - 10, (int)(N * log2(N) + N));
    merge_sort(sorted_items, 0, N, max(0, Q_sort_limit));

    // Initialize weight estimates based on harmonic sequence
    vector<double> hat_w(N);
    double h_sum = 0;
    for (int i = 0; i < N; ++i) {
        h_sum += 1.0 / (N - i);
        hat_w[sorted_items[i]] = 100000.0 * h_sum;
    }

    auto loop_limit = start_time + chrono::milliseconds(1700);
    mt19937 rng(456);

    // Iterative refinement with set comparisons
    while (q_count < Q) {
        if (chrono::steady_clock::now() > loop_limit) {
            query({sorted_items[0]}, {sorted_items[N-1]});
            continue;
        }

        vector<vector<int>> sets(D);
        vector<double> set_sums(D, 0);
        vector<int> rev_order = sorted_items;
        reverse(rev_order.begin(), rev_order.end());
        for (int x : rev_order) {
            int b = 0;
            for (int j = 1; j < D; j++) if (set_sums[j] < set_sums[b]) b = j;
            sets[b].push_back(x);
            set_sums[b] += hat_w[x];
        }
        optimize_partition(sets, set_sums, hat_w, 1500, loop_limit);

        vector<int> s_idx(D); iota(s_idx.begin(), s_idx.end(), 0);
        sort(s_idx.begin(), s_idx.end(), [&](int i, int j) { return set_sums[i] < set_sums[j]; });

        int p_idx = 0;
        if (rng() % 10 < 8) {
            double min_diff = 1e18;
            for (int i = 0; i < D - 1; i++) {
                if (set_sums[s_idx[i+1]] - set_sums[s_idx[i]] < min_diff) {
                    min_diff = set_sums[s_idx[i+1]] - set_sums[s_idx[i]];
                    p_idx = i;
                }
            }
        } else p_idx = rng() % (D - 1);

        int a = s_idx[p_idx], b = s_idx[p_idx + 1];

        if (sets[a].empty() || sets[b].empty()) {
            query({sorted_items[0]}, {sorted_items[N-1]});
        } else {
            char r_char = query(sets[a], sets[b]);
            int res = (r_char == '>' ? 1 : (r_char == '<' ? -1 : 0));
            all_constraints.push_back({sets[a], sets[b], res});
            if (all_constraints.size() > 500) all_constraints.erase(all_constraints.begin());
            adjust_weights(hat_w, 10, loop_limit);
        }
    }

    // Final partition search
    adjust_weights(hat_w, 500, start_time + chrono::milliseconds(1850));
    vector<vector<int>> final_sets(D);
    vector<double> final_sums(D, 0);
    vector<int> rev_order = sorted_items; reverse(rev_order.begin(), rev_order.end());
    for (int x : rev_order) {
        int b = 0;
        for (int j = 1; j < D; j++) if (final_sums[j] < final_sums[b]) b = j;
        final_sets[b].push_back(x);
        final_sums[b] += hat_w[x];
    }
    optimize_partition(final_sets, final_sums, hat_w, 3000000, start_time + chrono::milliseconds(1980));

    vector<int> ans(N);
    for (int i = 0; i < D; i++) for (int x : final_sets[i]) ans[x] = i;
    for (int i = 0; i < N; i++) cout << ans[i] << (i == N - 1 ? "" : " ");
    cout << endl;

    return 0;
}
