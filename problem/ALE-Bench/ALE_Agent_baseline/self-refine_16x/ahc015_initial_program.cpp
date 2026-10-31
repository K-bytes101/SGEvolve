#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstdint>

using namespace std;
using namespace std::chrono;

/**
 * Xorshift algorithm for fast pseudo-random number generation.
 * Essential for getting more Monte Carlo trials within the time limit.
 */
struct Xorshift {
    uint32_t x;
    Xorshift(uint32_t seed = 123456789) : x(seed) {}
    uint32_t next() {
        x ^= (x << 13);
        x ^= (x >> 17);
        x ^= (x << 5);
        return x;
    }
};

/**
 * Represents the 10x10 candy box grid.
 * We use a 1D array of 100 elements for better cache performance.
 */
struct Board {
    int8_t grid[100]; // 0: empty, 1-3: flavors

    Board() {
        for (int i = 0; i < 100; ++i) grid[i] = 0;
    }

    /**
     * Tilts the box in the specified direction.
     * 0: Forward (Up), 1: Backward (Down), 2: Left, 3: Right.
     */
    bool tilt(int dir) {
        bool changed = false;
        if (dir == 0) { // Up
            for (int j = 0; j < 10; ++j) {
                int last = 0;
                for (int i = 0; i < 10; ++i) {
                    int idx = i * 10 + j;
                    if (grid[idx]) {
                        if (i != last) {
                            grid[last * 10 + j] = grid[idx];
                            grid[idx] = 0;
                            changed = true;
                        }
                        last++;
                    }
                }
            }
        } else if (dir == 1) { // Down
            for (int j = 0; j < 10; ++j) {
                int last = 9;
                for (int i = 9; i >= 0; --i) {
                    int idx = i * 10 + j;
                    if (grid[idx]) {
                        if (i != last) {
                            grid[last * 10 + j] = grid[idx];
                            grid[idx] = 0;
                            changed = true;
                        }
                        last--;
                    }
                }
            }
        } else if (dir == 2) { // Left
            for (int i = 0; i < 10; ++i) {
                int last = 0;
                for (int j = 0; j < 10; ++j) {
                    int idx = i * 10 + j;
                    if (grid[idx]) {
                        if (j != last) {
                            grid[i * 10 + last] = grid[idx];
                            grid[idx] = 0;
                            changed = true;
                        }
                        last++;
                    }
                }
            }
        } else if (dir == 3) { // Right
            for (int i = 0; i < 10; ++i) {
                int last = 9;
                for (int j = 9; j >= 0; --j) {
                    int idx = i * 10 + j;
                    if (grid[idx]) {
                        if (j != last) {
                            grid[i * 10 + last] = grid[idx];
                            grid[idx] = 0;
                            changed = true;
                        }
                        last--;
                    }
                }
            }
        }
        return changed;
    }

    /**
     * Places the actual candy received into the p-th available empty cell.
     */
    void place_actual(int p, int flavor) {
        int count = 0;
        for (int i = 0; i < 100; ++i) {
            if (grid[i] == 0) {
                if (++count == p) {
                    grid[i] = (int8_t)flavor;
                    return;
                }
            }
        }
    }

    /**
     * Randomly places a candy of the given flavor into an empty cell.
     */
    void place_randomly(int flavor, Xorshift& rng) {
        int empty_indices[100], num_empty = 0;
        for (int i = 0; i < 100; ++i) if (grid[i] == 0) empty_indices[num_empty++] = i;
        if (num_empty > 0) {
            grid[empty_indices[rng.next() % num_empty]] = (int8_t)flavor;
        }
    }

    /**
     * Calculates the sum of squares of connected component sizes.
     * Note: This function modifies the grid to mark visited nodes.
     */
    int eval_score_and_destroy() {
        int sum_sq = 0;
        int q[100];
        for (int i = 0; i < 100; ++i) {
            if (grid[i] <= 0) continue;
            int head = 0, tail = 0, flavor = grid[i];
            q[tail++] = i;
            grid[i] = -flavor; // Mark as visited
            while (head < tail) {
                int cur = q[head++];
                if (cur >= 10 && grid[cur - 10] == flavor) { grid[cur - 10] = -flavor; q[tail++] = cur - 10; }
                if (cur < 90 && grid[cur + 10] == flavor) { grid[cur + 10] = -flavor; q[tail++] = cur + 10; }
                if (cur % 10 > 0 && grid[cur - 1] == flavor) { grid[cur - 1] = -flavor; q[tail++] = cur - 1; }
                if (cur % 10 < 9 && grid[cur + 1] == flavor) { grid[cur + 1] = -flavor; q[tail++] = cur + 1; }
            }
            sum_sq += tail * tail;
        }
        return sum_sq;
    }
};

// 4 different corner-clustering strategies for rollouts
const int policies[4][4] = {
    {0, 2, 3, 1}, // Up, Left priority (Top-Left corner)
    {0, 3, 2, 1}, // Up, Right priority (Top-Right corner)
    {1, 2, 3, 0}, // Down, Left priority (Bottom-Left corner)
    {1, 3, 2, 0}  // Down, Right priority (Bottom-Right corner)
};

inline void apply_policy(Board& b, int p_idx) {
    const int* p = policies[p_idx];
    if (b.tilt(p[0])) return;
    if (b.tilt(p[1])) return;
    if (b.tilt(p[2])) return;
    b.tilt(p[3]);
}

int main() {
    ios_base::sync_with_stdio(false); cin.tie(NULL);

    vector<int> flavors(100);
    for (int i = 0; i < 100; ++i) cin >> flavors[i];

    Xorshift rng;
    Board current_board;
    char d_chars[] = {'F', 'B', 'L', 'R'};
    auto start_prog = steady_clock::now();

    for (int t = 0; t < 100; ++t) {
        int p; cin >> p;
        current_board.place_actual(p, flavors[t]);

        auto turn_start = steady_clock::now();
        double elapsed = duration<double>(turn_start - start_prog).count();
        double turn_time = (1.95 - elapsed) / (100.0 - t);
        if (turn_time < 0.002) turn_time = 0.002;

        double scores[4] = {0};
        int trials[4] = {0};
        int total_trials = 0;
        int d_idx = rng.next() % 4;

        while (true) {
            if (total_trials % 32 == 0) {
                if (duration<double>(steady_clock::now() - turn_start).count() > turn_time) break;
            }

            int d = d_idx % 4;
            Board sim = current_board;
            sim.tilt(d);

            int p_idx = rng.next() % 4; // Use one of the 4 corner policies
            for (int k = t + 1; k < 100; ++k) {
                sim.place_randomly(flavors[k], rng);
                apply_policy(sim, p_idx);
            }

            scores[d] += sim.eval_score_and_destroy();
            trials[d]++;
            total_trials++;
            d_idx++;
        }

        int best_d = 0;
        double max_avg = -1.0;
        for (int d = 0; d < 4; ++d) {
            if (trials[d] > 0) {
                double avg = scores[d] / trials[d];
                if (avg > max_avg) {
                    max_avg = avg;
                    best_d = d;
                }
            }
        }

        cout << d_chars[best_d] << endl;
        current_board.tilt(best_d);
    }

    return 0;
}
