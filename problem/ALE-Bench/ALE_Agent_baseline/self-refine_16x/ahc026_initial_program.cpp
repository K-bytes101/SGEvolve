#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#include <random>
#include <chrono>

using namespace std;

/**
 * Problem Summary:
 * We have n=200 cardboard boxes in m=10 vertical stacks. We must carry them out in 
 * order 1 to n. Moving a block of k boxes from one stack to another costs k+1 energy.
 * Carrying out the current smallest box from the top of its stack costs 0 energy.
 * Goal: Minimize the total energy expended within 5000 operations and 2.0s time limit.
 *
 * Strategy:
 * 1. Greedy Simulation: Simulates the entire carry-out process for boxes 1..n.
 * 2. Sub-block Splitting: To access the target box v, the boxes stacked on top of it 
 *    are moved in descending "well-ordered" sub-blocks. This creates sequences that 
 *    can potentially be carried out later without being moved again.
 * 3. Detailed Heuristic Scoring: Evaluates all destination stacks for a sub-block based on
 *    tightness of fit, future retrieval cost (burial penalty), urgency of buried 
 *    boxes (lookahead), and stack balance.
 * 4. Optimization: Uses a two-phase optimization: 
 *    - Phase 1: Random parameter sampling to find several good regions in the parameter space.
 *    - Phase 2: Hill Climbing on the best parameters found to refine the result.
 */

// Move represents an Operation 1 (v, i > 0) or an Operation 2 (v, 0).
struct Move {
    int v, i;
};

// Simulation state stores current stack configurations and tracks energy consumption.
struct State {
    int n, m;
    vector<vector<int>> stacks;
    vector<int> box_to_stack;
    vector<Move> moves;
    long long energy;

    State(int n, int m, const vector<vector<int>>& initial_stacks) 
        : n(n), m(m), stacks(initial_stacks), energy(0) {
        box_to_stack.assign(n + 1, -1);
        for (int i = 0; i < m; ++i) {
            for (int val : stacks[i]) {
                box_to_stack[val] = i;
            }
        }
    }

    // Executes Operation 1: moves box v and everything stacked above it to another stack.
    void move_block(int v, int dest_idx) {
        int src_idx = box_to_stack[v];
        if (src_idx == dest_idx) return;
        auto& src = stacks[src_idx];
        auto& dst = stacks[dest_idx];
        int pos = -1;
        for (int i = (int)src.size() - 1; i >= 0; --i) {
            if (src[i] == v) { pos = i; break; }
        }
        if (pos == -1) return; // Should not happen
        int k = (int)src.size() - pos;
        energy += (k + 1);
        moves.push_back({v, dest_idx + 1});
        for (int i = pos; i < (int)src.size(); ++i) {
            box_to_stack[src[i]] = dest_idx;
            dst.push_back(src[i]);
        }
        src.resize(pos);
    }

    // Executes Operation 2: carries out box v from the top of its stack.
    void carry_out(int v) {
        int s_idx = box_to_stack[v];
        if (s_idx == -1) return; // Already carried out
        stacks[s_idx].pop_back();
        moves.push_back({v, 0});
    }
};

// Hyper-parameters for the heuristic scoring function.
struct Params {
    int limit, wf, ws, wu, la, tdp, tdn, pb, wes;
};

// Evaluates the heuristic fitness of a destination stack for a moving block S.
inline long long calculate_score(int n, const vector<int>& stack_d, const vector<int>& S, int v, const Params& p) {
    int bottom_v = S[0]; 
    int S_size = (int)S.size();
    long long score = 0;

    if (stack_d.empty()) {
        score = (long long)(n - bottom_v) * p.tdp + p.wes;
    } else {
        int top_v = stack_d.back();
        if (top_v >= bottom_v) {
            score = (long long)(top_v - bottom_v) * p.tdp;
        } else {
            score = (long long)p.pb + (long long)(bottom_v - top_v) * p.tdn;
        }

        int height = (int)stack_d.size();
        for (int j = 0; j < height; ++j) {
            int x = stack_d[j];
            if (x < bottom_v) {
                int dist = x - v;
                if (dist <= 0) continue;
                int boxes_above_x = (height - 1 - j) + S_size;
                long long cost_estimate = (long long)(boxes_above_x + 1);
                if (dist <= p.la) {
                    score += cost_estimate * p.wu * (p.la - dist + 1);
                } else {
                    score += cost_estimate * p.wf;
                }
            }
        }
    }
    score += (long long)stack_d.size() * p.ws;
    if (stack_d.size() > 25) {
        score += (long long)(stack_d.size() - 25) * (stack_d.size() - 25) * p.ws;
    }
    return score;
}

// Runs a full simulation of carrying out all boxes for a given set of heuristic parameters.
long long simulate(int n, int m, const vector<vector<int>>& initial_stacks, const Params& p, vector<Move>& out_moves) {
    State state(n, m, initial_stacks);
    for (int v = 1; v <= n; ++v) {
        int s_idx = state.box_to_stack[v];
        if (s_idx == -1) continue;
        if (state.stacks[s_idx].back() != v) {
            auto& s = state.stacks[s_idx];
            int pos = -1;
            for (int i = (int)s.size() - 1; i >= 0; --i) {
                if (s[i] == v) { pos = i; break; }
            }
            if (pos == -1) continue;

            vector<vector<int>> sub_blocks;
            vector<int> current_S = {s[pos + 1]};
            for (int i = pos + 2; i < (int)s.size(); ++i) {
                if (s[i-1] < s[i] || s[i-1] - s[i] > p.limit) {
                    sub_blocks.push_back(current_S);
                    current_S = {s[i]};
                } else {
                    current_S.push_back(s[i]);
                }
            }
            sub_blocks.push_back(current_S);

            for (int i = (int)sub_blocks.size() - 1; i >= 0; --i) {
                const auto& S = sub_blocks[i];
                int best_d = -1;
                long long min_score = LLONG_MAX;
                for (int d = 0; d < m; ++d) {
                    if (d == s_idx) continue;
                    long long score = calculate_score(n, state.stacks[d], S, v, p);
                    if (score < min_score) {
                        min_score = score;
                        best_d = d;
                    }
                }
                state.move_block(S[0], best_d);
            }
        }
        state.carry_out(v);
        if (state.moves.size() > 5000) return LLONG_MAX;
    }
    out_moves = state.moves;
    return state.energy;
}

struct Candidate {
    long long energy;
    Params p;
    bool operator<(const Candidate& other) const { return energy < other.energy; }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<vector<int>> initial_stacks(m);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n / m; ++j) {
            int b; cin >> b;
            initial_stacks[i].push_back(b);
        }
    }

    long long global_min_energy = LLONG_MAX;
    vector<Move> global_best_moves;
    mt19937 rng(424242);
    auto start_time = chrono::steady_clock::now();

    auto get_rand_params = [&]() {
        return Params{
            uniform_int_distribution<int>(1, 150)(rng),
            uniform_int_distribution<int>(1, 100)(rng),
            uniform_int_distribution<int>(1, 100)(rng),
            uniform_int_distribution<int>(100, 5000)(rng),
            uniform_int_distribution<int>(5, 80)(rng),
            uniform_int_distribution<int>(0, 100)(rng),
            uniform_int_distribution<int>(0, 100)(rng),
            uniform_int_distribution<int>(0, 20000)(rng),
            uniform_int_distribution<int>(-5000, 5000)(rng)
        };
    };

    vector<Candidate> top_candidates;
    int phase1_count = 0;
    while (true) {
        if ((phase1_count & 63) == 0) {
            auto cur_time = chrono::steady_clock::now();
            if (chrono::duration_cast<chrono::milliseconds>(cur_time - start_time).count() > 700) break;
        }
        phase1_count++;
        Params p = get_rand_params();
        vector<Move> cur_moves;
        long long energy = simulate(n, m, initial_stacks, p, cur_moves);
        if (energy != LLONG_MAX) {
            top_candidates.push_back({energy, p});
            if (energy < global_min_energy) {
                global_min_energy = energy;
                global_best_moves = cur_moves;
            }
        }
    }

    sort(top_candidates.begin(), top_candidates.end());
    if (top_candidates.size() > 15) top_candidates.resize(15);

    int phase2_count = 0;
    uniform_int_distribution<int> dist_delta(-10, 10);
    while (true) {
        if ((phase2_count & 31) == 0) {
            auto cur_time = chrono::steady_clock::now();
            if (chrono::duration_cast<chrono::milliseconds>(cur_time - start_time).count() > 1900) break;
        }
        phase2_count++;
        int idx = phase2_count % top_candidates.size();
        Params p = top_candidates[idx].p;
        int choice = uniform_int_distribution<int>(0, 8)(rng);
        int delta = dist_delta(rng);
        if (choice == 0) p.limit = max(1, p.limit + delta);
        else if (choice == 1) p.wf = max(1, p.wf + delta);
        else if (choice == 2) p.ws = max(1, p.ws + delta);
        else if (choice == 3) p.wu = max(1, p.wu + delta * 50);
        else if (choice == 4) p.la = max(1, p.la + delta);
        else if (choice == 5) p.tdp = max(0, p.tdp + delta);
        else if (choice == 6) p.tdn = max(0, p.tdn + delta);
        else if (choice == 7) p.pb = max(0, p.pb + delta * 200);
        else if (choice == 8) p.wes = p.wes + delta * 100;

        vector<Move> cur_moves;
        long long energy = simulate(n, m, initial_stacks, p, cur_moves);
        if (energy != LLONG_MAX && energy < top_candidates[idx].energy) {
            top_candidates[idx].energy = energy;
            top_candidates[idx].p = p;
            if (energy < global_min_energy) {
                global_min_energy = energy;
                global_best_moves = cur_moves;
            }
        }
    }

    for (const auto& move : global_best_moves) {
        cout << move.v << " " << move.i << "\n";
    }

    return 0;
}
