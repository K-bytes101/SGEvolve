#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

/**
 * World-class algorithm engineer's solution for the Takahashi Fisher problem.
 * 
 * Strategy:
 * 1. Find the best x-monotone or y-monotone rectilinear polygon.
 * 2. Monotone rectilinear polygons satisfy:
 *    - No self-intersections (by construction).
 *    - Vertex count limit (well under 1000).
 *    - Perimeter constraint (enforced within the DP objective).
 * 3. Use coordinate compression to build an adaptive grid based on fish locations.
 * 4. Use Dynamic Programming with Lagrangian relaxation to maximize:
 *    Objective = (Mackerels - Sardines) * SCALE - Perimeter
 *    Setting SCALE = 10,000,000 ensures mackerels are the primary priority.
 * 5. A penalty is applied to any path exceeding the 400,000 length limit.
 */

const int G_MAX = 86; // Grid resolution
const long long SCALE = 10000000LL;
const long long INF = 1e17;

struct Point {
    int x, y;
};

struct Result {
    int fish_score;
    long long perimeter;
    int i_start, i_end;
    vector<pair<int, int>> path;
    bool swapped;
    vector<int> xs, ys;
};

// Global DP tables to optimize memory usage and speed
static int cell_fish[G_MAX][G_MAX];
static int row_sum[G_MAX][G_MAX + 1];
static int fish_dp[G_MAX][G_MAX][G_MAX];
static int perim_dp[G_MAX][G_MAX][G_MAX];
static pair<short, short> p_table[G_MAX][G_MAX][G_MAX];

/**
 * Solve for an x-monotone rectilinear polygon in the given orientation.
 */
Result solve_monotone(int N, const vector<Point>& macks, const vector<Point>& sards, bool swap_xy, int grid_size, int d_limit) {
    vector<Point> m = macks, s = sards;
    if (swap_xy) {
        for (int i = 0; i < N; ++i) {
            swap(m[i].x, m[i].y);
            swap(s[i].x, s[i].y);
        }
    }

    // 1. Adaptive Coordinate Compression
    vector<int> X = {0, 100000}, Y = {0, 100000};
    for (int i = 0; i < N; ++i) {
        X.push_back(m[i].x); X.push_back(s[i].x);
        Y.push_back(m[i].y); Y.push_back(s[i].y);
    }
    sort(X.begin(), X.end()); X.erase(unique(X.begin(), X.end()), X.end());
    sort(Y.begin(), Y.end()); Y.erase(unique(Y.begin(), Y.end()), Y.end());

    int gx_cnt = min((int)X.size() - 1, grid_size);
    int gy_cnt = min((int)Y.size() - 1, grid_size);
    vector<int> xs, ys;
    for (int i = 0; i <= gx_cnt; ++i) xs.push_back(X[i * (X.size() - 1) / gx_cnt]);
    for (int i = 0; i <= gy_cnt; ++i) ys.push_back(Y[i * (Y.size() - 1) / gy_cnt]);

    // 2. Map fish density to the grid cells
    for (int i = 0; i < gx_cnt; ++i) for (int j = 0; j < gy_cnt; ++j) cell_fish[i][j] = 0;
    for (int i = 0; i < N; ++i) {
        int gx = lower_bound(xs.begin(), xs.end(), m[i].x) - xs.begin(); if (gx > 0 && (gx == gx_cnt || xs[gx] > m[i].x)) gx--;
        int gy = lower_bound(ys.begin(), ys.end(), m[i].y) - ys.begin(); if (gy > 0 && (gy == gy_cnt || ys[gy] > m[i].y)) gy--;
        if (gx < gx_cnt && gy < gy_cnt) cell_fish[gx][gy]++;
    }
    for (int i = 0; i < N; ++i) {
        int gx = lower_bound(xs.begin(), xs.end(), s[i].x) - xs.begin(); if (gx > 0 && (gx == gx_cnt || xs[gx] > s[i].x)) gx--;
        int gy = lower_bound(ys.begin(), ys.end(), s[i].y) - ys.begin(); if (gy > 0 && (gy == gy_cnt || ys[gy] > s[i].y)) gy--;
        if (gx < gx_cnt && gy < gy_cnt) cell_fish[gx][gy]--;
    }

    for (int i = 0; i < gx_cnt; ++i) {
        row_sum[i][0] = 0;
        for (int j = 0; j < gy_cnt; ++j) row_sum[i][j + 1] = row_sum[i][j] + cell_fish[i][j];
    }

    // 3. Main Dynamic Programming Loop
    for (int i = 0; i < gx_cnt; ++i) 
        for (int L = 0; L < gy_cnt; ++L) 
            for (int R = L; R < gy_cnt; ++R) {
                fish_dp[i][L][R] = -1000000;
                perim_dp[i][L][R] = 10000000;
            }

    for (int i = 0; i < gx_cnt; ++i) {
        int dx2 = 2 * (xs[i + 1] - xs[i]);
        for (int L = 0; L < gy_cnt; ++L) {
            for (int R = L; R < gy_cnt; ++R) {
                int cur_f = row_sum[i][R + 1] - row_sum[i][L];
                // Path Initiation: A single-column vertical rectangle
                fish_dp[i][L][R] = cur_f;
                perim_dp[i][L][R] = dx2 + (ys[R + 1] - ys[L]);
                p_table[i][L][R] = {-1, -1};

                // Path Extension: Connecting to an overlapping interval in the previous column
                if (i > 0) {
                    int L1_s = max(0, L - d_limit), L1_e = min({gy_cnt - 1, L + d_limit, R});
                    int R1_s = max({0, R - d_limit, L}), R1_e = min(gy_cnt - 1, R + d_limit);
                    for (int L1 = L1_s; L1 <= L1_e; ++L1) {
                        for (int R1 = max(L1, R1_s); R1 <= R1_e; ++R1) {
                            if (fish_dp[i - 1][L1][R1] < -5000) continue;
                            
                            int jump = abs(ys[L] - ys[L1]) + abs(ys[R + 1] - ys[R1 + 1]);
                            int nf = fish_dp[i - 1][L1][R1] + cur_f;
                            int np = perim_dp[i - 1][L1][R1] + jump + dx2;

                            // Objective: maximize fish, penalize perimeter, avoid exceeding 400,000 length
                            long long old_obj = (long long)fish_dp[i][L][R] * SCALE - perim_dp[i][L][R];
                            if (perim_dp[i][L][R] + (ys[R + 1] - ys[L]) > 400000) old_obj -= 100000000000000LL;
                            
                            long long new_obj = (long long)nf * SCALE - np;
                            if (np + (ys[R + 1] - ys[L]) > 400000) new_obj -= 100000000000000LL;

                            if (new_obj > old_obj) {
                                fish_dp[i][L][R] = nf;
                                perim_dp[i][L][R] = np;
                                p_table[i][L][R] = {(short)L1, (short)R1};
                            }
                        }
                    }
                }
            }
        }
    }

    // 4. Extract the Global Optimal Solution
    long long best_obj = -INF;
    int bi = -1, bl = -1, br = -1;
    for (int i = 0; i < gx_cnt; ++i) {
        for (int L = 0; L < gy_cnt; ++L) {
            for (int R = L; R < gy_cnt; ++R) {
                if (fish_dp[i][L][R] < -5000) continue;
                long long total_p = (long long)perim_dp[i][L][R] + (ys[R + 1] - ys[L]);
                long long obj = (long long)fish_dp[i][L][R] * SCALE - total_p;
                if (total_p > 400000) obj -= 100000000000000LL;
                if (obj > best_obj) {
                    best_obj = obj; bi = i; bl = L; br = R;
                }
            }
        }
    }

    if (bi == -1 || fish_dp[bi][bl][br] <= 0) return {0, 0, -1, -1, {}, swap_xy, xs, ys};

    vector<pair<int, int>> path;
    int ci = bi, cl = bl, cr = br;
    while (ci >= 0 && cl != -1) {
        path.push_back({cl, cr});
        auto p = p_table[ci][cl][cr];
        cl = p.first; cr = p.second; ci--;
    }
    reverse(path.begin(), path.end());
    return {fish_dp[bi][bl][br], (long long)perim_dp[bi][bl][br] + (ys[br + 1] - ys[bl]), bi - (int)path.size() + 1, bi, path, swap_xy, xs, ys};
}

/**
 * Output standard vertex format for a monotone rectilinear polygon.
 */
void output(const Result& res) {
    if (res.fish_score <= 0) {
        cout << "4\n0 0\n0 1\n1 1\n1 0" << endl;
        return;
    }
    vector<pair<int, int>> v;
    int n = res.path.size();
    v.push_back({res.xs[res.i_start], res.ys[res.path[0].first]});
    v.push_back({res.xs[res.i_start], res.ys[res.path[0].second + 1]});
    for (int i = 0; i < n - 1; ++i) {
        v.push_back({res.xs[res.i_start + i + 1], res.ys[res.path[i].second + 1]});
        v.push_back({res.xs[res.i_start + i + 1], res.ys[res.path[i + 1].second + 1]});
    }
    v.push_back({res.xs[res.i_end + 1], res.ys[res.path.back().second + 1]});
    v.push_back({res.xs[res.i_end + 1], res.ys[res.path.back().first]});
    for (int i = n - 1; i > 0; --i) {
        v.push_back({res.xs[res.i_start + i], res.ys[res.path[i].first]});
        v.push_back({res.xs[res.i_start + i], res.ys[res.path[i - 1].first]});
    }
    if (res.swapped) for (auto& p : v) swap(p.first, p.second);
    
    vector<pair<int, int>> uv;
    for (auto p : v) if (uv.empty() || p != uv.back()) uv.push_back(p);
    if (uv.size() > 1 && uv.front() == uv.back()) uv.pop_back();

    cout << uv.size() << endl;
    for (auto p : uv) cout << p.first << " " << p.second << endl;
}

int main() {
    ios_base::sync_with_stdio(false); cin.tie(NULL);
    int N; if (!(cin >> N)) return 0;
    vector<Point> macks(N), sards(N);
    for (int i = 0; i < N; ++i) cin >> macks[i].x >> macks[i].y;
    for (int i = 0; i < N; ++i) cin >> sards[i].x >> sards[i].y;

    Result best = {0, 0, -1, -1, {}, false, {}, {}};
    // Evaluate x-monotone and y-monotone polygons and pick the best one.
    for (bool s_xy : {false, true}) {
        Result cur = solve_monotone(N, macks, sards, s_xy, 85, 10);
        if (cur.fish_score > best.fish_score || (cur.fish_score == best.fish_score && cur.perimeter < best.perimeter)) {
            best = cur;
        }
    }
    output(best);
    return 0;
}