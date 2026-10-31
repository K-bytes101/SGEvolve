#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <chrono>

using namespace std;

// Skating rink constants
const int N = 20;
const int N2 = 400;
const int dr[] = {-1, 1, 0, 0}; // Up, Down, Left, Right
const int dc[] = {0, 0, -1, 1};
const char dchar[] = {'U', 'D', 'L', 'R'};
const int INF = 1e7;

// To store parent information for path reconstruction
struct Parent {
    int v;
    char type, dir;
};

/**
 * Fast Dijkstra using Dial's algorithm (3-bucket queue) for edge weights 1 and 2.
 * Weights: Slide = 1, Move = 1 (empty square), Alter + Move = 2 (blocked square).
 * Calculates the shortest path distance and parents from start_v.
 */
void fast_dijkstra(int start_v, const bool* grid, int* dists, Parent* parents = nullptr) {
    fill(dists, dists + N2, INF);
    if (start_v < 0 || start_v >= N2) return;
    dists[start_v] = 0;

    static int q[3][1024];
    int h[3] = {0, 0, 0}, t[3] = {0, 0, 0};
    q[0][t[0]++] = start_v;

    int d = 0, rem = 1;
    while (rem > 0) {
        int b = d % 3;
        while (h[b] < t[b]) {
            int u = q[b][h[b]++]; rem--;
            if (dists[u] < d) continue;

            int cr = u / N, cc = u % N;
            // Action: Move (cost 1 if empty, 2 if blocked by adding/removing a block)
            for (int i = 0; i < 4; ++i) {
                int nr = cr + dr[i], nc = cc + dc[i];
                if (nr >= 0 && nr < N && nc >= 0 && nc < N) {
                    int v = nr * N + nc;
                    int cost = 1 + (grid[v] ? 1 : 0);
                    if (dists[v] > d + cost) {
                        dists[v] = d + cost;
                        if (parents) parents[v] = {u, 'M', dchar[i]};
                        q[(d + cost) % 3][t[(d + cost) % 3]++] = v; rem++;
                    }
                }
            }
            // Action: Slide (cost 1)
            for (int i = 0; i < 4; ++i) {
                int sr = cr, sc = cc;
                while (true) {
                    int nr = sr + dr[i], nc = sc + dc[i];
                    if (nr < 0 || nr >= N || nc < 0 || nc >= N || grid[nr * N + nc]) break;
                    sr = nr; sc = nc;
                }
                if (sr != cr || sc != cc) {
                    int v = sr * N + sc;
                    if (dists[v] > d + 1) {
                        dists[v] = d + 1;
                        if (parents) parents[v] = {u, 'S', dchar[i]};
                        q[(d + 1) % 3][t[(d + 1) % 3]++] = v; rem++;
                    }
                }
            }
        }
        h[b] = t[b] = 0; d++;
    }
}

/**
 * Backward Dijkstra to handle slide asymmetry. 
 * Calculates turn-costs from all squares towards target_v.
 */
void fast_backward_dijkstra(int target_v, const bool* grid, int* dists) {
    fill(dists, dists + N2, INF);
    if (target_v < 0 || target_v >= N2) return;
    dists[target_v] = 0;

    static int q[3][1024];
    int h[3] = {0, 0, 0}, t[3] = {0, 0, 0};
    q[0][t[0]++] = target_v;

    int d = 0, rem = 1;
    while (rem > 0) {
        int b = d % 3;
        while (h[b] < t[b]) {
            int u = q[b][h[b]++]; rem--;
            if (dists[u] < d) continue;

            int cr = u / N, cc = u % N;
            // Reverse Move
            for (int i = 0; i < 4; ++i) {
                int pr = cr - dr[i], pc = cc - dc[i];
                if (pr >= 0 && pr < N && pc >= 0 && pc < N) {
                    int cost = 1 + (grid[u] ? 1 : 0);
                    int v = pr * N + pc;
                    if (dists[v] > d + cost) {
                        dists[v] = d + cost;
                        q[(d + cost) % 3][t[(d + cost) % 3]++] = v; rem++;
                    }
                }
            }
            // Reverse Slide: Finds all squares that would stop at current (cr, cc) when sliding.
            for (int i = 0; i < 4; ++i) {
                int br = cr + dr[i], bc = cc + dc[i];
                if (br < 0 || br >= N || bc < 0 || bc >= N || grid[br * N + bc]) {
                    int pr = cr - dr[i], pc = cc - dc[i];
                    while (pr >= 0 && pr < N && pc >= 0 && pc < N && !grid[pr * N + pc]) {
                        int v = pr * N + pc;
                        if (dists[v] > d + 1) {
                            dists[v] = d + 1;
                            q[(d + 1) % 3][t[(d + 1) % 3]++] = v; rem++;
                        }
                        pr -= dr[i]; pc -= dc[i];
                    }
                }
            }
        }
        h[b] = t[b] = 0; d++;
    }
}

/**
 * Reconstructs the movement sequence for a path, outputs actions, and updates global rink state.
 */
void reconstruct_path(int start_v, int end_v, Parent* parents, bool* grid, int& cur_v) {
    if (start_v == end_v) return;
    vector<pair<char, char>> path;
    int v = end_v;
    while (v != start_v) {
        Parent p = parents[v];
        path.push_back({p.type, p.dir});
        v = p.v;
    }
    reverse(path.begin(), path.end());
    for (auto& p : path) {
        int di = (p.second == 'U' ? 0 : p.second == 'D' ? 1 : p.second == 'L' ? 2 : 3);
        if (p.first == 'M') {
            int next_v = cur_v + dr[di] * N + dc[di];
            if (grid[next_v]) { cout << "A " << p.second << "\n"; grid[next_v] = false; }
            cout << "M " << p.second << "\n";
            cur_v = next_v;
        } else {
            cout << "S " << p.second << "\n";
            while (true) {
                int nr = cur_v / N + dr[di], nc = cur_v % N + dc[di];
                if (nr < 0 || nr >= N || nc < 0 || nc >= N || grid[nr * N + nc]) break;
                cur_v = nr * N + nc;
            }
        }
    }
}

// Global reusable buffers
bool grid[N2];
int d_cur[N2], bdm_mod[N2], d_temp[N2], d_final[N2], d_baseline[41][N2];
Parent p_initial[N2], p_final[N2];
const int MAX_LOOK_AHEAD = 10;

int main() {
    ios::sync_with_stdio(0); cin.tie(0);
    int n_in, m_in;
    if (!(cin >> n_in >> m_in)) return 0;
    vector<int> targets(m_in);
    for (int i = 0; i < m_in; i++) { 
        int r, c; cin >> r >> c; targets[i] = r * N + c; 
    }
    fill(grid, grid + N2, false);
    int cur_v = targets[0];
    auto start_time = chrono::steady_clock::now();

    for (int k = 1; k < m_in; k++) {
        int tr = targets[k];
        fast_dijkstra(cur_v, grid, d_cur, p_initial);
        
        // Calculate look-ahead baseline turn costs for pruning
        long long best_eval = d_cur[tr];
        int lv = tr;
        for (int fk = k + 1; fk < min(m_in, k + MAX_LOOK_AHEAD); ++fk) {
            fast_dijkstra(lv, grid, d_temp);
            best_eval += d_temp[targets[fk]];
            lv = targets[fk];
        }

        int b_bv = -1, b_av = -1; char b_adir = ' ';
        // Optimize path by potentially flipping one block to enable Slides or better paths
        for (int br = 0; br < N; br++) {
            if (chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start_time).count() > 1900) break;
            for (int bc = 0; bc < N; bc++) {
                int bv = br * N + bc;
                if (bv == tr || bv == cur_v) continue;

                grid[bv] = !grid[bv];
                fast_backward_dijkstra(tr, grid, bdm_mod);
                
                long long mck = INF; int cav = -1; char cad = ' ';
                // Check all squares adjacent to the flip target to start from
                for (int d = 0; d < 4; ++d) {
                    int ar = br + dr[d], ac = bc + dc[d];
                    if (ar >= 0 && ar < N && ac >= 0 && ac < N) {
                        int av = ar * N + ac;
                        if (d_cur[av] < INF && bdm_mod[av] < INF) {
                            long long cost = (long long)d_cur[av] + 1 + bdm_mod[av];
                            if (cost < mck) { mck = cost; cav = av; cad = dchar[d ^ 1]; }
                        }
                    }
                }
                if (mck < best_eval) {
                    long long eval = mck;
                    int nlv = tr;
                    for (int fk = k + 1; fk < min(m_in, k + MAX_LOOK_AHEAD); ++fk) {
                        fast_dijkstra(nlv, grid, d_temp);
                        eval += d_temp[targets[fk]];
                        nlv = targets[fk];
                        if (eval >= best_eval) break;
                    }
                    if (eval < best_eval) {
                        best_eval = eval; b_bv = bv; b_av = cav; b_adir = cad;
                    }
                }
                grid[bv] = !grid[bv]; // Revert the flip
            }
        }

        if (b_bv == -1) {
            reconstruct_path(cur_v, tr, p_initial, grid, cur_v);
        } else {
            reconstruct_path(cur_v, b_av, p_initial, grid, cur_v);
            cout << "A " << b_adir << "\n";
            grid[b_bv] = !grid[b_bv];
            fast_dijkstra(b_av, grid, d_final, p_final);
            reconstruct_path(b_av, tr, p_final, grid, cur_v);
        }
    }
    return 0;
}
