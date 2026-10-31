#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

// Takahashi City grid size is fixed to 50x50 and number of wards is 100.
const int N = 50;
const int M = 100;

int original_grid[N][N];
int current_grid[N][N];
int best_grid[N][N];
int E[M + 1][M + 1];
int ward_size[M + 1];
bool original_adj[M + 1][M + 1];
bool is_locked[N][N];
int dx[] = {0, 0, 1, -1};
int dy[] = {1, -1, 0, 0};

// Static structures for BFS connectivity checks.
int vis[N][N];
int vis_token = 0;
pair<int, int> q_arr[N * N];

/**
 * Helper to get grid color, returning 0 (outside) for out-of-bounds coordinates.
 */
inline int get_color(int r, int c) {
    if (r < 0 || r >= N || c < 0 || c >= N) return 0;
    return current_grid[r][c];
}

/**
 * Checks if a ward (color c_old) remains connected if square (r, c) is removed.
 * Ensures that all same-colored neighbors of (r, c) can still reach each other.
 */
bool check_connectivity(int r, int c, int c_old, const vector<pair<int, int>>& neighbors) {
    if (neighbors.size() <= 1) return true;
    vis_token++;
    int head = 0, tail = 0;
    q_arr[tail++] = neighbors[0];
    vis[neighbors[0].first][neighbors[0].second] = vis_token;
    vis[r][c] = vis_token; // Act as a wall (as square (r, c) will be changed to 0).

    int found_count = 1;
    while (head < tail) {
        pair<int, int> curr = q_arr[head++];
        for (int k = 0; k < 4; k++) {
            int ni = curr.first + dx[k], nj = curr.second + dy[k];
            if (ni >= 0 && ni < N && nj >= 0 && nj < N && current_grid[ni][nj] == c_old && vis[ni][nj] != vis_token) {
                vis[ni][nj] = vis_token;
                q_arr[tail++] = {ni, nj};
                for (auto& neighbor : neighbors) {
                    if (ni == neighbor.first && nj == neighbor.second) {
                        found_count++;
                        break;
                    }
                }
                // Stop BFS early if all target neighbor squares are reached.
                if (found_count == (int)neighbors.size()) return true;
            }
        }
    }
    return false;
}

/**
 * Validates if square (r, c) can be changed to color 0 without violating connectivity or adjacency rules.
 */
bool can_remove(int r, int c) {
    int c_old = current_grid[r][c];
    if (ward_size[c_old] <= 1) return false;

    // Constraint: Color 0 must remain connected.
    // Ensure the new 0-tile is part of a component that touches the boundary.
    bool next_to_0 = false;
    if (r == 0 || r == N - 1 || c == 0 || c == N - 1) next_to_0 = true;
    else {
        for (int k = 0; k < 4; k++) {
            if (current_grid[r + dx[k]][c + dy[k]] == 0) {
                next_to_0 = true;
                break;
            }
        }
    }
    if (!next_to_0) return false;

    // Check ward-ward and ward-0 adjacency constraints.
    vector<pair<int, int>> c_old_neighbors;
    int neighbor_wards[4];
    int edges_to_ward[4];
    int num_neighbor_wards = 0;

    for (int k = 0; k < 4; k++) {
        int ni = r + dx[k], nj = c + dy[k];
        int d = get_color(ni, nj);
        if (d == c_old) {
            c_old_neighbors.push_back({ni, nj});
        } else if (d > 0) {
            int found_idx = -1;
            for (int j = 0; j < num_neighbor_wards; j++) {
                if (neighbor_wards[j] == d) { found_idx = j; break; }
            }
            if (found_idx == -1) {
                neighbor_wards[num_neighbor_wards] = d;
                edges_to_ward[num_neighbor_wards] = 1;
                num_neighbor_wards++;
            } else {
                edges_to_ward[found_idx]++;
            }
        }
    }

    // Ensure ward-ward adjacencies are not lost.
    for (int j = 0; j < num_neighbor_wards; j++) {
        if (E[c_old][neighbor_wards[j]] - edges_to_ward[j] <= 0) return false;
    }

    // Ensure ward c_old remains connected.
    return check_connectivity(r, c, c_old, c_old_neighbors);
}

/**
 * Updates the current grid and global adjacency tallies.
 */
void remove_and_update(int r, int c) {
    int c_old = current_grid[r][c];
    for (int k = 0; k < 4; k++) {
        int ni = r + dx[k], nj = c + dy[k];
        int d = (ni < 0 || ni >= N || nj < 0 || nj >= N) ? 0 : current_grid[ni][nj];
        E[c_old][d]--;
        E[d][c_old]--;
        E[0][d]++;
        E[d][0]++;
    }
    current_grid[r][c] = 0;
    ward_size[c_old]--;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n_in, m_in;
    if (!(cin >> n_in >> m_in)) return 0;
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) cin >> original_grid[i][j];

    // Calculate original ward-to-ward adjacencies.
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int c = original_grid[i][j];
            for (int k = 0; k < 4; k++) {
                int ni = i + dx[k], nj = j + dy[k];
                int d = (ni < 0 || ni >= N || nj < 0 || nj >= N) ? 0 : original_grid[ni][nj];
                original_adj[c][d] = true;
                original_adj[d][c] = true;
            }
        }
    }

    // Identify squares that can never be turned into 0 due to adjacency constraints.
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int c = original_grid[i][j];
            is_locked[i][j] = false;
            // A square is locked if its ward or any neighbor ward was not originally adjacent to 0.
            if (!original_adj[c][0]) is_locked[i][j] = true;
            else {
                for (int k = 0; k < 4; k++) {
                    int ni = i + dx[k], nj = j + dy[k];
                    int d = (ni < 0 || ni >= N || nj < 0 || nj >= N) ? 0 : original_grid[ni][nj];
                    if (d > 0 && d != c && !original_adj[d][0]) {
                        is_locked[i][j] = true;
                        break;
                    }
                }
            }
        }
    }

    vector<pair<int, int>> candidate_coords;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (!is_locked[i][j]) candidate_coords.push_back({i, j});
        }
    }

    auto start_time = chrono::steady_clock::now();
    mt19937 rng(1337);
    int best_E_count = -1;

    // Restart the greedy search from the original grid within the time limit.
    while (chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start_time).count() < 1900) {
        // Initialize current state.
        for (int i = 0; i <= M; i++) {
            ward_size[i] = 0;
            for (int j = 0; j <= M; j++) E[i][j] = 0;
        }
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                current_grid[i][j] = original_grid[i][j];
                ward_size[current_grid[i][j]]++;
            }
        }
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < 4; k++) {
                    int d = get_color(i + dx[k], j + dy[k]);
                    E[current_grid[i][j]][d]++;
                }
            }
        }

        // Apply greedy removals until a local optimum is reached.
        while (true) {
            shuffle(candidate_coords.begin(), candidate_coords.end(), rng);
            bool changed_any = false;
            for (auto& p : candidate_coords) {
                if (current_grid[p.first][p.second] != 0 && can_remove(p.first, p.second)) {
                    remove_and_update(p.first, p.second);
                    changed_any = true;
                }
            }
            if (!changed_any) break;
        }

        int current_E_count = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) if (current_grid[i][j] == 0) current_E_count++;
        }

        if (current_E_count > best_E_count) {
            best_E_count = current_E_count;
            for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) best_grid[i][j] = current_grid[i][j];
        }
    }

    // Output the grid with the maximum number of 0 squares.
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << best_grid[i][j] << (j == N - 1 ? "" : " ");
        }
        cout << "\n";
    }

    return 0;
}
