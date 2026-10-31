#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <random>
#include <cmath>
#include <cstring>

using namespace std;

int N, T_max;
int initial_board[10][10];
int initial_er, initial_ec;

// Global buffers to avoid allocation
int current_board[10][10];
int next_board[10][10];
bool vis[10][10];
int qr[100], qc[100];

int get_tree_score(const int board[10][10], int& total_edges) {
    int max_s = 0;
    total_edges = 0;
    memset(vis, 0, sizeof(vis));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (board[i][j] != 0 && !vis[i][j]) {
                int nodes = 0, edges = 0;
                int head = 0, tail = 0;
                qr[tail] = i; qc[tail] = j; tail++;
                vis[i][j] = true;
                while (head < tail) {
                    int r = qr[head], c = qc[head]; head++;
                    nodes++;
                    int type_u = board[r][c];
                    // mask: 1 left, 2 up, 4 right, 8 down
                    if (r > 0 && (type_u & 2) && (board[r-1][c] & 8)) {
                        edges++;
                        if (!vis[r-1][c]) { vis[r-1][c] = true; qr[tail] = r-1; qc[tail] = c; tail++; }
                    }
                    if (r < N - 1 && (type_u & 8) && (board[r+1][c] & 2)) {
                        edges++;
                        if (!vis[r+1][c]) { vis[r+1][c] = true; qr[tail] = r+1; qc[tail] = c; tail++; }
                    }
                    if (c > 0 && (type_u & 1) && (board[r][c-1] & 4)) {
                        edges++;
                        if (!vis[r][c-1]) { vis[r][c-1] = true; qr[tail] = r; qc[tail] = c-1; tail++; }
                    }
                    if (c < N - 1 && (type_u & 4) && (board[r][c+1] & 1)) {
                        edges++;
                        if (!vis[r][c+1]) { vis[r][c+1] = true; qr[tail] = r; qc[tail] = c+1; tail++; }
                    }
                }
                if (edges == (nodes - 1) * 2) max_s = max(max_s, nodes);
                total_edges += edges / 2;
            }
        }
    }
    return max_s;
}

double get_score(int tree_size, int move_count, int total_edges) {
    if (tree_size == N * N - 1) {
        return 2000000.0 + (T_max - move_count);
    }
    return (double)tree_size * 1000.0 + total_edges;
}

bool simulate(const string& moves, int& er, int& ec, int board[10][10]) {
    er = initial_er; ec = initial_ec;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            board[i][j] = initial_board[i][j];
        }
    }
    for (char m : moves) {
        int nr = er, nc = ec;
        if (m == 'U') nr--;
        else if (m == 'D') nr++;
        else if (m == 'L') nc--;
        else if (m == 'R') nc++;
        if (nr < 0 || nr >= N || nc < 0 || nc >= N) return false;
        board[er][ec] = board[nr][nc];
        er = nr; ec = nc;
    }
    board[er][ec] = 0;
    return true;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> N >> T_max)) return 0;
    for (int i = 0; i < N; i++) {
        string row; cin >> row;
        for (int j = 0; j < N; j++) {
            int type = (row[j] >= 'a' ? row[j] - 'a' + 10 : row[j] - '0');
            initial_board[i][j] = type;
            if (type == 0) { initial_er = i; initial_ec = j; }
        }
    }

    auto start_time = chrono::steady_clock::now();
    mt19937 rng(42);
    uniform_real_distribution<double> dist_u(0, 1);

    string current_moves = "";
    int current_er, current_ec;
    simulate(current_moves, current_er, current_ec, current_board);
    int current_edges;
    int current_tree = get_tree_score(current_board, current_edges);
    double current_score = get_score(current_tree, 0, current_edges);

    string best_moves = current_moves;
    double best_score = current_score;

    double initial_temp = 50.0;
    double final_temp = 0.1;

    long long iter = 0;
    while (true) {
        if ((iter & 0x3FF) == 0) {
            auto elapsed = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start_time).count();
            if (elapsed > 2900) break;
        }
        iter++;

        string next_moves = current_moves;
        int m = rng() % 5;
        if (m == 0 && next_moves.size() < T_max) { // Append
            next_moves += "UDLR"[rng() % 4];
        } else if (m == 1 && next_moves.size() > 0) { // Pop
            next_moves.pop_back();
        } else if (m == 2 && next_moves.size() > 0) { // Replace
            next_moves[rng() % next_moves.size()] = "UDLR"[rng() % 4];
        } else if (m == 3 && next_moves.size() > 0) { // Delete
            next_moves.erase(next_moves.begin() + (rng() % next_moves.size()));
        } else if (m == 4 && next_moves.size() < T_max) { // Insert
            next_moves.insert(next_moves.begin() + (rng() % (next_moves.size() + 1)), "UDLR"[rng() % 4]);
        }

        int next_er, next_ec;
        if (simulate(next_moves, next_er, next_ec, next_board)) {
            int next_edges;
            int next_tree = get_tree_score(next_board, next_edges);
            double next_score = get_score(next_tree, (int)next_moves.size(), next_edges);

            auto elapsed = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start_time).count();
            double temp = initial_temp * pow(final_temp / initial_temp, elapsed / 2900.0);

            if (next_score >= current_score || dist_u(rng) < exp((next_score - current_score) / temp)) {
                current_moves = next_moves;
                current_score = next_score;
                if (current_score > best_score) {
                    best_score = current_score;
                    best_moves = current_moves;
                }
            }
        }
    }

    cout << best_moves << endl;

    return 0;
}
