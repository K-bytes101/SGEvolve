#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>

using namespace std;

/**
 * PROBLEM ANALYSIS & STRATEGY:
 * The goal is to minimize average dirtiness S_bar = (1/L) * sum_{t=L to 2L-1} S_t.
 * Average dirtiness for a repeated route of length L is approximately:
 * S_bar ≈ (1/L) * sum_{i,j} d_{i,j} * sum_{m=1 to k_ij} (g_{i,j,m}^2 / 2),
 * where g_{i,j,m} are the time intervals (gaps) between successive cleanings of square (i,j).
 * 
 * Our approach uses a Spanning Tree DFS tour as the base cleaning route.
 * In a DFS tour on a spanning tree, each edge is traversed twice, so the route length L = 2*(N^2 - 1).
 * The gaps g_{i,j,m} for a square u are determined by the sizes of the connected components
 * formed by removing node u from the tree.
 * 
 * We optimize the spanning tree using Simulated Annealing (SA) with an objective function
 * that estimates the average dirtiness. After SA, we perform greedy shortcutting on the 
 * resulting DFS tour to further reduce the average dirtiness.
 */

const int MAXN = 45;
const int MAXN2 = 1605;
int N, N2;
int flat_d[MAXN2];
string h_walls[MAXN], v_walls[MAXN];
vector<int> tree_adj[MAXN2];
bool is_in_tree_arr[MAXN2][MAXN2];
int parent_in_tree[MAXN2];
int subtree_size[MAXN2];
bool adj_matrix[MAXN2][MAXN2];

struct Edge {
    int u, v;
};
vector<Edge> all_edges;

/**
 * Iteratively computes subtree sizes and parent pointers for the spanning tree.
 * Iterative approach avoids stack overflow issues for deep trees.
 */
void dfs_size_iterative(int start_node) {
    static int stack_arr[MAXN2];
    static int child_idx[MAXN2];
    int top = 0;
    stack_arr[top++] = start_node;
    parent_in_tree[start_node] = -1;
    for (int i = 0; i < N2; ++i) {
        child_idx[i] = 0;
        subtree_size[i] = 1;
    }
    while (top > 0) {
        int u = stack_arr[top - 1];
        if (child_idx[u] < (int)tree_adj[u].size()) {
            int v = tree_adj[u][child_idx[u]++];
            if (v != parent_in_tree[u]) {
                parent_in_tree[v] = u;
                stack_arr[top++] = v;
            }
        } else {
            top--;
            if (top > 0) {
                subtree_size[stack_arr[top - 1]] += subtree_size[u];
            }
        }
    }
}

/**
 * Objective function for SA: Estimates average dirtiness for a DFS tour on the current tree.
 * Formula: S_bar ≈ (1 / (N^2-1)) * sum_{u} (d_u * sum_{branches} size_of_branch^2).
 */
double compute_tree_avg_dirtiness() {
    double total = 0;
    for (int u = 0; u < N2; ++u) {
        long long sum_sq = 0;
        for (int v : tree_adj[u]) {
            if (v == parent_in_tree[u]) {
                long long s = N2 - subtree_size[u];
                sum_sq += s * s;
            } else {
                long long s = subtree_size[v];
                sum_sq += s * s;
            }
        }
        total += (double)flat_d[u] * (double)sum_sq;
    }
    return total / (double)(N2 - 1);
}

// Spanning tree modification utilities
void add_tree_edge(int u, int v) {
    tree_adj[u].push_back(v);
    tree_adj[v].push_back(u);
    is_in_tree_arr[u][v] = is_in_tree_arr[v][u] = true;
}

void remove_tree_edge(int u, int v) {
    tree_adj[u].erase(remove(tree_adj[u].begin(), tree_adj[u].end(), v), tree_adj[u].end());
    tree_adj[v].erase(remove(tree_adj[v].begin(), tree_adj[v].end(), u), tree_adj[v].end());
    is_in_tree_arr[u][v] = is_in_tree_arr[v][u] = false;
}

vector<int> get_path_to_root(int u) {
    vector<int> path;
    while (u != -1) {
        path.push_back(u);
        u = parent_in_tree[u];
    }
    return path;
}

// Union-Find for initial spanning tree construction
int dsu_arr[MAXN2];
int find_dsu(int i) {
    return dsu_arr[i] == i ? i : dsu_arr[i] = find_dsu(dsu_arr[i]);
}

void gen_tour_recursive(int u, int p, vector<int>& tour) {
    tour.push_back(u);
    for (int v : tree_adj[u]) {
        if (v != p) {
            gen_tour_recursive(v, u, tour);
            tour.push_back(u);
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // 1. Read input
    if (!(cin >> N)) return 0;
    N2 = N * N;
    for (int i = 0; i < N - 1; i++) cin >> h_walls[i];
    for (int i = 0; i < N; i++) cin >> v_walls[i];
    for (int i = 0; i < N2; i++) cin >> flat_d[i];

    // Identify all valid moves (edges) in the grid
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int u = i * N + j;
            if (i + 1 < N && h_walls[i][j] == '0') {
                int v = (i + 1) * N + j;
                all_edges.push_back({u, v});
                adj_matrix[u][v] = adj_matrix[v][u] = true;
            }
            if (j + 1 < N && v_walls[i][j] == '0') {
                int v = i * N + (j + 1);
                all_edges.push_back({u, v});
                adj_matrix[u][v] = adj_matrix[v][u] = true;
            }
        }
    }

    // 2. Initialize spanning tree
    mt19937 rng(123);
    vector<Edge> edges = all_edges;
    shuffle(edges.begin(), edges.end(), rng);
    for (int i = 0; i < N2; i++) dsu_arr[i] = i;
    for (auto& e : edges) {
        int r1 = find_dsu(e.u), r2 = find_dsu(e.v);
        if (r1 != r2) {
            dsu_arr[r1] = r2;
            add_tree_edge(e.u, e.v);
        }
    }

    // 3. Simulated Annealing to optimize spanning tree
    dfs_size_iterative(0);
    double current_avg = compute_tree_avg_dirtiness();
    
    double start_temp = 1e6;
    double end_temp = 1e2;
    auto start_time = chrono::steady_clock::now();
    int iterations = 0;

    while (true) {
        if ((iterations & 255) == 0) {
            auto now = chrono::steady_clock::now();
            double elapsed = chrono::duration<double>(now - start_time).count();
            if (elapsed > 1.85) break;
            double progress = elapsed / 1.85;
            double T = start_temp * pow(end_temp / start_temp, progress);

            // Edge swap move: Add a non-tree edge, find the cycle, remove a tree edge from it
            Edge e_in = all_edges[uniform_int_distribution<int>(0, (int)all_edges.size() - 1)(rng)];
            if (is_in_tree_arr[e_in.u][e_in.v]) { iterations++; continue; }

            vector<int> path_u = get_path_to_root(e_in.u);
            vector<int> path_v = get_path_to_root(e_in.v);
            int iu = (int)path_u.size() - 1, iv = (int)path_v.size() - 1;
            while (iu >= 0 && iv >= 0 && path_u[iu] == path_v[iv]) { iu--; iv--; }
            vector<Edge> cycle_edges;
            for (int k = 0; k <= iu; k++) cycle_edges.push_back({path_u[k], path_u[k+1]});
            for (int k = 0; k <= iv; k++) cycle_edges.push_back({path_v[k], path_v[k+1]});
            if (cycle_edges.empty()) { iterations++; continue; }

            Edge e_out = cycle_edges[uniform_int_distribution<int>(0, (int)cycle_edges.size() - 1)(rng)];
            remove_tree_edge(e_out.u, e_out.v);
            add_tree_edge(e_in.u, e_in.v);
            dfs_size_iterative(0);
            double next_avg = compute_tree_avg_dirtiness();

            if (next_avg < current_avg || uniform_real_distribution<double>(0, 1)(rng) < exp((current_avg - next_avg) / T)) {
                current_avg = next_avg;
            } else {
                remove_tree_edge(e_in.u, e_in.v);
                add_tree_edge(e_out.u, e_out.v);
                dfs_size_iterative(0);
            }
        }
        iterations++;
    }

    // 4. Generate DFS tour and perform greedy shortcutting
    vector<int> tour;
    gen_tour_recursive(0, -1, tour);
    int L = (int)tour.size();
    vector<int> counts(N2, 0);
    for (int i = 0; i < L - 1; i++) counts[tour[i]]++;

    vector<bool> removed(L, false);
    static int counts_in_seg[MAXN2];
    for (int i = 0; i < L - 1; i++) {
        if (removed[i]) continue;
        int best_j = -1;
        for (int k = 0; k < N2; k++) counts_in_seg[k] = 0;
        bool possible = true;
        for (int j = i + 1; j < L; j++) {
            if (removed[j]) continue;
            // A shortcut from tour[i] to tour[j] is valid if they are adjacent and 
            // no square is completely skipped by the shortcut.
            if (j > i + 1 && adj_matrix[tour[i]][tour[j]] && possible) {
                best_j = j;
            }
            int u = tour[j];
            counts_in_seg[u]++;
            if (counts_in_seg[u] == counts[u]) possible = false;
        }
        if (best_j != -1) {
            for (int k = i + 1; k < best_j; k++) {
                if (!removed[k]) {
                    counts[tour[k]]--;
                    removed[k] = true;
                }
            }
        }
    }

    // 5. Build final result move string
    vector<int> final_nodes;
    for (int i = 0; i < L; i++) if (!removed[i]) final_nodes.push_back(tour[i]);

    string res = "";
    for (size_t i = 0; i + 1 < final_nodes.size(); i++) {
        int u = final_nodes[i], v = final_nodes[i+1];
        if (v == u + 1) res += 'R';
        else if (v == u - 1) res += 'L';
        else if (v == u + N) res += 'D';
        else if (v == u - N) res += 'U';
    }
    cout << res << endl;

    return 0;
}
