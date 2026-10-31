#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <algorithm>

using namespace std;

const int GRID_SIZE = 30;
const int TURNS = 300;

struct Pet { int x, y, type; };
struct Human { int x, y; };

int N_pets, M_humans;
vector<Pet> pets;
vector<Human> humans;
bool is_blocked[32][32];

// Check if a square (tx, ty) can be blocked according to the contest rules.
bool can_block(int tx, int ty, const vector<Human>& current_h) {
    if (tx < 1 || tx > GRID_SIZE || ty < 1 || ty > GRID_SIZE) return false;
    if (is_blocked[tx][ty]) return false;
    // Rule: Cannot block if square contains humans at the start of this turn.
    for (int i = 0; i < M_humans; i++) {
        if (current_h[i].x == tx && current_h[i].y == ty) return false;
    }
    // Rule: Cannot block if square contains a pet or any adjacent square contains a pet at the start.
    for (int i = 0; i < N_pets; i++) {
        if (abs(pets[i].x - tx) + abs(pets[i].y - ty) <= 1) return false;
    }
    return true;
}

// BFS to find the shortest path move for human h_idx towards the target (tx, ty).
string bfs_move(int h_idx, int tx, int ty, const bool being_blocked[32][32]) {
    if (humans[h_idx].x == tx && humans[h_idx].y == ty) return ".";
    int dist[32][32];
    pair<int, int> parent[32][32];
    for (int r = 0; r < 32; r++) for (int c = 0; c < 32; c++) dist[r][c] = -1;
    queue<pair<int, int>> q;
    q.push({humans[h_idx].x, humans[h_idx].y});
    dist[humans[h_idx].x][humans[h_idx].y] = 0;
    int dx[] = {-1, 1, 0, 0}, dy[] = {0, 0, -1, 1};
    char mc[] = {'U', 'D', 'L', 'R'};
    while (!q.empty()) {
        pair<int, int> curr = q.front(); q.pop();
        if (curr.first == tx && curr.second == ty) {
            pair<int, int> p = {tx, ty};
            while (parent[p.first][p.second].first != humans[h_idx].x || parent[p.first][p.second].second != humans[h_idx].y) {
                p = parent[p.first][p.second];
            }
            if (p.first < humans[h_idx].x) return "U";
            if (p.first > humans[h_idx].x) return "D";
            if (p.second < humans[h_idx].y) return "L";
            if (p.second > humans[h_idx].y) return "R";
        }
        for (int k = 0; k < 4; k++) {
            int nx = curr.first + dx[k], ny = curr.second + dy[k];
            if (nx < 1 || nx > GRID_SIZE || ny < 1 || ny > GRID_SIZE || is_blocked[nx][ny] || being_blocked[nx][ny]) continue;
            if (dist[nx][ny] == -1) {
                dist[nx][ny] = dist[curr.first][curr.second] + 1;
                parent[nx][ny] = curr;
                q.push({nx, ny});
            }
        }
    }
    return ".";
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> N_pets)) return 0;
    pets.resize(N_pets);
    for (int i = 0; i < N_pets; i++) cin >> pets[i].x >> pets[i].y >> pets[i].type;
    cin >> M_humans;
    humans.resize(M_humans);
    for (int i = 0; i < M_humans; i++) cin >> humans[i].x >> humans[i].y;

    vector<int> wall_cols = {29, 26, 23, 20, 17, 14, 11, 8, 5, 2};
    vector<int> wall_prog(wall_cols.size(), 1);
    bool gate_done[10] = {false};

    for (int t = 0; t < TURNS; t++) {
        string actions(M_humans, '.');
        bool being_blocked_this_turn[32][32] = {false};
        vector<Human> next_h_pos = humans;

        // Step 1: Identify humans who will block a square this turn.
        for (int i = 0; i < M_humans; i++) {
            // Check wall segments
            for (int k = 0; k < (int)wall_cols.size(); k++) {
                if (k % M_humans != i || wall_prog[k] > GRID_SIZE) continue;
                int tx = wall_prog[k], ty = wall_cols[k];
                if (tx == 15) continue;
                int sx = tx, sy = ty + 1;
                if (humans[i].x == sx && humans[i].y == sy) {
                    if (can_block(tx, ty, humans) && !being_blocked_this_turn[tx][ty]) {
                        if (tx < humans[i].x) actions[i] = 'u';
                        else if (tx > humans[i].x) actions[i] = 'd';
                        else if (ty < humans[i].y) actions[i] = 'l';
                        else actions[i] = 'r';
                        being_blocked_this_turn[tx][ty] = true;
                    }
                }
                break;
            }
            // Check gates
            if (actions[i] == '.') {
                for (int k = 0; k < (int)wall_cols.size(); k++) {
                    if (k % M_humans != i || wall_prog[k] <= GRID_SIZE || gate_done[k]) continue;
                    int tx = 15, ty = wall_cols[k];
                    int sx = 15, sy = ty + 1;
                    if (humans[i].x == sx && humans[i].y == sy) {
                        int min_y = ty + 1, max_y = (k == 0) ? 30 : wall_cols[k-1] - 1;
                        bool safe = true;
                        for (int p = 0; p < N_pets; p++) if (pets[p].y >= min_y && pets[p].y <= max_y) safe = false;
                        if (safe || t > 250) {
                            if (can_block(tx, ty, humans) && !being_blocked_this_turn[tx][ty]) {
                                actions[i] = 'l';
                                being_blocked_this_turn[tx][ty] = true;
                                gate_done[k] = true;
                            }
                        }
                    }
                    break;
                }
            }
        }

        // Step 2: Identify movement for humans not blocking.
        for (int i = 0; i < M_humans; i++) {
            if (actions[i] == '.') {
                int tx = -1, ty = -1;
                for (int k = 0; k < (int)wall_cols.size(); k++) {
                    if (k % M_humans != i) continue;
                    if (wall_prog[k] <= GRID_SIZE) {
                        tx = wall_prog[k]; ty = wall_cols[k] + 1;
                        if (tx == 15) { 
                            int next_seg = tx + 1;
                            while (next_seg <= GRID_SIZE && is_blocked[next_seg][wall_cols[k]]) next_seg++;
                            tx = next_seg;
                        }
                        break;
                    } else if (!gate_done[k]) {
                        tx = 15; ty = wall_cols[k] + 1;
                        break;
                    }
                }
                if (tx != -1 && tx <= GRID_SIZE) {
                    string mv = bfs_move(i, tx, ty, being_blocked_this_turn);
                    actions[i] = mv[0];
                    if (mv == "U") next_h_pos[i].x--; else if (mv == "D") next_h_pos[i].x++;
                    else if (mv == "L") next_h_pos[i].y--; else if (mv == "R") next_h_pos[i].y++;
                }
            }
        }

        cout << actions << endl;

        for (int i = 0; i < M_humans; i++) {
            if (actions[i] >= 'a' && actions[i] <= 'z') {
                int bx = humans[i].x, by = humans[i].y;
                if (actions[i] == 'u') bx--; else if (actions[i] == 'd') bx++;
                else if (actions[i] == 'l') by--; else if (actions[i] == 'r') by++;
                if (bx >= 1 && bx <= GRID_SIZE && by >= 1 && by <= GRID_SIZE) is_blocked[bx][by] = true;
            } else {
                humans[i] = next_h_pos[i];
            }
        }

        for (int k = 0; k < 10; k++) {
            while (wall_prog[k] <= GRID_SIZE && (wall_prog[k] == 15 || is_blocked[wall_prog[k]][wall_cols[k]])) wall_prog[k]++;
        }

        for (int i = 0; i < N_pets; i++) {
            string pm; if (!(cin >> pm)) break;
            for (char c : pm) {
                if (c == 'U') pets[i].x--; else if (c == 'D') pets[i].x++;
                else if (c == 'L') pets[i].y--; else if (c == 'R') pets[i].y++;
            }
        }
    }
    return 0;
}
