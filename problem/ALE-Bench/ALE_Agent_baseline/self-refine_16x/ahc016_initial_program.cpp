#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace std;

/**
 * Normal CDF function: Phi(x) = 0.5 * (1 + erf((x - mean) / (sigma * sqrt(2))))
 * This function calculates the probability that a random variable following a normal 
 * distribution with the given mean and sigma is less than x.
 */
double normal_cdf(double x, double mean, double sigma) {
    if (sigma < 1e-9) {
        if (x < mean) return 0.0;
        if (x > mean) return 1.0;
        return 0.5;
    }
    return 0.5 * (1.0 + erf((x - mean) / (sigma * sqrt(2.0))));
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int M;
    double eps;
    // Read the number of graphs M and the error rate epsilon.
    if (!(cin >> M >> eps)) return 0;

    int bestN = 4;
    double bestExpectedScore = -1.0;

    // Iterate through all possible values for N (the number of vertices) to find the one that 
    // maximizes the expected score. The score is calculated based on the expected number of 
    // errors using an edge count encoding/decoding method.
    for (int N = 4; N <= 100; ++N) {
        int K = N * (N - 1) / 2; // Maximum number of possible edges.
        vector<int> Es(M);
        for (int s = 0; s < M; ++s) {
            // Encode the number of edges for each graph G_s.
            Es[s] = (int)round((double)s * K / (M - 1));
        }

        // Standard deviation of the edge count after noise (Binomial sum approximation).
        double sigma = sqrt((double)K * eps * (1.0 - eps));
        double P_avg = 0.0;

        for (int s = 0; s < M; ++s) {
            // Count how many graphs have the same number of edges as G_s.
            int c = 0;
            for (int j = 0; j < M; ++j) {
                if (Es[j] == Es[s]) c++;
            }

            // The expected mean number of edges in the noisy graph H generated from G_s.
            double mu_s = (double)Es[s] * (1.0 - 2.0 * eps) + (double)K * eps;

            // Find the closest different edge counts to identify the decision boundaries.
            double E_prev = -1e18;
            for (int j = 0; j < M; ++j) {
                if (Es[j] < Es[s]) {
                    if (E_prev < -1e17 || Es[j] > E_prev) {
                        E_prev = (double)Es[j];
                    }
                }
            }

            double E_next = 1e18;
            for (int j = 0; j < M; ++j) {
                if (Es[j] > Es[s]) {
                    if (E_next > 1e17 || Es[j] < E_next) {
                        E_next = (double)Es[j];
                    }
                }
            }

            double low, high;
            if (E_prev < -1e17) {
                low = -1e18; // No lower bound for the first category.
            } else {
                double mu_prev = E_prev * (1.0 - 2.0 * eps) + (double)K * eps;
                low = (mu_s + mu_prev) / 2.0;
            }

            if (E_next > 1e17) {
                high = 1e18; // No upper bound for the last category.
            } else {
                double mu_next = E_next * (1.0 - 2.0 * eps) + (double)K * eps;
                high = (mu_s + mu_next) / 2.0;
            }

            // Estimate the probability of correct decoding for the s-th graph.
            double prob_correct_in_range;
            if (sigma < 1e-9) {
                prob_correct_in_range = 1.0;
            } else {
                prob_correct_in_range = normal_cdf(high, mu_s, sigma) - normal_cdf(low, mu_s, sigma);
            }
            // If multiple graphs have the same edge count, they are indistinguishable.
            P_avg += prob_correct_in_range / (double)c;
        }

        // Calculate the overall expected probability of a correct prediction.
        P_avg /= (double)M;
        double p_error = 1.0 - P_avg;
        // Expected absolute score formula: (1 - 0.1 * p_error)^100 / N.
        double expectedScore = pow(1.0 - 0.1 * p_error, 100) / (double)N;

        if (expectedScore > bestExpectedScore) {
            bestExpectedScore = expectedScore;
            bestN = N;
        }
    }

    // Set up the selected N and the corresponding graphs G_0, ..., G_{M-1}.
    int N = bestN;
    int K = N * (N - 1) / 2;
    vector<int> Es(M);
    for (int s = 0; s < M; ++s) {
        Es[s] = (int)round((double)s * K / (M - 1));
    }

    // Output the chosen number of vertices N.
    cout << N << endl;
    // Output the string representations for the M graphs.
    for (int s = 0; s < M; ++s) {
        string g(K, '0');
        for (int i = 0; i < Es[s]; ++i) {
            g[i] = '1';
        }
        cout << g << endl;
    }
    cout.flush();

    // Process 100 noisy graph queries.
    for (int q = 0; q < 100; ++q) {
        string H;
        if (!(cin >> H)) break;
        // Count the number of edges in the received graph H.
        int EH = 0;
        for (char c : H) {
            if (c == '1') EH++;
        }

        // Find the graph s whose expected noisy edge count mean is closest to EH.
        int best_s = 0;
        double min_diff = 1e18;
        for (int s = 0; s < M; ++s) {
            double mu_s = (double)Es[s] * (1.0 - 2.0 * eps) + (double)K * eps;
            double diff = abs((double)EH - mu_s);
            if (diff < min_diff) {
                min_diff = diff;
                best_s = s;
            }
        }
        // Output the prediction for the current query.
        cout << best_s << endl;
        cout.flush();
    }

    return 0;
}
