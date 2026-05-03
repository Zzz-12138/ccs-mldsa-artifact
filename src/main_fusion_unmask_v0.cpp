#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <omp.h>
#include <queue>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace mldsa_sieve {
namespace {

constexpr int kPolyN = 256;
constexpr int32_t kQ = 8380417;
constexpr size_t kArenaBlockStates = 1u << 15;
constexpr size_t kInitialFrontierReserve = 1u << 22;
constexpr size_t kMaxArenaBlocks = 7000;

static const int32_t kZetas[256] = {
         0,    25847, -2608894,  -518909,   237124,  -777960,  -876248,   466468,
   1826347,  2353451,  -359251, -2091905,  3119733, -2884855,  3111497,  2680103,
   2725464,  1024112, -1079900,  3585928,  -549488, -1119584,  2619752, -2108549,
  -2118186, -3859737, -1399561, -3277672,  1757237,   -19422,  4010497,   280005,
   2706023,    95776,  3077325,  3530437, -1661693, -3592148, -2537516,  3915439,
  -3861115, -3043716,  3574422, -2867647,  3539968,  -300467,  2348700,  -539299,
  -1699267, -1643818,  3505694, -3821735,  3507263, -2140649, -1600420,  3699596,
    811944,   531354,   954230,  3881043,  3900724, -2556880,  2071892, -2797779,
  -3930395, -1528703, -3677745, -3041255, -1452451,  3475950,  2176455, -1585221,
  -1257611,  1939314, -4083598, -1000202, -3190144, -3157330, -3632928,   126922,
   3412210,  -983419,  2147896,  2715295, -2967645, -3693493,  -411027, -2477047,
   -671102, -1228525,   -22981, -1308169,  -381987,  1349076,  1852771, -1430430,
  -3343383,   264944,   508951,  3097992,    44288, -1100098,   904516,  3958618,
  -3724342,    -8578,  1653064, -3249728,  2389356,  -210977,   759969, -1316856,
    189548, -3553272,  3159746, -1851402, -2409325,  -177440,  1315589,  1341330,
   1285669, -1584928,  -812732, -1439742, -3019102, -3881060, -3628969,  3839961,
   2091667,  3407706,  2316500,  3817976, -3342478,  2244091, -2446433, -3562462,
    266997,  2434439, -1235728,  3513181, -3520352, -3759364, -1197226, -3193378,
    900702,  1859098,   909542,   819034,   495491, -1613174,   -43260,  -522500,
   -655327, -3122442,  2031748,  3207046, -3556995,  -525098,  -768622, -3595838,
    342297,   286988, -2437823,  4108315,  3437287, -3342277,  1735879,   203044,
   2842341,  2691481, -2590150,  1265009,  4055324,  1247620,  2486353,  1595974,
  -3767016,  1250494,  2635921, -3548272, -2994039,  1869119,  1903435, -1050970,
  -1333058,  1237275, -3318210, -1430225,  -451100,  1312455,  3306115, -1962642,
  -1279661,  1917081, -2546312, -1374803,  1500165,   777191,  2235880,  3406031,
   -542412, -2831860, -1671176, -1846953, -2584293, -3724270,   594136, -3776993,
  -2013608,  2432395,  2454455,  -164721,  1957272,  3369112,   185531, -1207385,
  -3183426,   162844,  1616392,  3014001,   810149,  1652634, -3694233, -1799107,
  -3038916,  3523897,  3866901,   269760,  2213111,  -975884,  1717735,   472078,
   -426683,  1723600, -1803090,  1910376, -1667432, -1104333,  -260646, -3833893,
  -2939036, -2235985,  -420899, -2286327,   183443,  -976891,  1612842, -3545687,
   -554416,  3919660,   -48306, -1362209,  3937738,  1400424,  -846154,  1976782
};

inline int32_t MontgomeryReduce(int64_t a) {
    constexpr int32_t kQInv = 58728449;
    const int32_t t = static_cast<int32_t>(a) * kQInv;
    const int64_t m = static_cast<int64_t>(t) * kQ;
    return static_cast<int32_t>((a - m) >> 32);
}

inline int32_t NormalizeModQ(int32_t x) {
    int64_t v = static_cast<int64_t>(x) % kQ;
    if (v < 0) v += kQ;
    return static_cast<int32_t>(v);
}

inline int32_t CenteredModQ(int32_t x) {
    int32_t v = NormalizeModQ(x);
    if (v > (kQ / 2)) v -= kQ;
    return v;
}

void invntt_tomont(int32_t poly[kPolyN]) {
    constexpr int32_t kInvN = 41978;
    unsigned int k = kPolyN;

    for (unsigned int len = 1; len < kPolyN; len <<= 1) {
        for (unsigned int start = 0; start < kPolyN;) {
            const int32_t zeta = -kZetas[--k];
            unsigned int j = start;
            for (; j < start + len; ++j) {
                const int32_t t = poly[j];
                poly[j] = t + poly[j + len];
                poly[j + len] = t - poly[j + len];
                poly[j + len] = MontgomeryReduce(static_cast<int64_t>(zeta) * poly[j + len]);
            }
            start = j + len;
        }
    }

    for (int i = 0; i < kPolyN; ++i) {
        poly[i] = MontgomeryReduce(static_cast<int64_t>(kInvN) * poly[i]);
    }
}

bool check_eta(int32_t poly[kPolyN], int eta) {
    bool all_zero = true;
    for (int i = 0; i < kPolyN; ++i) {
        int32_t spatial_val = MontgomeryReduce(static_cast<int64_t>(poly[i]));
        spatial_val = CenteredModQ(spatial_val);
        poly[i] = spatial_val;
        if (spatial_val != 0) all_zero = false;
        if (spatial_val < -eta || spatial_val > eta) return false;
    }
    return !all_zero;
}

}  // namespace

struct AStarEnumerationResult {
    bool found = false;
    int64_t oracle_checks = 0;
    int64_t states_generated = 0;
    size_t peak_frontier_size = 0;
    uint32_t found_rank_penalty = 0;
    std::array<int32_t, kPolyN> found_ntt{};
    std::array<int32_t, kPolyN> found_coeff{};
};

class LdAStarPriorityEnumerator {
public:
    struct State {
        uint32_t rank_penalty;
        uint16_t last_modified_dim;
        uint16_t num_altered_dims;
        uint64_t serial;
        std::array<uint16_t, kPolyN> depth_idx;
    };

    static_assert(std::is_trivially_copyable<State>::value, "State must be trivially copyable.");

    LdAStarPriorityEnumerator(
        const std::vector<std::vector<float>>& scores,
        const std::vector<std::vector<int32_t>>& top_guesses,
        int eta)
        : eta_(eta),
          frontier_(StatePtrLess{}, MakeReservedFrontierContainer()) {
        Preprocess(scores, top_guesses);
        arena_.ReserveBlocks(64);
    }

    AStarEnumerationResult Run(int64_t budget_limit) {
        AStarEnumerationResult result;
        if (budget_limit <= 0) return result;

        arena_.PrepareThreadCaches(std::max(1, omp_get_max_threads()));

        State* root = NewState();
        root->rank_penalty = 0;
        root->last_modified_dim = 0;
        root->num_altered_dims = 0;
        root->serial = next_serial_.fetch_add(1, std::memory_order_relaxed);
        root->depth_idx.fill(0);

        frontier_.push(root);
        result.states_generated = 1;
        result.peak_frontier_size = 1;

        std::vector<const State*> batch;
        batch.reserve(kBatchSize);

        while (!frontier_.empty() && result.oracle_checks < budget_limit) {
            if (arena_.blocks.size() > kMaxArenaBlocks) break;

            batch.clear();
            const size_t remaining_budget = static_cast<size_t>(budget_limit - result.oracle_checks);
            const size_t target_batch = std::min(kBatchSize, remaining_budget);

            for (size_t i = 0; i < target_batch && !frontier_.empty(); ++i) {
                batch.push_back(frontier_.top());
                frontier_.pop();
            }
            if (batch.empty()) break;

            std::atomic<bool> key_found{false};
            std::atomic<int> best_hit_index{static_cast<int>(batch.size())};
            int64_t batch_checks = 0;

            #pragma omp parallel
            {
                std::array<int32_t, kPolyN> candidate_ntt{};
                std::array<int32_t, kPolyN> candidate_coeff{};
                std::vector<State*> thread_local_children;
                thread_local_children.reserve(4096);
                int64_t local_checks = 0;

                #pragma omp for schedule(static)
                for (int i = 0; i < static_cast<int>(batch.size()); ++i) {
                    if (i >= best_hit_index.load(std::memory_order_acquire)) continue;

                    BuildCandidateNtt(*batch[static_cast<size_t>(i)], candidate_ntt);
                    candidate_coeff = candidate_ntt;
                    ++local_checks;

                    invntt_tomont(candidate_coeff.data());
                    if (!check_eta(candidate_coeff.data(), eta_)) {
                        if (!key_found.load(std::memory_order_acquire)) {
                            Expand(*batch[static_cast<size_t>(i)], thread_local_children);
                        }
                        continue;
                    }

                    key_found.store(true, std::memory_order_release);
                    int observed = best_hit_index.load(std::memory_order_relaxed);
                    while (i < observed &&
                           !best_hit_index.compare_exchange_weak(
                               observed, i,
                               std::memory_order_acq_rel,
                               std::memory_order_relaxed)) {
                    }
                }

                #pragma omp atomic update
                batch_checks += local_checks;

                if (!thread_local_children.empty() && !key_found.load(std::memory_order_acquire)) {
                    #pragma omp critical(ld_astar_frontier_merge)
                    {
                        for (State* child : thread_local_children) {
                            frontier_.push(child);
                        }
                        result.states_generated += static_cast<int64_t>(thread_local_children.size());
                    }
                }
            }

            result.oracle_checks += batch_checks;

            const int winner_index = best_hit_index.load(std::memory_order_acquire);
            if (key_found.load(std::memory_order_acquire) && winner_index < static_cast<int>(batch.size())) {
                const State* winner = batch[static_cast<size_t>(winner_index)];
                std::array<int32_t, kPolyN> candidate_ntt{};
                std::array<int32_t, kPolyN> candidate_coeff{};
                BuildCandidateNtt(*winner, candidate_ntt);
                candidate_coeff = candidate_ntt;
                invntt_tomont(candidate_coeff.data());
                check_eta(candidate_coeff.data(), eta_);

                result.found = true;
                result.found_rank_penalty = winner->rank_penalty;
                result.found_ntt = candidate_ntt;
                result.found_coeff = candidate_coeff;
                return result;
            }

            result.peak_frontier_size = std::max(result.peak_frontier_size, frontier_.size());
        }

        return result;
    }

private:
    struct Dimension {
        int original_index = -1;
        float leading_degree = 0.0f;
        std::vector<int32_t> guess;
    };

    struct Arena {
        struct ThreadCache {
            State* current_block = nullptr;
            size_t next_index = kArenaBlockStates;
        };

        std::vector<std::unique_ptr<State[]>> blocks;
        std::vector<ThreadCache> thread_caches;

        void ReserveBlocks(size_t reserve_blocks) {
            blocks.reserve(reserve_blocks);
        }

        void PrepareThreadCaches(int thread_count) {
            if (thread_count <= 0) thread_count = 1;
            if (static_cast<int>(thread_caches.size()) < thread_count) {
                thread_caches.resize(static_cast<size_t>(thread_count));
            }
        }

        State* Allocate() {
            const int thread_id = omp_in_parallel() ? omp_get_thread_num() : 0;
            if (thread_caches.empty()) PrepareThreadCaches(1);
            ThreadCache& cache = thread_caches[static_cast<size_t>(thread_id)];

            if (cache.current_block == nullptr || cache.next_index == kArenaBlockStates) {
                #pragma omp critical(ld_astar_arena_block)
                {
                    blocks.emplace_back(std::make_unique<State[]>(kArenaBlockStates));
                    cache.current_block = blocks.back().get();
                    cache.next_index = 0;
                }
            }
            return &cache.current_block[cache.next_index++];
        }
    };

    struct StatePtrLess {
        bool operator()(const State* lhs, const State* rhs) const {
            if (lhs->rank_penalty != rhs->rank_penalty) {
                return lhs->rank_penalty > rhs->rank_penalty;
            }
            return lhs->serial > rhs->serial;
        }
    };

    static std::vector<State*> MakeReservedFrontierContainer() {
        std::vector<State*> storage;
        storage.reserve(kInitialFrontierReserve);
        return storage;
    }

    State* NewState() {
        return arena_.Allocate();
    }

    State* CloneState(const State& src) {
        State* dst = NewState();
        std::memcpy(dst, &src, sizeof(State));
        return dst;
    }

    void Preprocess(
        const std::vector<std::vector<float>>& scores,
        const std::vector<std::vector<int32_t>>& top_guesses) {
        std::vector<Dimension> tmp;
        tmp.reserve(kPolyN);

        for (int i = 0; i < kPolyN; ++i) {
            std::vector<std::pair<float, int32_t>> pairs;
            pairs.reserve(scores[i].size());

            for (size_t k = 0; k < scores[i].size() && k < top_guesses[i].size(); ++k) {
                const float score = scores[i][k];
                if (std::isfinite(score)) {
                    pairs.emplace_back(score, NormalizeModQ(top_guesses[i][k]));
                }
            }

            std::sort(
                pairs.begin(),
                pairs.end(),
                [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
                    if (a.first != b.first) return a.first > b.first;
                    return a.second < b.second;
                });

            Dimension dim;
            dim.original_index = i;
            dim.guess.reserve(pairs.size());

            std::vector<float> unique_scores;
            for (const auto& entry : pairs) {
                if (!dim.guess.empty() && dim.guess.back() == entry.second) continue;
                dim.guess.push_back(entry.second);
                unique_scores.push_back(entry.first);
            }

            if (dim.guess.empty()) {
                dim.guess.push_back(0);
                unique_scores.push_back(-std::numeric_limits<float>::infinity());
            }

            dim.leading_degree = unique_scores.size() > 1
                ? unique_scores[0] - unique_scores[1]
                : std::numeric_limits<float>::infinity();
            tmp.push_back(std::move(dim));
        }

        std::stable_sort(
            tmp.begin(),
            tmp.end(),
            [](const Dimension& a, const Dimension& b) {
                if (a.leading_degree != b.leading_degree) {
                    return a.leading_degree < b.leading_degree;
                }
                return a.original_index < b.original_index;
            });

        for (int i = 0; i < kPolyN; ++i) {
            dims_[i] = std::move(tmp[i]);
        }
    }

    void BuildCandidateNtt(const State& state, std::array<int32_t, kPolyN>& candidate_ntt) const {
        for (int sorted_dim = 0; sorted_dim < kPolyN; ++sorted_dim) {
            const Dimension& dim = dims_[sorted_dim];
            candidate_ntt[dim.original_index] = dim.guess[state.depth_idx[sorted_dim]];
        }
    }

    void Expand(const State& current, std::vector<State*>& out_children) {
        constexpr uint16_t kMaxAlterations = 4;

        for (uint16_t sorted_dim = current.last_modified_dim; sorted_dim < kPolyN; ++sorted_dim) {
            const Dimension& dim = dims_[sorted_dim];
            const uint16_t current_depth = current.depth_idx[sorted_dim];

            if (current_depth == 0 && current.num_altered_dims >= kMaxAlterations) continue;

            const size_t next_depth = static_cast<size_t>(current_depth) + 1u;
            if (next_depth >= dim.guess.size()) continue;

            State* child = CloneState(current);
            child->depth_idx[sorted_dim] = static_cast<uint16_t>(next_depth);
            child->last_modified_dim = sorted_dim;
            child->num_altered_dims = current.num_altered_dims + (current_depth == 0 ? 1 : 0);
            child->serial = next_serial_.fetch_add(1, std::memory_order_relaxed);

            const uint32_t step_cost = (sorted_dim < 8) ? 1 : 20;
            child->rank_penalty = current.rank_penalty + step_cost;
            out_children.push_back(child);
        }
    }

    static constexpr size_t kBatchSize = 131072;
    int eta_ = 0;
    std::atomic<uint64_t> next_serial_{0};
    std::array<Dimension, kPolyN> dims_{};
    Arena arena_;
    std::priority_queue<State*, std::vector<State*>, StatePtrLess> frontier_;
};

AStarEnumerationResult EnumerateLdAStarKey(
    const std::vector<std::vector<float>>& scores,
    const std::vector<std::vector<int32_t>>& top_guesses,
    int64_t budget_limit,
    int eta) {
    LdAStarPriorityEnumerator enumerator(scores, top_guesses, eta);
    return enumerator.Run(budget_limit);
}

}  // namespace mldsa_sieve

extern "C" void gpu_solve_round_unmasked(
    bool is_new_instance,
    const std::vector<float>& inst_cs1_traces,
    const std::vector<int32_t>& c_labels,
    int N_cs1,
    int max_N_cs1,
    int T_cs1,
    const std::vector<float>& inst_ay_traces,
    const std::vector<int32_t>& c_ntt,
    const std::vector<int32_t>& z_ntt,
    const std::vector<int32_t>& a_col,
    int N_ay_phys,
    int max_N_ay_phys,
    int T_ay,
    int K_ROWS,
    int total_guesses,
    int current_round,
    std::vector<float>& out_cs1,
    std::vector<float>& out_ay);

namespace Consts {
constexpr int32_t Q = 8380417;
constexpr int N_ROUNDS = 256;
}

int32_t normalize_mod_q_host(int32_t x) {
    int64_t v = static_cast<int64_t>(x) % Consts::Q;
    if (v < 0) v += Consts::Q;
    return static_cast<int32_t>(v);
}

struct AttackConfig {
    std::string data_dir;
    std::string opt_level;
    std::string sampling_tag;
    std::string results_path;
    int instances = 1;
    int mode = 44;
    int threads = 1;
    int start = 1;
    int end = 1;
    int step = 1;
    int max_k = 8;
};

template <typename T>
bool load_binary(const std::string& path, std::vector<T>& data) {
    namespace fs = std::filesystem;

    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return false;

    const std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    if (sz <= 0 || sz % static_cast<std::streamsize>(sizeof(T)) != 0) return false;

    data.resize(static_cast<size_t>(sz) / sizeof(T));
    const bool success = f.read(reinterpret_cast<char*>(data.data()), sz).good();

    if (success) {
        std::error_code ec;
        fs::path p(path);
        const uintmax_t fsize = fs::file_size(p, ec);
        std::cout << "[FILE LOADED] " << p.filename().string()
                  << " (" << std::fixed << std::setprecision(2)
                  << (fsize / (1024.0 * 1024.0)) << " MB)\n";
    }

    return success;
}

std::string parse_opt_level(const std::string& input) {
    if (input == "o3") return "o3_time_opt";
    if (input == "o0") return "o0_no_opt";
    return input;
}

std::string parse_sampling_tag(std::string input) {
    if (input.find("MHz") == std::string::npos) input += "MHz";
    return input;
}

std::filesystem::path data_path(const std::string& base, const std::string& name) {
    return std::filesystem::path(base) / name;
}

void ensure_parent_dir(const std::string& path) {
    const std::filesystem::path p(path);
    if (!p.parent_path().empty()) {
        std::filesystem::create_directories(p.parent_path());
    }
}

std::string unique_filename(const std::string& csv_path) {
    namespace fs = std::filesystem;
    fs::path p(csv_path);
    if (!p.parent_path().empty()) fs::create_directories(p.parent_path());
    if (!fs::exists(p)) return p.string();

    const std::string stem = p.stem().string();
    const std::string ext = p.extension().string();
    for (int cnt = 1;; ++cnt) {
        fs::path candidate = p.parent_path() / (stem + "_" + std::to_string(cnt) + ext);
        if (!fs::exists(candidate)) return candidate.string();
    }
}

std::string find_file_with_pattern(const std::string& directory, const std::string& prefix, const std::string& suffix) {
    namespace fs = std::filesystem;
    if (!fs::exists(directory)) return "";

    std::vector<std::string> matches;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        const std::string filename = entry.path().filename().string();
        if (filename.size() < prefix.size() + suffix.size()) continue;
        if (filename.compare(0, prefix.size(), prefix) == 0 &&
            filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) == 0) {
            matches.push_back(entry.path().string());
        }
    }
    std::sort(matches.begin(), matches.end());
    return matches.empty() ? "" : matches.front();
}

int parse_number_before(const std::string& filename, size_t pos) {
    size_t start = pos;
    while (start > 0 && std::isdigit(static_cast<unsigned char>(filename[start - 1]))) {
        --start;
    }
    if (start == pos) return -1;
    return std::stoi(filename.substr(start, pos - start));
}

int extract_trace_count(const std::string& filepath) {
    const std::string filename = std::filesystem::path(filepath).filename().string();

    size_t pos = filename.find("tr_");
    if (pos != std::string::npos) {
        const int n = parse_number_before(filename, pos);
        if (n > 0) return n;
    }

    pos = filename.find("k_");
    if (pos != std::string::npos) {
        const int n = parse_number_before(filename, pos);
        if (n > 0) return n * 1000;
    }

    pos = filename.find('e');
    while (pos != std::string::npos) {
        if (pos > 0 && pos + 1 < filename.size() &&
            std::isdigit(static_cast<unsigned char>(filename[pos - 1])) &&
            std::isdigit(static_cast<unsigned char>(filename[pos + 1]))) {
            const int base = parse_number_before(filename, pos);
            size_t exp_start = pos + 1;
            size_t exp_end = exp_start;
            while (exp_end < filename.size() &&
                   std::isdigit(static_cast<unsigned char>(filename[exp_end]))) {
                ++exp_end;
            }
            const int exp = std::stoi(filename.substr(exp_start, exp_end - exp_start));
            int mult = 1;
            for (int i = 0; i < exp; ++i) mult *= 10;
            return base * mult;
        }
        pos = filename.find('e', pos + 1);
    }

    throw std::runtime_error("Could not parse trace count from filename: " + filename);
}

template <typename T>
int checked_window_len(const std::vector<T>& data, int traces, int units_per_trace, const std::string& label) {
    const size_t denom = static_cast<size_t>(traces) * static_cast<size_t>(units_per_trace);
    if (traces <= 0 || denom == 0 || data.size() % denom != 0) {
        throw std::runtime_error("Bad dimensions for " + label);
    }
    return static_cast<int>(data.size() / denom);
}

float fisher_z(float rho) {
    return std::atanh(std::clamp(rho, -0.9999f, 0.9999f));
}

struct FusionEval {
    float ld = -std::numeric_limits<float>::infinity();
    int32_t best_g = 0;
    float alpha_ay = 0.0f;
    std::vector<int32_t> topK;
    std::vector<float> topScores;
};

FusionEval search_ld_fusion(
    const std::vector<float>& z_ay,
    const std::vector<float>& z_cs1,
    int max_k,
    int grid = 20,
    int ld_m = 4) {
    FusionEval best;

    #pragma omp parallel
    {
        FusionEval local;

        #pragma omp for schedule(dynamic)
        for (int a = 0; a <= grid; ++a) {
            const float alpha = static_cast<float>(a) / static_cast<float>(grid);
            std::vector<float> top_scores(max_k, -std::numeric_limits<float>::infinity());
            std::vector<int32_t> top_guesses(max_k, 0);
            double sum = 0.0;
            double sum_sq = 0.0;

            for (int g = 0; g < Consts::Q; ++g) {
                const float score = alpha * z_ay[g] + (1.0f - alpha) * z_cs1[g];
                sum += score;
                sum_sq += static_cast<double>(score) * score;

                if (score > top_scores[max_k - 1]) {
                    top_scores[max_k - 1] = score;
                    top_guesses[max_k - 1] = g;
                    for (int j = max_k - 1; j > 0; --j) {
                        if (top_scores[j] > top_scores[j - 1]) {
                            std::swap(top_scores[j], top_scores[j - 1]);
                            std::swap(top_guesses[j], top_guesses[j - 1]);
                        } else {
                            break;
                        }
                    }
                }
            }

            const double mean = sum / Consts::Q;
            const float variance = static_cast<float>((sum_sq / Consts::Q) - mean * mean);
            const float std_dev = std::sqrt(std::max(variance, 1e-12f));

            const int runners = std::min(ld_m, max_k - 1);
            float ld = 0.0f;
            for (int m = 1; m <= runners; ++m) {
                ld += ((runners + 1.0f - m) / (runners + 1.0f)) *
                      (top_scores[0] - top_scores[m]) /
                      (1.41421356f * std_dev);
            }

            if (ld > local.ld) {
                local.ld = ld;
                local.best_g = top_guesses[0];
                local.alpha_ay = alpha;
                local.topK = std::move(top_guesses);
                local.topScores = std::move(top_scores);
            }
        }

        #pragma omp critical
        {
            if (local.ld > best.ld) best = std::move(local);
        }
    }

    return best;
}

int count_matches_ntt(const std::vector<int32_t>& truth, const std::array<int32_t, 256>& candidate) {
    int matches = 0;
    for (int i = 0; i < Consts::N_ROUNDS; ++i) {
        const int32_t true_s1 = normalize_mod_q_host(truth[i]);
        if (candidate[i] == true_s1) ++matches;
    }
    return matches;
}

int main(int argc, char* argv[]) {
    using Clock = std::chrono::high_resolution_clock;

    try {
        if (argc < 11) {
            throw std::runtime_error(
                "Usage: ./fusion_unprotected_artifact <data_dir> <instances> <mode:44|65|87> "
                "<opt:o0|o3> <sampling:25|100> <threads> <start> <end> <step> <max_k> [results_csv]");
        }

        AttackConfig cfg;
        cfg.data_dir = argv[1];
        cfg.instances = std::stoi(argv[2]);
        cfg.mode = std::stoi(argv[3]);
        cfg.opt_level = parse_opt_level(argv[4]);
        cfg.sampling_tag = parse_sampling_tag(argv[5]);
        cfg.threads = std::stoi(argv[6]);
        cfg.start = std::stoi(argv[7]);
        cfg.end = std::stoi(argv[8]);
        cfg.step = std::stoi(argv[9]);
        cfg.max_k = std::stoi(argv[10]);
        cfg.results_path = (argc >= 12)
            ? argv[11]
            : "results/unprotected_mldsa_" + std::to_string(cfg.mode) + "_" +
              cfg.opt_level + "_" + cfg.sampling_tag + ".csv";

        if (cfg.instances <= 0) throw std::runtime_error("instances must be positive");
        if (cfg.step <= 0) throw std::runtime_error("step must be positive");
        if (cfg.max_k < 5) throw std::runtime_error("max_k must be at least 5");

        ensure_parent_dir(cfg.results_path);
        cfg.results_path = unique_filename(cfg.results_path);
        omp_set_num_threads(cfg.threads);

        const int eta_bound = (cfg.mode == 65) ? 4 : 2;
        const int K_ROWS = (cfg.mode == 44) ? 4 : (cfg.mode == 65) ? 6 : 8;
        const std::string suffix = cfg.opt_level + ".bin";

        std::cout << "Mode=ML-DSA-" << cfg.mode
                  << " Opt=" << cfg.opt_level
                  << " Sampling=" << cfg.sampling_tag
                  << " K_ROWS=" << K_ROWS
                  << " Results=" << cfg.results_path << "\n";

        std::vector<int32_t> s1_true;
        if (!load_binary(data_path(cfg.data_dir, "poly0_s1_ntt_true.bin").string(), s1_true) &&
            !load_binary(data_path(cfg.data_dir, "ay_poly0_s1_ntt_true.bin").string(), s1_true)) {
            throw std::runtime_error("Failed to load secret-key ground truth");
        }

        std::vector<int32_t> cs1_c_labels;
        if (!load_binary(data_path(cfg.data_dir, "poly0_c_ntt_labels.bin").string(), cs1_c_labels)) {
            throw std::runtime_error("Failed to load poly0_c_ntt_labels.bin");
        }

        const std::string cs1_file = find_file_with_pattern(
            cfg.data_dir,
            "poly0_cs1_" + cfg.sampling_tag + "_",
            suffix);
        if (cs1_file.empty()) throw std::runtime_error("Could not find CS1 trace file");

        std::vector<float> cs1_traces_vec;
        if (!load_binary(cs1_file, cs1_traces_vec)) throw std::runtime_error("Failed to load CS1 trace file");

        const int cs1_traces_count = extract_trace_count(cs1_file);
        const int T_cs1 = checked_window_len(cs1_traces_vec, cs1_traces_count, Consts::N_ROUNDS, "CS1");

        std::vector<int32_t> ay_c_ntt, ay_z_ntt, ay_a_matrix;
        if (!load_binary(data_path(cfg.data_dir, "ay_poly0_c_ntt.bin").string(), ay_c_ntt)) {
            throw std::runtime_error("Failed to load ay_poly0_c_ntt.bin");
        }
        if (!load_binary(data_path(cfg.data_dir, "ay_poly0_z_ntt.bin").string(), ay_z_ntt)) {
            throw std::runtime_error("Failed to load ay_poly0_z_ntt.bin");
        }
        if (!load_binary(data_path(cfg.data_dir, "ay_poly0_A_matrix.bin").string(), ay_a_matrix)) {
            throw std::runtime_error("Failed to load ay_poly0_A_matrix.bin");
        }

        const std::string ay_file = find_file_with_pattern(
            cfg.data_dir,
            "ay_poly0_" + cfg.sampling_tag + "_",
            suffix);
        if (ay_file.empty()) throw std::runtime_error("Could not find AY trace file");

        std::vector<float> ay_traces_vec;
        if (!load_binary(ay_file, ay_traces_vec)) throw std::runtime_error("Failed to load AY trace file");

        const int ay_traces_count = extract_trace_count(ay_file);
        const int T_ay = checked_window_len(
            ay_traces_vec,
            ay_traces_count,
            K_ROWS * Consts::N_ROUNDS,
            "AY");

        const int physical_traces = std::min(cs1_traces_count, ay_traces_count);
        if (cfg.instances * cfg.end > physical_traces) {
            throw std::runtime_error("Not enough traces for requested instances and end value");
        }

        std::cout << "TraceCount: CS1=" << cs1_traces_count
                  << " AY=" << ay_traces_count
                  << " Requested=" << (cfg.instances * cfg.end)
                  << " (" << cfg.instances << " instances x " << cfg.end << " traces)\n";

        std::vector<int> trace_steps;
        for (int req = cfg.start; req <= cfg.end; req += cfg.step) trace_steps.push_back(req);

        std::vector<std::vector<int>> inst_chunked_idxs(cfg.instances);
        for (int inst = 0; inst < cfg.instances; ++inst) {
            const int start_idx = inst * cfg.end;
            for (int i = 0; i < cfg.end; ++i) {
                inst_chunked_idxs[inst].push_back(start_idx + i);
            }
        }

        std::ofstream res_file(cfg.results_path);
        if (!res_file.is_open()) throw std::runtime_error("Failed to open results CSV");
        res_file << "instance,num_traces,Succ_cs1,Succ_ay,Succ_fusion,Succ_fusion_sieve,"
                 << "time_score_s,time_fusion_s,time_sieve_s,time_total_s\n";
        res_file.flush();

        const auto global_start = Clock::now();

        for (int inst = 0; inst < cfg.instances; ++inst) {
            const auto& idxs = inst_chunked_idxs[inst];
            const int max_N = static_cast<int>(idxs.size());

            std::vector<float> inst_cs1_traces(static_cast<size_t>(max_N) * Consts::N_ROUNDS * T_cs1);
            std::vector<int32_t> inst_cs1_labels(static_cast<size_t>(max_N) * Consts::N_ROUNDS);
            std::vector<float> inst_ay_traces(static_cast<size_t>(max_N) * K_ROWS * Consts::N_ROUNDS * T_ay);
            std::vector<int32_t> inst_ay_c_ntt(static_cast<size_t>(max_N) * Consts::N_ROUNDS);
            std::vector<int32_t> inst_ay_z_ntt(static_cast<size_t>(max_N) * Consts::N_ROUNDS);

            for (int i = 0; i < max_N; ++i) {
                const int original_idx = idxs[i];
                for (int r = 0; r < Consts::N_ROUNDS; ++r) {
                    inst_cs1_labels[static_cast<size_t>(i) * Consts::N_ROUNDS + r] =
                        cs1_c_labels[static_cast<size_t>(original_idx) * Consts::N_ROUNDS + r];

                    for (int t = 0; t < T_cs1; ++t) {
                        inst_cs1_traces[static_cast<size_t>(i) * Consts::N_ROUNDS * T_cs1 + r * T_cs1 + t] =
                            cs1_traces_vec[static_cast<size_t>(original_idx) * Consts::N_ROUNDS * T_cs1 + r * T_cs1 + t];
                    }

                    inst_ay_c_ntt[static_cast<size_t>(i) * Consts::N_ROUNDS + r] =
                        ay_c_ntt[static_cast<size_t>(original_idx) * Consts::N_ROUNDS + r];
                    inst_ay_z_ntt[static_cast<size_t>(i) * Consts::N_ROUNDS + r] =
                        ay_z_ntt[static_cast<size_t>(original_idx) * Consts::N_ROUNDS + r];

                    for (int k = 0; k < K_ROWS; ++k) {
                        for (int t = 0; t < T_ay; ++t) {
                            const size_t src_idx =
                                static_cast<size_t>(original_idx) * (K_ROWS * Consts::N_ROUNDS) * T_ay +
                                (k * Consts::N_ROUNDS + r) * T_ay + t;
                            const size_t dst_idx =
                                static_cast<size_t>(i) * (K_ROWS * Consts::N_ROUNDS) * T_ay +
                                (k * Consts::N_ROUNDS + r) * T_ay + t;
                            inst_ay_traces[dst_idx] = ay_traces_vec[src_idx];
                        }
                    }
                }
            }

            bool is_new_instance = true;

            for (int req_traces : trace_steps) {
                std::cout << "[STEP BEGIN] inst=" << (inst + 1)
                          << "/" << cfg.instances
                          << " traces=" << req_traces << "\n";

                const auto step_start = Clock::now();
                double time_score_s = 0.0;
                double time_fusion_s = 0.0;
                double time_sieve_s = 0.0;

                int s_cs1 = 0;
                int s_ay = 0;
                int s_fusion = 0;

                std::vector<std::vector<float>> score_ld(
                    Consts::N_ROUNDS,
                    std::vector<float>(cfg.max_k, -std::numeric_limits<float>::infinity()));
                std::vector<std::vector<int32_t>> topK_ld(
                    Consts::N_ROUNDS,
                    std::vector<int32_t>(cfg.max_k, 0));

                for (int r = 0; r < Consts::N_ROUNDS; ++r) {
                    std::vector<int32_t> inst_a_col(K_ROWS);
                    for (int k = 0; k < K_ROWS; ++k) {
                        inst_a_col[k] = ay_a_matrix[k * Consts::N_ROUNDS + r];
                    }

                    const int32_t true_s1 = normalize_mod_q_host(s1_true[r]);
                    std::vector<float> sc_cs1(Consts::Q, -1.0f);
                    std::vector<float> sc_ay(Consts::Q, -1.0f);

                    const auto score_start = Clock::now();
                    gpu_solve_round_unmasked(
                        is_new_instance,
                        inst_cs1_traces,
                        inst_cs1_labels,
                        req_traces,
                        max_N,
                        T_cs1,
                        inst_ay_traces,
                        inst_ay_c_ntt,
                        inst_ay_z_ntt,
                        inst_a_col,
                        req_traces,
                        max_N,
                        T_ay,
                        K_ROWS,
                        Consts::Q,
                        r,
                        sc_cs1,
                        sc_ay);
                    is_new_instance = false;
                    time_score_s += std::chrono::duration<double>(Clock::now() - score_start).count();

                    const int32_t b_cs1 = static_cast<int32_t>(
                        std::distance(sc_cs1.begin(), std::max_element(sc_cs1.begin(), sc_cs1.end())));
                    if (b_cs1 == true_s1 || b_cs1 == Consts::Q - true_s1) ++s_cs1;

                    const int32_t b_ay = static_cast<int32_t>(
                        std::distance(sc_ay.begin(), std::max_element(sc_ay.begin(), sc_ay.end())));
                    if (b_ay == true_s1) ++s_ay;

                    const auto fusion_start = Clock::now();
                    std::vector<float> z_ay(Consts::Q);
                    std::vector<float> z_cs1(Consts::Q);
                    #pragma omp parallel for
                    for (int g = 0; g < Consts::Q; ++g) {
                        z_ay[g] = fisher_z(sc_ay[g]);
                        z_cs1[g] = fisher_z(sc_cs1[g]);
                    }

                    FusionEval chosen = search_ld_fusion(z_ay, z_cs1, cfg.max_k);
                    time_fusion_s += std::chrono::duration<double>(Clock::now() - fusion_start).count();

                    if (chosen.best_g == true_s1) ++s_fusion;
                    topK_ld[r] = std::move(chosen.topK);
                    score_ld[r] = std::move(chosen.topScores);
                }

                const auto sieve_start = Clock::now();
                const auto sieve_res = mldsa_sieve::EnumerateLdAStarKey(score_ld, topK_ld, 1000000LL, eta_bound);
                int s_fusion_sieve = s_fusion;
                if (sieve_res.found) {
                    s_fusion_sieve = count_matches_ntt(s1_true, sieve_res.found_ntt);
                }
                time_sieve_s = std::chrono::duration<double>(Clock::now() - sieve_start).count();

                const double time_total_s = std::chrono::duration<double>(Clock::now() - step_start).count();

                res_file << (inst + 1) << ","
                         << req_traces << ","
                         << s_cs1 << ","
                         << s_ay << ","
                         << s_fusion << ","
                         << s_fusion_sieve << ","
                         << std::fixed << std::setprecision(4)
                         << time_score_s << ","
                         << time_fusion_s << ","
                         << time_sieve_s << ","
                         << time_total_s << "\n";
                res_file.flush();

                std::cout << "inst=" << (inst + 1)
                          << " traces=" << req_traces
                          << " cs1=" << s_cs1
                          << " ay=" << s_ay
                          << " fusion=" << s_fusion
                          << " sieve=" << s_fusion_sieve
                          << " time=" << std::fixed << std::setprecision(2)
                          << time_total_s << "s\n";
            }
        }

        std::cout << "Done in "
                  << std::chrono::duration<double>(Clock::now() - global_start).count()
                  << "s\n";
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
