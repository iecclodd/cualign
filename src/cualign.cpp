#include "CuAlign.hpp"

#include "Graph.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;
using Mapping = std::vector<uint32_t>;
using CandidateRows = std::vector<std::vector<uint32_t>>;
const uint32_t unmatched = std::numeric_limits<uint32_t>::max();

struct Options {
    std::string source, target, candidates, start, output;
    std::string backend = "cpu";
    unsigned iterations = 10;
    unsigned maxFactors = 10000000;
    double damping = 0.99;
};

void usage() {
    std::cout << "Usage: cualign --g1 SOURCE.gw --g2 TARGET.gw "
                 "--candidates PARTNERS --output NEW.align [--start START.align]\n"
                 "       [--backend cpu|cuda|verify] [--iterations 10] "
                 "[--damping 0.99] [--max-factors 10000000]\n";
}

unsigned readUnsigned(const std::string& value, const std::string& flag, unsigned max) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Expected an integer for " + flag);
    const auto n = std::stoull(value);
    if (n > max) throw std::runtime_error("Value too large for " + flag);
    return static_cast<unsigned>(n);
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (++i == argc) throw std::runtime_error("Missing value for " + flag);
        const std::string value = argv[i];
        if (flag == "--g1") options.source = value;
        else if (flag == "--g2") options.target = value;
        else if (flag == "--candidates") options.candidates = value;
        else if (flag == "--start") options.start = value;
        else if (flag == "--output") options.output = value;
        else if (flag == "--backend") options.backend = value;
        else if (flag == "--iterations") options.iterations = readUnsigned(value, flag, 10000);
        else if (flag == "--max-factors") options.maxFactors = readUnsigned(value, flag, unmatched / 2);
        else if (flag == "--damping") {
            size_t length = 0;
            options.damping = std::stod(value, &length);
            if (length != value.size() || !(options.damping > 0 && options.damping <= 1))
                throw std::runtime_error("--damping must be in (0, 1]");
        } else {
            throw std::runtime_error("Unknown option " + flag);
        }
    }
    if (options.source.empty() || options.target.empty() ||
        options.candidates.empty() || options.output.empty()) {
        throw std::runtime_error("--g1, --g2, --candidates and --output are required");
    }
    if (options.backend != "cpu" && options.backend != "cuda" && options.backend != "verify")
        throw std::runtime_error("Backend must be cpu, cuda or verify");
#ifndef CUALIGN_CUDA
    if (options.backend != "cpu") throw std::runtime_error("Use cualign.cuda for this backend");
#endif
    return options;
}

Graph readGw(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open graph " + path);
    std::string line;
    const auto nextLine = [&]() {
        if (!std::getline(file, line)) throw std::runtime_error("Unexpected end of GW graph " + path);
        if (!line.empty() && line.back() == '\r') line.pop_back();
    };
    nextLine();
    if (line != "LEDA.GRAPH") throw std::runtime_error("Expected a plain LEDA GW graph: " + path);
    nextLine(); // Node label type
    nextLine(); // Edge label type
    nextLine();
    if (line != "-1" && line != "-2") throw std::runtime_error("Invalid GW direction field: " + path);
    nextLine();
    const auto nodes = readUnsigned(line, "GW node count", unmatched - 1);
    if (!nodes) throw std::runtime_error("Empty GW node set: " + path);
    std::vector<std::string> names;
    names.reserve(nodes);
    for (unsigned i = 0; i < nodes; ++i) {
        nextLine();
        if (line.size() < 5 || line.compare(0, 2, "|{") || line.compare(line.size() - 2, 2, "}|"))
            throw std::runtime_error("Invalid GW node label: " + path);
        auto label = line.substr(2, line.size() - 4);
        if (label.find_first_of(" \t\r\n\f\v") != std::string::npos ||
            label.find('\0') != std::string::npos)
            throw std::runtime_error("GW node labels must be single tokens: " + path);
        names.push_back(std::move(label));
    }
    nextLine();
    const auto count = readUnsigned(line, "GW arc count", unmatched - 1);
    std::vector<std::array<uint32_t, 2>> edges;
    edges.reserve(count);
    for (unsigned i = 0; i < count; ++i) {
        nextLine();
        const char* cursor = line.c_str();
        const auto endpoint = [&]() {
            cursor += std::strspn(cursor, " \t");
            if (*cursor < '0' || *cursor > '9')
                throw std::runtime_error("Invalid GW arc endpoint: " + path);
            char* end = nullptr;
            const auto value = std::strtoul(cursor, &end, 10);
            if (!value || value > nodes || (*end && *end != ' ' && *end != '\t'))
                throw std::runtime_error("Invalid GW arc endpoint: " + path);
            cursor = end;
            return static_cast<uint32_t>(value - 1);
        };
        const auto u = endpoint();
        const auto v = endpoint();
        edges.push_back({u, v});
    }
    while (std::getline(file, line))
        if (line.find_first_not_of(" \t\r") != std::string::npos)
            throw std::runtime_error("Unexpected content after GW arcs: " + path);
    if (file.bad()) throw std::runtime_error("Error reading GW graph " + path);
    return Graph(std::move(edges), std::move(names));
}

uint32_t nodeId(const Graph& graph, const std::string& name, const std::string& path, size_t line) {
    if (!graph.hasNodeName(name))
        throw std::runtime_error(path + ":" + std::to_string(line) + ": unknown node " + name);
    return graph.getNameIndex(name);
}

Mapping readStart(const Graph& source, const Graph& target, const std::string& path) {
    if (path.empty()) return {};
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open starting alignment " + path);
    Mapping mapping(source.getNumNodes(), unmatched);
    std::vector<bool> used(target.getNumNodes(), false);
    std::string line;
    size_t lineNumber = 0, count = 0;
    while (std::getline(file, line)) {
        ++lineNumber;
        std::istringstream words(line);
        std::string a, b, extra;
        if (!(words >> a)) continue;
        if (!(words >> b) || (words >> extra))
            throw std::runtime_error("Expected two alignment labels on line " + std::to_string(lineNumber));
        const auto u = nodeId(source, a, path, lineNumber);
        const auto v = nodeId(target, b, path, lineNumber);
        if (mapping[u] != unmatched || used[v])
            throw std::runtime_error("Starting alignment is not injective on line " + std::to_string(lineNumber));
        mapping[u] = v;
        used[v] = true;
        ++count;
    }
    if (file.bad()) throw std::runtime_error("Error reading starting alignment " + path);
    if (count != mapping.size()) throw std::runtime_error("Starting alignment must cover every source node");
    return mapping;
}

CandidateRows readCandidates(const Graph& source, const Graph& target, const std::string& path,
                             const Mapping& start) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open candidate file " + path);
    CandidateRows rows(source.getNumNodes());
    std::string line;
    size_t lineNumber = 0;
    while (std::getline(file, line)) {
        ++lineNumber;
        std::istringstream words(line);
        std::string a, b;
        if (!(words >> a)) continue;
        auto& row = rows[nodeId(source, a, path, lineNumber)];
        if (!(words >> b)) throw std::runtime_error("No target on candidate line " + std::to_string(lineNumber));
        do {
            row.push_back(nodeId(target, b, path, lineNumber));
        } while (words >> b);
    }
    if (file.bad()) throw std::runtime_error("Error reading candidate file " + path);
    for (size_t u = 0; u < rows.size(); ++u) {
        auto& row = rows[u];
        if (!start.empty()) row.push_back(start[u]);
        if (row.empty())
            throw std::runtime_error("Missing candidates for source " + source.getNodeName(u) +
                                     ". Supply a complete candidate set or --start.");
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }
    return rows;
}

cualign::CuAlignProblem buildProblem(const Graph& source, const Graph& target,
                                    const CandidateRows& rows, unsigned maxFactors) {
    cualign::CuAlignProblem problem;
    problem.sourceOffsets.push_back(0);
    problem.targetOffsets.resize(static_cast<size_t>(target.getNumNodes()) + 1, 0);
    for (uint32_t u = 0; u < source.getNumNodes(); ++u) {
        if (rows[u].size() >= unmatched - problem.target.size())
            throw std::runtime_error("Too many candidates for 32-bit sparse indices");
        const bool loop = source.hasEdge(u, u);
        for (const auto v : rows[u]) {
            problem.source.push_back(u);
            problem.target.push_back(v);
            problem.unary.push_back(loop && target.hasEdge(v, v) ? 1.0 : 0.0);
            ++problem.targetOffsets[v + 1];
        }
        problem.sourceOffsets.push_back(static_cast<uint32_t>(problem.target.size()));
    }
    for (size_t i = 1; i < problem.targetOffsets.size(); ++i)
        problem.targetOffsets[i] += problem.targetOffsets[i - 1];
    auto nextTarget = problem.targetOffsets;
    problem.targetCandidates.resize(problem.target.size());
    for (uint32_t i = 0; i < problem.target.size(); ++i)
        problem.targetCandidates[nextTarget[problem.target[i]]++] = i;

    // Encode each conserved directed arc as the unordered pair of candidate IDs.
    // A reciprocal arc produces the same key and raises that factor's weight to 2.
    std::vector<uint64_t> pairKeys;
    const auto overBudget = [] {
        return std::runtime_error("Overlap factor limit exceeded. Use fewer candidates or increase --max-factors after sizing memory.");
    };
    const auto appendFactor = [&](uint32_t i, uint32_t j) {
        if (pairKeys.size() == 2ULL * maxFactors) throw overBudget();
        pairKeys.push_back((static_cast<uint64_t>(std::min(i, j)) << 32) | std::max(i, j));
    };
    for (const auto& arc : *source.getEdgeList()) {
        const uint32_t u = arc[0], w = arc[1];
        if (u == w) continue;
        const auto first = problem.target.begin() + problem.sourceOffsets[w];
        const auto end = problem.target.begin() + problem.sourceOffsets[w + 1];
        for (uint32_t i = problem.sourceOffsets[u]; i < problem.sourceOffsets[u + 1]; ++i) {
            const uint32_t v = problem.target[i];
            const auto& neighbors = *target.getAdjList(v);
            if (neighbors.size() < rows[w].size()) {
                for (const auto x : neighbors) {
                    if (v == x) continue;
                    const auto found = std::lower_bound(first, end, x);
                    if (found != end && *found == x)
                        appendFactor(i, static_cast<uint32_t>(found - problem.target.begin()));
                }
            } else {
                for (auto j = first; j != end; ++j)
                    if (v != *j && target.hasEdge(v, *j))
                        appendFactor(i, static_cast<uint32_t>(j - problem.target.begin()));
            }
        }
    }
    std::sort(pairKeys.begin(), pairKeys.end());
    problem.factorOffsets.resize(problem.target.size() + 1, 0);
    size_t factorCount = 0;
    for (size_t k = 0; k < pairKeys.size(); ++k) {
        if (k && pairKeys[k] == pairKeys[k - 1]) continue;
        if (++factorCount > maxFactors) throw overBudget();
        ++problem.factorOffsets[(pairKeys[k] >> 32) + 1];
        ++problem.factorOffsets[static_cast<uint32_t>(pairKeys[k]) + 1];
    }
    for (size_t i = 1; i < problem.factorOffsets.size(); ++i)
        problem.factorOffsets[i] += problem.factorOffsets[i - 1];
    auto nextFactor = problem.factorOffsets;
    problem.reverse.resize(2 * factorCount);
    problem.arcMultiplicity.resize(2 * factorCount);
    for (size_t k = 0; k < pairKeys.size();) {
        const uint64_t key = pairKeys[k];
        const uint32_t a = nextFactor[key >> 32]++;
        const uint32_t b = nextFactor[static_cast<uint32_t>(key)]++;
        size_t end = k + 1;
        while (end < pairKeys.size() && pairKeys[end] == key) ++end;
        if (end - k > 2) throw std::runtime_error("Expected unique directed graph arcs");
        problem.arcMultiplicity[a] = problem.arcMultiplicity[b] = static_cast<uint8_t>(end - k);
        problem.reverse[a] = b;
        problem.reverse[b] = a;
        k = end;
    }
    return problem;
}

uint64_t score(const Graph& source, const Graph& target, const Mapping& mapping) {
    uint64_t count = 0;
    for (const auto& arc : *source.getEdgeList())
        count += target.hasEdge(mapping[arc[0]], mapping[arc[1]]);
    return count;
}

void save(const Graph& source, const Graph& target, const Mapping& mapping, const std::string& output) {
    // Publish only a complete file and preserve any existing output.
    std::string pattern = output + ".tmp.XXXXXX";
    std::vector<char> temp(pattern.begin(), pattern.end());
    temp.push_back('\0');
    int fd = mkstemp(temp.data());
    if (fd < 0) throw std::runtime_error("Cannot create temporary alignment for " + output);
    auto* file = fdopen(fd, "w");
    if (!file) {
        close(fd);
        unlink(temp.data());
        throw std::runtime_error("Cannot open temporary alignment for " + output);
    }
    try {
        for (uint32_t u = 0; u < mapping.size(); ++u) {
            if (std::fprintf(file, "%s\t%s\n", source.getNodeName(u).c_str(),
                             target.getNodeName(mapping[u]).c_str()) < 0)
                throw std::runtime_error("Error writing alignment " + output);
        }
        if (std::fflush(file)) throw std::runtime_error("Error writing alignment " + output);
        if (link(temp.data(), output.c_str()) != 0)
            throw std::runtime_error("Cannot publish " + output + ": " + std::strerror(errno));
    } catch (...) {
        std::fclose(file);
        unlink(temp.data());
        throw;
    }
    std::fclose(file);
    unlink(temp.data());
}

double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void verifyBeliefs(const std::vector<double>& cpu, const std::vector<double>& cuda) {
    if (cpu.size() != cuda.size()) throw std::runtime_error("CPU/CUDA belief size mismatch");
    for (size_t i = 0; i < cpu.size(); ++i) {
        if (!(std::abs(cpu[i] - cuda[i]) <= 1e-10 * std::max(1.0, std::abs(cpu[i]))))
            throw std::runtime_error("CPU/CUDA belief mismatch at candidate " + std::to_string(i));
    }
}

void run(const Options& options) {
    const auto begin = Clock::now();
    if (access(options.output.c_str(), F_OK) == 0)
        throw std::runtime_error("Output already exists: " + options.output);
    const Graph source = readGw(options.source);
    const Graph target = readGw(options.target);
    if (!source.getNumNodes() || source.getNumNodes() > target.getNumNodes())
        throw std::runtime_error("The source must be nonempty and no larger than the target");
    const auto graphSeconds = seconds(begin);

    const auto prepStart = Clock::now();
    const Mapping start = readStart(source, target, options.start);
    const auto problem = buildProblem(source, target,
        readCandidates(source, target, options.candidates, start), options.maxFactors);
    const auto prepSeconds = seconds(prepStart);
    std::cerr << "cuAlign prepared " << problem.target.size() << " candidates and "
              << problem.reverse.size() / 2 << " overlap factors\n";

    Mapping best = start;
    uint64_t bestScore = start.empty() ? 0 : score(source, target, start);
    const uint64_t startScore = bestScore;
    const auto consider = [&](Mapping candidate) {
        if (candidate == best) return;
        const auto candidateScore = score(source, target, candidate);
        if (best.empty() || candidateScore > bestScore) {
            best = std::move(candidate);
            bestScore = candidateScore;
        }
    };

    const auto setup = Clock::now();
    std::unique_ptr<cualign::CuAlignStep> cpu;
    std::unique_ptr<cualign::CuAlignCudaStep> cuda;
    if (options.iterations && options.backend != "cuda")
        cpu = cualign::makeCuAlignCpu(problem, 1.0, options.damping);
#ifdef CUALIGN_CUDA
    if (options.iterations && options.backend != "cpu")
        cuda = cualign::makeCuAlignCuda(problem, 1.0, options.damping);
#endif
    const auto backendSeconds = seconds(setup);
    double propagationSeconds = 0.0, matchingSeconds = 0.0;
    size_t cpuRepairs = 0;
    std::vector<double> y, z, gpuY, gpuZ;
    const auto roundGpu = [&](bool useZ, std::vector<double>& weights) {
        auto mapping = cuda->roundDevice(useZ);
        if (std::find(mapping.begin(), mapping.end(), unmatched) != mapping.end()) {
            ++cpuRepairs;
            cuda->downloadBeliefs(useZ, weights);
            return cualign::completeCuAlign(problem, weights, std::move(mapping));
        }
        return mapping;
    };
    auto roundStart = Clock::now();
    if (best.empty()) {
        try {
            consider(cualign::roundCuAlign(problem, problem.unary));
        } catch (const std::runtime_error& error) {
            throw std::runtime_error(std::string(error.what()) +
                                     ". Supply a complete candidate set or --start.");
        }
    }
    matchingSeconds += seconds(roundStart);
    for (unsigned step = 0; step < options.iterations; ++step) {
        const auto propagationStart = Clock::now();
        if (cpu) cpu->advance(y, z);
        if (cuda) cuda->advanceDevice();
        if (cpu && cuda) {
            cuda->downloadBeliefs(false, gpuY);
            cuda->downloadBeliefs(true, gpuZ);
            verifyBeliefs(y, gpuY);
            verifyBeliefs(z, gpuZ);
        }
        propagationSeconds += seconds(propagationStart);

        roundStart = Clock::now();
        auto roundY = cuda ? roundGpu(false, gpuY) : cualign::roundCuAlign(problem, y);
        auto roundZ = cuda ? roundGpu(true, gpuZ) : cualign::roundCuAlign(problem, z);
        if (cpu && cuda && (roundY != cualign::roundCuAlign(problem, y) ||
                            roundZ != cualign::roundCuAlign(problem, z))) {
            throw std::runtime_error("CPU/CUDA rounding mismatch at iteration " + std::to_string(step + 1));
        }
        if (roundY == roundZ) consider(std::move(roundY));
        else {
            consider(std::move(roundY));
            consider(std::move(roundZ));
        }
        matchingSeconds += seconds(roundStart);
    }

    save(source, target, best, options.output);
    std::cout << std::setprecision(9)
              << "cualign,backend=" << options.backend
              << ",iterations=" << options.iterations << ",candidates=" << problem.target.size()
              << ",factors=" << problem.reverse.size() / 2
              << ",source_nodes=" << source.getNumNodes() << ",target_nodes=" << target.getNumNodes()
              << ",start_conserved=" << (start.empty() ? "none" : std::to_string(startScore))
              << ",conserved=" << bestScore << ",source_arcs=" << source.getNumEdges()
              << ",graph_s=" << graphSeconds << ",prep_s=" << prepSeconds
              << ",backend_setup_s=" << backendSeconds << ",propagation_s=" << propagationSeconds
              << ",round_and_score_s=" << matchingSeconds << ",cpu_repairs=" << cpuRepairs
              << ",total_s=" << seconds(begin) << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--help")) {
        usage();
        return argc == 1 ? 2 : 0;
    }
    try {
        run(parse(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cualign: " << error.what() << '\n';
        return 1;
    }
}
