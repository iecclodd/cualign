#include "CuAlign.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>

namespace cualign {
namespace {
constexpr uint32_t none = std::numeric_limits<uint32_t>::max();

void validate(const CuAlignProblem& p) {
    const size_t n = p.source.size();
    if (p.target.size() != n || p.unary.size() != n || p.sourceOffsets.empty() ||
        p.targetOffsets.empty() || p.factorOffsets.size() != n + 1 ||
        p.reverse.size() != p.arcMultiplicity.size() || p.targetCandidates.size() != n)
        throw std::invalid_argument("cuAlign: inconsistent problem dimensions");
    auto offsets = [](const std::vector<uint32_t>& a, size_t end) {
        return !a.empty() && a.front() == 0 && a.back() == end &&
            std::is_sorted(a.begin(), a.end());
    };
    if (!offsets(p.sourceOffsets, n) || !offsets(p.targetOffsets, n) ||
        !offsets(p.factorOffsets, p.reverse.size()))
        throw std::invalid_argument("cuAlign: invalid CSR offsets");
    std::vector<uint8_t> seen(n, 0);
    for (size_t s = 0; s + 1 < p.sourceOffsets.size(); ++s)
        for (uint32_t i = p.sourceOffsets[s]; i < p.sourceOffsets[s+1]; ++i)
            if (p.source[i] != s || p.target[i] >= p.targetOffsets.size()-1 || !std::isfinite(p.unary[i]))
                throw std::invalid_argument("cuAlign: invalid candidate");
    for (size_t t = 0; t + 1 < p.targetOffsets.size(); ++t)
        for (uint32_t j = p.targetOffsets[t]; j < p.targetOffsets[t+1]; ++j) {
            uint32_t i = p.targetCandidates[j];
            if (i >= n || seen[i]++ || p.target[i] != t)
                throw std::invalid_argument("cuAlign: invalid target candidate index");
        }
    for (size_t i = 0; i < p.reverse.size(); ++i)
        if (p.reverse[i] >= p.reverse.size() || p.reverse[p.reverse[i]] != i ||
            p.arcMultiplicity[i] == 0 || p.arcMultiplicity[i] != p.arcMultiplicity[p.reverse[i]])
            throw std::invalid_argument("cuAlign: invalid reverse factor arc");
}

struct TopTwo {
    double first = 0.0, second = 0.0;
    uint32_t firstId = none;
    void add(double value, uint32_t id) {
        if (value > first) { second = first; first = value; firstId = id; }
        else if (value > second) second = value;
    }
    double other(uint32_t id) const { return id == firstId ? second : first; }
};

class CpuStep final : public CuAlignStep {
    const CuAlignProblem& p_;
    double beta_, damping_, mix_;
    std::vector<double> yp_, zp_, sp_;
public:
    CpuStep(const CuAlignProblem& p, double beta, double damping)
        : p_(p), beta_(beta), damping_(damping), mix_(damping), yp_(p.source.size(), 0),
          zp_(p.source.size(), 0), sp_(p.reverse.size(), 0) {
        validate(p_);
        if (!std::isfinite(beta) || beta < 0 || !std::isfinite(damping) || damping < 0 || damping > 1)
            throw std::invalid_argument("cuAlign: invalid beta or damping");
    }
    void advance(std::vector<double>& y, std::vector<double>& z) override {
        const size_t n = p_.source.size();
        std::vector<double> f(sp_.size()), d(p_.unary), yc(n), zc(n);
        for (size_t i = 0; i < n; ++i)
            for (uint32_t a = p_.factorOffsets[i]; a < p_.factorOffsets[i+1]; ++a) {
                const double cap = beta_ * p_.arcMultiplicity[a];
                f[a] = std::max(0.0, std::min(cap, cap + sp_[p_.reverse[a]]));
                d[i] += f[a];
            }
        std::vector<TopTwo> row(p_.sourceOffsets.size()-1), col(p_.targetOffsets.size()-1);
        for (uint32_t i = 0; i < n; ++i) {
            row[p_.source[i]].add(yp_[i], i);
            col[p_.target[i]].add(zp_[i], i);
        }
        for (uint32_t i = 0; i < n; ++i) {
            yc[i] = d[i] - col[p_.target[i]].other(i);
            zc[i] = d[i] - row[p_.source[i]].other(i);
        }
        for (size_t i = 0; i < n; ++i)
            for (uint32_t a = p_.factorOffsets[i]; a < p_.factorOffsets[i+1]; ++a)
                sp_[a] = mix_ * (yc[i] + zc[i] - d[i] - f[a]) + (1.0 - mix_) * sp_[a];
        for (size_t i = 0; i < n; ++i) {
            yp_[i] = mix_ * yc[i] + (1.0 - mix_) * yp_[i];
            zp_[i] = mix_ * zc[i] + (1.0 - mix_) * zp_[i];
        }
        mix_ *= damping_;
        y = std::move(yc);
        z = std::move(zc);
    }
};
} // namespace

std::unique_ptr<CuAlignStep> makeCuAlignCpu(const CuAlignProblem& p, double beta, double damping) {
    return std::unique_ptr<CuAlignStep>(new CpuStep(p, beta, damping));
}

std::vector<uint32_t> completeCuAlign(const CuAlignProblem& p, const std::vector<double>& weights,
                                      std::vector<uint32_t> match) {
    const size_t n = p.source.size();
    if (p.sourceOffsets.empty() || p.targetOffsets.empty() || p.target.size() != n ||
        weights.size() != n || p.sourceOffsets.back() != n ||
        !std::all_of(weights.begin(), weights.end(), [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("cuAlign: invalid rounding dimensions or weights");
    const size_t ns = p.sourceOffsets.size()-1, nt = p.targetOffsets.size()-1;
    if (match.size() != ns) throw std::invalid_argument("cuAlign: invalid partial matching size");
    if (ns > nt) throw std::runtime_error("cuAlign: candidate support has no full injection");
    std::vector<uint32_t> owner(nt, none);
    bool incomplete = false;
    for (size_t s = 0; s < ns; ++s) {
        const uint32_t t = match[s];
        if (t == none) { incomplete = true; continue; }
        if (t >= nt || owner[t] != none)
            throw std::invalid_argument("cuAlign: partial matching is not injective");
        bool supported = false;
        for (uint32_t i = p.sourceOffsets[s]; i < p.sourceOffsets[s+1]; ++i)
            if (p.target[i] == t) { supported = true; break; }
        if (!supported) throw std::invalid_argument("cuAlign: partial matching contains an unsupported edge");
        owner[t] = static_cast<uint32_t>(s);
    }
    if (!incomplete) return match;
    std::vector<std::vector<uint32_t>> bySource(ns);
    for (size_t s = 0; s < ns; ++s) {
        auto& ids = bySource[s];
        for (uint32_t i = p.sourceOffsets[s]; i < p.sourceOffsets[s+1]; ++i) ids.push_back(i);
        std::sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b) {
            return weights[a] != weights[b] ? weights[a] > weights[b] : a < b;
        });
    }
    std::vector<uint64_t> seenSource(ns, 0), seenTarget(nt, 0);
    std::vector<uint32_t> parentTarget(nt, none);
    uint64_t stamp = 0;
    for (uint32_t root = 0; root < ns; ++root) {
        if (match[root] != none) continue;
        ++stamp;
        std::queue<uint32_t> q;
        q.push(root); seenSource[root] = stamp;
        uint32_t freeTarget = none;
        while (!q.empty() && freeTarget == none) {
            uint32_t s = q.front(); q.pop();
            for (uint32_t i : bySource[s]) {
                uint32_t t = p.target[i];
                if (seenTarget[t] == stamp) continue;
                seenTarget[t] = stamp; parentTarget[t] = s;
                if (owner[t] == none) { freeTarget = t; break; }
                uint32_t next = owner[t];
                if (seenSource[next] != stamp) { seenSource[next] = stamp; q.push(next); }
            }
        }
        if (freeTarget == none) throw std::runtime_error("cuAlign: candidate support has no full injection");
        for (uint32_t t = freeTarget; t != none;) {
            uint32_t s = parentTarget[t], previous = match[s];
            match[s] = t; owner[t] = s; t = previous;
        }
    }
    return match;
}

std::vector<uint32_t> roundCuAlign(const CuAlignProblem& p, const std::vector<double>& weights) {
    const size_t n = p.source.size();
    if (p.sourceOffsets.empty() || p.targetOffsets.empty() || p.target.size() != n ||
        weights.size() != n || p.sourceOffsets.back() != n ||
        !std::all_of(weights.begin(), weights.end(), [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("cuAlign: invalid rounding dimensions or weights");
    const size_t ns = p.sourceOffsets.size()-1, nt = p.targetOffsets.size()-1;
    if (ns > nt) throw std::runtime_error("cuAlign: candidate support has no full injection");
    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return weights[a] != weights[b] ? weights[a] > weights[b] : a < b;
    });
    std::vector<uint32_t> match(ns, none), owner(nt, none);
    for (uint32_t i : order) {
        if (weights[i] <= 0) break;
        uint32_t s = p.source[i], t = p.target[i];
        if (match[s] == none && owner[t] == none) { match[s] = t; owner[t] = s; }
    }
    return completeCuAlign(p, weights, std::move(match));
}
} // namespace cualign
