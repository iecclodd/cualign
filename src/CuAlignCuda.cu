#include "CuAlign.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cualign {
namespace {

void check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::ostringstream message;
    message << operation << ": " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

template <class T> class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count = 0) : size_(count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::overflow_error("cuAlign device allocation size overflow");
        if (count) check(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "cudaMalloc cuAlign");
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
    void upload(const std::vector<T>& source) {
        if (source.size() != size_) throw std::invalid_argument("cuAlign upload size mismatch");
        if (size_) check(cudaMemcpy(data_, source.data(), size_ * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy cuAlign upload");
    }
    void zero() {
        if (size_) check(cudaMemset(data_, 0, size_ * sizeof(T)), "cudaMemset cuAlign");
    }
    void download(std::vector<T>& destination) const {
        destination.resize(size_);
        if (size_) check(cudaMemcpy(destination.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy cuAlign download");
    }
private:
    T* data_ = nullptr;
    std::size_t size_;
};

struct MaxTwo { double first, second; uint32_t owner; };

// A single thread per group scans once. Keeping the owner of the maximum
// ensures a tied maximum remains available to the excluded candidate.
__global__ void groupMax(const uint32_t* offsets, const uint32_t* ids,
                         const double* values, uint32_t groups, MaxTwo* maxima) {
    const uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) return;
    MaxTwo m{0.0, 0.0, UINT32_MAX};
    for (uint32_t j = offsets[group]; j < offsets[group + 1]; ++j) {
        const uint32_t id = ids ? ids[j] : j;
        const double value = values[id];
        if (value > m.first) {
            m.second = m.first; m.first = value; m.owner = id;
        } else if (value > m.second) {
            m.second = value;
        }
    }
    maxima[group] = m;
}

__global__ void factorValues(const uint32_t* reverse, const uint8_t* multiplicity,
                             const double* previous, uint32_t count, double beta,
                             double* f) {
    const uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= count) return;
    const double limit = beta * multiplicity[e];
    f[e] = fmin(limit, fmax(0.0, limit + previous[reverse[e]]));
}

__global__ void updateCandidates(const uint32_t* source, const uint32_t* target,
                                 const uint32_t* offsets, const double* unary,
                                 const double* f, const double* oldFactor,
                                 const double* oldY, const double* oldZ,
                                 const MaxTwo* sourceMax, const MaxTwo* targetMax,
                                 uint32_t count, double damping, double* newFactor,
                                 double* newY, double* newZ, double* beliefY,
                                 double* beliefZ) {
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    double d = unary[id];
    for (uint32_t e = offsets[id]; e < offsets[id + 1]; ++e) d += f[e];
    const MaxTwo row = sourceMax[source[id]], column = targetMax[target[id]];
    const double yc = d - (column.owner == id ? column.second : column.first);
    const double zc = d - (row.owner == id ? row.second : row.first);
    beliefY[id] = yc; beliefZ[id] = zc;
    newY[id] = damping * yc + (1.0 - damping) * oldY[id];
    newZ[id] = damping * zc + (1.0 - damping) * oldZ[id];
    for (uint32_t e = offsets[id]; e < offsets[id + 1]; ++e) {
        const double sp = (yc + zc - d) - f[e];
        newFactor[e] = damping * sp + (1.0 - damping) * oldFactor[e];
    }
}

// Each group picks its best eligible edge from the matching snapshot.
__global__ void pickBest(const uint32_t* offsets, const uint32_t* ids,
                         const uint32_t* source, const uint32_t* target,
                         const uint32_t* sourceMatch, const uint32_t* targetMatch,
                         const double* weights, uint32_t groups, uint32_t* best) {
    const uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) return;
    uint32_t selected = UINT32_MAX;
    double maxWeight = 0.0;
    for (uint32_t j = offsets[group]; j < offsets[group + 1]; ++j) {
        const uint32_t id = ids ? ids[j] : j;
        const double weight = weights[id];
        if (sourceMatch[source[id]] == UINT32_MAX && targetMatch[target[id]] == UINT32_MAX &&
            weight > 0.0 && (weight > maxWeight || (weight == maxWeight && id < selected))) {
            selected = id;
            maxWeight = weight;
        }
    }
    best[group] = selected;
}

// Mutual selections are disjoint; kernel ordering separates writes from picks.
__global__ void commitBest(const uint32_t* rowBest, const uint32_t* columnBest,
                           const uint32_t* target, uint32_t rows,
                           uint32_t* sourceMatch, uint32_t* targetMatch,
                           unsigned int* committed) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const uint32_t id = rowBest[row];
    if (id == UINT32_MAX) return;
    const uint32_t column = target[id];
    if (columnBest[column] == id) {
        sourceMatch[row] = column;
        targetMatch[column] = row;
        atomicAdd(committed, 1u);
    }
}

void validateOffsets(const std::vector<uint32_t>& offsets, std::size_t expected,
                     const char* name) {
    if (offsets.empty() || offsets.front() != 0 || offsets.back() != expected ||
        !std::is_sorted(offsets.begin(), offsets.end()))
        throw std::invalid_argument(std::string("invalid cuAlign ") + name + " offsets");
}

void validate(const CuAlignProblem& p, double beta, double damping) {
    const std::size_t n = p.unary.size(), m = p.reverse.size();
    if (n > UINT32_MAX || m > UINT32_MAX || p.sourceOffsets.size() > UINT32_MAX ||
        p.targetOffsets.size() > UINT32_MAX || p.source.size() != n || p.target.size() != n ||
        p.factorOffsets.size() != n + 1 || p.arcMultiplicity.size() != m ||
        p.targetCandidates.size() != n || !std::isfinite(beta) || beta < 0 ||
        !std::isfinite(damping) || damping < 0 || damping > 1)
        throw std::invalid_argument("invalid cuAlign problem dimensions or parameters");
    validateOffsets(p.sourceOffsets, n, "source");
    validateOffsets(p.targetOffsets, n, "target");
    validateOffsets(p.factorOffsets, m, "factor");
    std::vector<uint8_t> seen(n, 0);
    for (std::size_t s = 0; s + 1 < p.sourceOffsets.size(); ++s)
        for (uint32_t i = p.sourceOffsets[s]; i < p.sourceOffsets[s + 1]; ++i)
            if (p.source[i] != s || p.target[i] >= p.targetOffsets.size() - 1 || !std::isfinite(p.unary[i])) throw std::invalid_argument("invalid cuAlign candidate");
    for (std::size_t t = 0; t + 1 < p.targetOffsets.size(); ++t)
        for (uint32_t j = p.targetOffsets[t]; j < p.targetOffsets[t + 1]; ++j) {
            const uint32_t i = p.targetCandidates[j];
            if (i >= n || seen[i] || p.target[i] != t) throw std::invalid_argument("invalid cuAlign target grouping");
            seen[i] = 1;
        }
    for (std::size_t e = 0; e < m; ++e)
        if (p.reverse[e] >= m || p.reverse[p.reverse[e]] != e || p.arcMultiplicity[e] == 0 || p.arcMultiplicity[e] != p.arcMultiplicity[p.reverse[e]])
            throw std::invalid_argument("invalid cuAlign reverse factor index");
}

class CudaStep final : public CuAlignCudaStep {
public:
    CudaStep(const CuAlignProblem& p, double beta, double damping)
        : n_(static_cast<uint32_t>(p.unary.size())), m_(static_cast<uint32_t>(p.reverse.size())),
          rows_(static_cast<uint32_t>(p.sourceOffsets.size() - 1)),
          columns_(static_cast<uint32_t>(p.targetOffsets.size() - 1)), beta_(beta),
          damping_(damping), mix_(damping),
          source_(n_), target_(n_), unary_(n_), sourceOffsets_(p.sourceOffsets.size()),
          targetOffsets_(p.targetOffsets.size()), targetIds_(n_), factorOffsets_(p.factorOffsets.size()),
          reverse_(m_), multiplicity_(m_), f_(m_), factorsA_(m_), factorsB_(m_),
          yA_(n_), yB_(n_), zA_(n_), zB_(n_), beliefY_(n_), beliefZ_(n_),
          rowMax_(rows_), columnMax_(columns_) {
        source_.upload(p.source); target_.upload(p.target); unary_.upload(p.unary);
        sourceOffsets_.upload(p.sourceOffsets); targetOffsets_.upload(p.targetOffsets);
        targetIds_.upload(p.targetCandidates); factorOffsets_.upload(p.factorOffsets);
        reverse_.upload(p.reverse); multiplicity_.upload(p.arcMultiplicity);
        factorsA_.zero(); yA_.zero(); zA_.zero();
        oldF_ = factorsA_.get(); newF_ = factorsB_.get();
        oldY_ = yA_.get(); newY_ = yB_.get(); oldZ_ = zA_.get(); newZ_ = zB_.get();
    }
    void advanceDevice() override {
        constexpr unsigned threads = 256;
        auto blocks = [](uint32_t n) { return static_cast<unsigned>((static_cast<uint64_t>(n) + threads - 1) / threads); };
        if (m_) factorValues<<<blocks(m_), threads>>>(reverse_.get(), multiplicity_.get(), oldF_, m_, beta_, f_.get());
        if (rows_) groupMax<<<blocks(rows_), threads>>>(sourceOffsets_.get(), nullptr, oldY_, rows_, rowMax_.get());
        if (columns_) groupMax<<<blocks(columns_), threads>>>(targetOffsets_.get(), targetIds_.get(), oldZ_, columns_, columnMax_.get());
        if (n_) updateCandidates<<<blocks(n_), threads>>>(source_.get(), target_.get(), factorOffsets_.get(), unary_.get(), f_.get(), oldF_, oldY_, oldZ_, rowMax_.get(), columnMax_.get(), n_, mix_, newF_, newY_, newZ_, beliefY_.get(), beliefZ_.get());
        check(cudaGetLastError(), "cuAlign kernel launch");
        check(cudaDeviceSynchronize(), "cuAlign propagation synchronization");
        std::swap(oldF_, newF_); std::swap(oldY_, newY_); std::swap(oldZ_, newZ_);
        mix_ *= damping_;
        hasBeliefs_ = true;
    }
    void advance(std::vector<double>& y, std::vector<double>& z) override {
        advanceDevice();
        downloadBeliefs(false, y);
        downloadBeliefs(true, z);
    }
    void downloadBeliefs(bool useZ, std::vector<double>& weights) override {
        if (!hasBeliefs_) throw std::logic_error("cuAlign beliefs requested before advance");
        (useZ ? beliefZ_ : beliefY_).download(weights);
        if (!n_) check(cudaDeviceSynchronize(), "cuAlign synchronization");
    }
    std::vector<uint32_t> roundDevice(bool useZ) override {
        if (!hasBeliefs_) throw std::logic_error("cuAlign rounding requested before advance");
        DeviceBuffer<uint32_t> sourceMatch(rows_), targetMatch(columns_), rowBest(rows_), columnBest(columns_);
        DeviceBuffer<unsigned int> committed(1);
        if (rows_) check(cudaMemset(sourceMatch.get(), 0xff, size_t(rows_) * sizeof(uint32_t)), "initialize cuAlign source match");
        if (columns_) check(cudaMemset(targetMatch.get(), 0xff, size_t(columns_) * sizeof(uint32_t)), "initialize cuAlign target match");
        constexpr unsigned threads = 256;
        auto blocks = [](uint32_t n) { return static_cast<unsigned>((static_cast<uint64_t>(n) + threads - 1) / threads); };
        const double* weights = useZ ? beliefZ_.get() : beliefY_.get();
        if (rows_ && columns_) {
            for (;;) {
                committed.zero();
                pickBest<<<blocks(rows_), threads>>>(sourceOffsets_.get(), nullptr, source_.get(), target_.get(), sourceMatch.get(), targetMatch.get(), weights, rows_, rowBest.get());
                pickBest<<<blocks(columns_), threads>>>(targetOffsets_.get(), targetIds_.get(), source_.get(), target_.get(), sourceMatch.get(), targetMatch.get(), weights, columns_, columnBest.get());
                commitBest<<<blocks(rows_), threads>>>(rowBest.get(), columnBest.get(), target_.get(), rows_, sourceMatch.get(), targetMatch.get(), committed.get());
                check(cudaGetLastError(), "cuAlign rounding kernel launch");
                std::vector<unsigned int> count;
                committed.download(count);
                if (count[0] == 0) break;
            }
        }
        std::vector<uint32_t> result;
        sourceMatch.download(result);
        if (!rows_) check(cudaDeviceSynchronize(), "cuAlign rounding synchronization");
        return result;
    }
private:
    bool hasBeliefs_ = false;
    uint32_t n_, m_, rows_, columns_;
    double beta_, damping_, mix_;
    DeviceBuffer<uint32_t> source_, target_;
    DeviceBuffer<double> unary_;
    DeviceBuffer<uint32_t> sourceOffsets_, targetOffsets_, targetIds_, factorOffsets_, reverse_;
    DeviceBuffer<uint8_t> multiplicity_;
    DeviceBuffer<double> f_, factorsA_, factorsB_, yA_, yB_, zA_, zB_, beliefY_, beliefZ_;
    DeviceBuffer<MaxTwo> rowMax_, columnMax_;
    double *oldF_, *newF_, *oldY_, *newY_, *oldZ_, *newZ_;
};

} // namespace

std::unique_ptr<CuAlignCudaStep> makeCuAlignCuda(const CuAlignProblem& problem, double beta, double damping) {
    validate(problem, beta, damping);
    return std::unique_ptr<CuAlignCudaStep>(new CudaStep(problem, beta, damping));
}

} // namespace cualign
