#ifndef CUALIGN_HPP
#define CUALIGN_HPP

#include <cstdint>
#include <memory>
#include <vector>

namespace cualign {

struct CuAlignProblem {
    // Candidate IDs index source, target, unary, and the messages y and z.
    std::vector<uint32_t> source, target;
    std::vector<double> unary;
    // Candidate IDs are grouped by source; targetCandidates groups IDs by target.
    std::vector<uint32_t> sourceOffsets, targetOffsets, targetCandidates;
    // One directed CSR entry per orientation of a symmetric pair factor.
    std::vector<uint32_t> factorOffsets, reverse;
    std::vector<uint8_t> arcMultiplicity;
};

class CuAlignStep {
public:
    virtual ~CuAlignStep() = default;
    // Retain damped messages for the next iteration and expose current beliefs.
    virtual void advance(std::vector<double>& y, std::vector<double>& z) = 0;
};

class CuAlignCudaStep : public CuAlignStep {
public:
    virtual void advanceDevice() = 0;
    virtual void downloadBeliefs(bool useZ, std::vector<double>& weights) = 0;
    // Round positive edges on the device. Unmatched sources have UINT32_MAX.
    virtual std::vector<uint32_t> roundDevice(bool useZ) = 0;
};

std::unique_ptr<CuAlignStep> makeCuAlignCpu(
    const CuAlignProblem& problem, double beta, double damping);
std::unique_ptr<CuAlignCudaStep> makeCuAlignCuda(
    const CuAlignProblem& problem, double beta, double damping);

// Returns target IDs indexed by source ID. Throws if support lacks a full match.
std::vector<uint32_t> roundCuAlign(
    const CuAlignProblem& problem, const std::vector<double>& weights);
std::vector<uint32_t> completeCuAlign(
    const CuAlignProblem& problem, const std::vector<double>& weights,
    std::vector<uint32_t> partial);

} // namespace cualign

#endif
