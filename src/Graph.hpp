#ifndef CUALIGN_GRAPH_HPP
#define CUALIGN_GRAPH_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// A directed graph with stable node indices and sorted outgoing neighbors.
class Graph {
public:
    using Edge = std::array<std::uint32_t, 2>;

    Graph(std::vector<Edge> edges, std::vector<std::string> names)
        : edges_(std::move(edges)), names_(std::move(names)), adjacency_(names_.size()) {
        if (names_.size() > std::numeric_limits<std::uint32_t>::max() ||
            edges_.size() > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("Graph exceeds the supported size");
        for (std::size_t i = 0; i < names_.size(); ++i) {
            if (!indices_.emplace(names_[i], static_cast<std::uint32_t>(i)).second)
                throw std::invalid_argument("Duplicate graph node name: " + names_[i]);
        }
        for (const auto& edge : edges_) {
            if (edge[0] >= names_.size() || edge[1] >= names_.size())
                throw std::invalid_argument("Graph arc endpoint out of range");
            adjacency_[edge[0]].push_back(edge[1]);
        }
        for (auto& neighbors : adjacency_) {
            std::sort(neighbors.begin(), neighbors.end());
            if (std::adjacent_find(neighbors.begin(), neighbors.end()) != neighbors.end())
                throw std::invalid_argument("Duplicate graph arc");
        }
    }

    std::uint32_t getNumNodes() const { return static_cast<std::uint32_t>(names_.size()); }
    std::uint32_t getNumEdges() const { return static_cast<std::uint32_t>(edges_.size()); }
    bool hasNodeName(const std::string& name) const { return indices_.find(name) != indices_.end(); }
    std::uint32_t getNameIndex(const std::string& name) const { return indices_.at(name); }
    const std::string& getNodeName(std::uint32_t node) const { return names_.at(node); }
    const std::vector<Edge>* getEdgeList() const { return &edges_; }
    const std::vector<std::uint32_t>* getAdjList(std::uint32_t node) const { return &adjacency_.at(node); }
    bool hasEdge(std::uint32_t from, std::uint32_t to) const {
        const auto& neighbors = adjacency_.at(from);
        return std::binary_search(neighbors.begin(), neighbors.end(), to);
    }

private:
    std::vector<Edge> edges_;
    std::vector<std::string> names_;
    std::vector<std::vector<std::uint32_t>> adjacency_;
    std::unordered_map<std::string, std::uint32_t> indices_;
};

#endif
