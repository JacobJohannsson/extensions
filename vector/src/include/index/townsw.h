#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lbug {
namespace vector_extension {

// Work counters for one vector search. Every search mode fills the generic counters, so modes can
// be compared by work done and not only by time. The TOWNSW mode also fills the rest.
struct VectorSearchStats {
    std::string searchType;
    double searchMs = 0;
    uint64_t numResults = 0;

    // Generic.
    uint64_t distances = 0;    // distance computations to indexed vectors
    uint64_t listsScanned = 0; // adjacency lists read from the HNSW layers
    uint64_t expansions = 0;   // nodes popped from the candidate queue and expanded

    // TOWNSW.
    uint64_t selectedWitnesses = 0;   // witnesses that passed the filter
    uint64_t aliveLeaves = 0;         // leaves of selected witnesses
    uint64_t treeNodesMarked = 0;     // tree nodes touched while marking alive branches
    uint64_t treeNodesOpened = 0;     // tree nodes expanded while routing
    uint64_t centroidDistances = 0;   // distances to tree-node and leaf centers
    uint64_t leavesPulled = 0;        // leaves whose entry papers were used as start points
    uint64_t entryDistances = 0;      // subset of `distances`: entry papers
    uint64_t descentDistances = 0;    // subset of `distances`: greedy descent from an entry paper
    uint64_t witnessListsScanned = 0; // per-witness NSW adjacency lists read
    uint64_t edgesScanned = 0;        // neighbor IDs looked at, both edge kinds
    uint64_t maskChecks = 0;          // "is this paper's witness selected" checks

    static void publish(const VectorSearchStats& stats);
    static VectorSearchStats last();
};

// The offline-built part of a TOWNSW index: which witness each indexed node belongs to,
// the leaves (groups of nodes of one witness, each with a center and entry nodes), and a tree
// over the leaf centers. Built by scripts/build_index.py. Node IDs are node offsets.
//
// A child reference c >= 0 is tree node c; c < 0 is leaf -(c + 1). Tree node 0 is the root.
struct TownswData {
    uint64_t numNodes = 0;
    uint64_t numWitnesses = 0;
    uint64_t numLeaves = 0;
    uint64_t numTreeNodes = 0;
    uint64_t dim = 0;
    uint64_t entriesPerLeaf = 0;

    std::vector<int32_t> nodeWitness;
    std::vector<int32_t> leafWitness;
    std::vector<uint32_t> leafCount;
    std::vector<int64_t> leafEntries; // numLeaves * entriesPerLeaf, padded with -1
    std::vector<float> leafCenter;    // numLeaves * dim
    std::vector<int32_t> leafParent;
    std::vector<int32_t> treeParent; // -1 for the root
    std::vector<uint64_t> treeChildStart;
    std::vector<int32_t> treeChildren;
    std::vector<float> treeCenter; // numTreeNodes * dim
    std::vector<uint64_t> witnessLeafStart;
    std::vector<int32_t> witnessLeaves;

    // Loads the file once per path and keeps it in memory for the life of the process.
    static std::shared_ptr<const TownswData> getOrLoad(const std::string& path);
};

} // namespace vector_extension
} // namespace lbug
