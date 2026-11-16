#include "index/townsw.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "common/exception/runtime.h"
#include "index/hnsw_index.h"
#include "simsimd.h"
#include <format>

using namespace lbug::storage;
using namespace lbug::transaction;

namespace lbug {
namespace vector_extension {

// ---- Counters ----

static std::mutex lastStatsMtx;
static VectorSearchStats lastStats;

void VectorSearchStats::publish(const VectorSearchStats& stats) {
    std::lock_guard lck{lastStatsMtx};
    lastStats = stats;
}

VectorSearchStats VectorSearchStats::last() {
    std::lock_guard lck{lastStatsMtx};
    return lastStats;
}

// ---- Loading the offline-built file ----

template<typename T>
static void readArray(std::ifstream& in, std::vector<T>& out, uint64_t count,
    const std::string& path) {
    out.resize(count);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (!in) {
        throw common::RuntimeException{std::format("TOWNSW index file {} is truncated.", path)};
    }
}

static std::shared_ptr<const TownswData> loadTownswData(const std::string& path) {
    std::ifstream in{path, std::ios::binary};
    if (!in) {
        throw common::RuntimeException{std::format("Cannot open TOWNSW index file {}.", path)};
    }
    char magic[8];
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, "TOWNSW01", sizeof(magic)) != 0) {
        throw common::RuntimeException{std::format("{} is not a TOWNSW index file.", path)};
    }
    uint64_t header[7];
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    auto data = std::make_shared<TownswData>();
    data->numNodes = header[0];
    data->numWitnesses = header[1];
    data->numLeaves = header[2];
    data->numTreeNodes = header[3];
    data->dim = header[4];
    data->entriesPerLeaf = header[5];
    const auto numChildRefs = header[6];
    readArray(in, data->nodeWitness, data->numNodes, path);
    readArray(in, data->leafWitness, data->numLeaves, path);
    readArray(in, data->leafCount, data->numLeaves, path);
    readArray(in, data->leafEntries, data->numLeaves * data->entriesPerLeaf, path);
    readArray(in, data->leafCenter, data->numLeaves * data->dim, path);
    readArray(in, data->leafParent, data->numLeaves, path);
    readArray(in, data->treeParent, data->numTreeNodes, path);
    readArray(in, data->treeChildStart, data->numTreeNodes + 1, path);
    readArray(in, data->treeChildren, numChildRefs, path);
    readArray(in, data->treeCenter, data->numTreeNodes * data->dim, path);
    readArray(in, data->witnessLeafStart, data->numWitnesses + 1, path);
    readArray(in, data->witnessLeaves, data->numLeaves, path);
    return data;
}

std::shared_ptr<const TownswData> TownswData::getOrLoad(const std::string& path) {
    static std::mutex mtx;
    static std::unordered_map<std::string, std::shared_ptr<const TownswData>> loaded;
    std::lock_guard lck{mtx};
    auto it = loaded.find(path);
    if (it != loaded.end()) {
        return it->second;
    }
    auto data = loadTownswData(path);
    loaded.emplace(path, data);
    return data;
}

// ---- Search ----

namespace {

struct TreeRef {
    double key;
    int32_t ref; // >= 0: tree node; < 0: leaf -(ref + 1)
};
struct TreeRefFarther {
    bool operator()(const TreeRef& l, const TreeRef& r) const { return l.key > r.key; }
};

} // namespace

// TOWNSW search (docs/design.md in the townsw repo, Section 4).
//
// The graph searched is the HNSW's lower layer plus one NSW per witness set, restricted to the
// nodes whose witness passed the filter. Both edge kinds are read whenever a node is expanded;
// nothing is materialized per query. The tree over leaf centers is only used to choose where the
// search starts: it yields the leaves of selected witnesses nearest-center-first, and each
// leaf's entry nodes seed the beam.
std::vector<NodeWithDistance> OnDiskHNSWIndex::searchTownsw(Transaction* transaction,
    const EmbeddingHandle& queryVector, HNSWSearchState& searchState) const {
    if (config.quantization != QuantizationType::NONE ||
        typeInfo.getChildType().getLogicalTypeID() != common::LogicalTypeID::FLOAT) {
        throw common::RuntimeException{
            "search_type 'townsw' needs an unquantized FLOAT vector index."};
    }
    DASSERT(searchState.townsw != nullptr && searchState.witnessGraph != nullptr);
    const auto& data = *searchState.townsw;
    auto& stats = *searchState.stats;
    const auto& embeddings = *searchState.embeddings;
    const auto dim = embeddings.getDimension();
    const auto numTotalRows = nodeTable.getNumTotalRows(transaction);
    if (data.dim != dim || data.numNodes != numTotalRows) {
        throw common::RuntimeException{
            std::format("The TOWNSW index file was built for {} nodes of dimension {}, but the "
                        "table has {} nodes of dimension {}.",
                data.numNodes, data.dim, numTotalRows, dim)};
    }
    const auto tableID = indexInfo.tableID;
    const auto& hnswStorageInfo = storageInfo->cast<HNSWStorageInfo>();
    const auto ef = searchState.ef;
    const auto& queryConfig = searchState.config;

    // ---- Which witnesses are selected, and which tree branches lead to them ----
    const auto allSelected = searchState.witnessMask == nullptr;
    std::vector<uint8_t> witnessSelected;
    std::vector<uint8_t> treeAlive;
    if (allSelected) {
        stats.selectedWitnesses = data.numWitnesses;
        stats.aliveLeaves = data.numLeaves;
    } else {
        witnessSelected.assign(data.numWitnesses, 0);
        treeAlive.assign(data.numTreeNodes, 0);
        for (const auto witness : searchState.witnessMask->range(0, data.numWitnesses)) {
            witnessSelected[witness] = 1;
            stats.selectedWitnesses++;
            for (auto i = data.witnessLeafStart[witness]; i < data.witnessLeafStart[witness + 1];
                i++) {
                stats.aliveLeaves++;
                auto parent = data.leafParent[data.witnessLeaves[i]];
                while (parent >= 0 && !treeAlive[parent]) {
                    treeAlive[parent] = 1;
                    stats.treeNodesMarked++;
                    parent = data.treeParent[parent];
                }
            }
        }
    }
    const auto isWitnessSelected = [&](int32_t witness) {
        return allSelected || witnessSelected[witness] != 0;
    };
    const auto isNodeSelected = [&](common::offset_t offset) {
        stats.maskChecks++;
        return offset < data.numNodes && isWitnessSelected(data.nodeWitness[offset]);
    };
    if (stats.aliveLeaves == 0) {
        return {};
    }

    // ---- Routing: leaves of selected witnesses, nearest center first ----
    const auto* query = static_cast<const float*>(queryVector.getPtr());
    simsimd_distance_t querySq = 0;
    simsimd_dot_f32(query, query, dim, &querySq);
    const auto queryNorm = querySq > 0 ? std::sqrt(querySq) : 1.0;
    // A center is the mean of unit vectors, so 1 - (q . center) / |q| is the mean cosine distance
    // from the query to the vectors under that center.
    const auto centerKey = [&](const float* center) {
        simsimd_distance_t dot = 0;
        simsimd_dot_f32(query, center, dim, &dot);
        stats.centroidDistances++;
        return 1.0 - dot / queryNorm;
    };
    std::priority_queue<TreeRef, std::vector<TreeRef>, TreeRefFarther> treeQueue;
    treeQueue.push({0.0, 0});
    const auto pullLeaf = [&]() -> int32_t {
        while (!treeQueue.empty()) {
            const auto ref = treeQueue.top().ref;
            treeQueue.pop();
            if (ref < 0) {
                return -(ref + 1);
            }
            stats.treeNodesOpened++;
            for (auto i = data.treeChildStart[ref]; i < data.treeChildStart[ref + 1]; i++) {
                const auto child = data.treeChildren[i];
                if (child >= 0) {
                    if (allSelected || treeAlive[child]) {
                        treeQueue.push({centerKey(&data.treeCenter[child * dim]), child});
                    }
                } else {
                    const auto leaf = -(child + 1);
                    if (isWitnessSelected(data.leafWitness[leaf])) {
                        treeQueue.push({centerKey(&data.leafCenter[leaf * dim]), child});
                    }
                }
            }
        }
        return -1;
    };

    // ---- Beam search over lower-layer edges plus witness NSW edges ----
    searchState.visited.resize(numTotalRows);
    searchState.visited.reset();
    searchState.nbrScanState = searchState.lowerGraph->prepareRelScan(
        *searchState.lowerRelTableEntry, hnswStorageInfo.lowerRelTableID, tableID, {});
    searchState.witnessNbrScanState = searchState.witnessGraph->prepareRelScan(
        *searchState.witnessRelEntry, searchState.witnessRelTableID, tableID, {});
    auto& visited = searchState.visited;
    min_node_priority_queue_t candidates;
    max_node_priority_queue_t results;

    const auto distanceTo = [&](common::offset_t offset) -> std::optional<double> {
        const auto vector = embeddings.getEmbedding(offset, *searchState.embeddingScanState);
        if (vector.isNull()) {
            return std::nullopt;
        }
        return searchState.metricFunc(queryVector.getPtr(), vector.getPtr(), dim);
    };
    // Returns true if the node entered the beam.
    const auto consider = [&](common::offset_t offset, double dist) {
        if (results.size() < ef || dist < results.top().distance) {
            if (results.size() >= ef) {
                results.pop();
            }
            results.push({offset, dist});
            candidates.push({offset, dist});
            return true;
        }
        return false;
    };
    // Visits every unvisited witness-NSW neighbor of `node`. Returns the nearest one seen and
    // whether any entered the beam.
    const auto scanWitnessNbrs = [&](common::offset_t node, uint64_t& distanceCounter,
                                     NodeWithDistance& nearest) {
        auto entered = false;
        stats.witnessListsScanned++;
        auto itr = searchState.witnessGraph->scanFwd(common::nodeID_t{node, tableID},
            *searchState.witnessNbrScanState);
        for (const auto chunk : itr) {
            chunk.forEach([&](auto neighbors, auto, auto i) {
                const auto nbr = neighbors[i].offset;
                stats.edgesScanned++;
                if (visited.contains(nbr)) {
                    return;
                }
                visited.add(nbr);
                const auto dist = distanceTo(nbr);
                if (!dist) {
                    return;
                }
                distanceCounter++;
                entered |= consider(nbr, *dist);
                if (*dist < nearest.distance) {
                    nearest = {nbr, *dist};
                }
            });
        }
        return entered;
    };
    // Uses a leaf as a start point: its entry nodes, then optionally a greedy walk from the best
    // entry toward the query inside the witness NSW. Returns true if anything entered the beam.
    const auto seedFromLeaf = [&](int32_t leaf) {
        stats.leavesPulled++;
        auto entered = false;
        NodeWithDistance nearest;
        for (auto i = 0u; i < data.entriesPerLeaf; i++) {
            const auto entry = data.leafEntries[leaf * data.entriesPerLeaf + i];
            if (entry < 0) {
                break;
            }
            if (visited.contains(entry)) {
                continue;
            }
            visited.add(entry);
            const auto dist = distanceTo(entry);
            if (!dist) {
                continue;
            }
            stats.entryDistances++;
            entered |= consider(entry, *dist);
            if (*dist < nearest.distance) {
                nearest = {static_cast<common::offset_t>(entry), *dist};
            }
        }
        if (!queryConfig.leafDescent) {
            return entered;
        }
        auto current = common::INVALID_OFFSET;
        while (nearest.nodeOffset != current) {
            current = nearest.nodeOffset;
            entered |= scanWitnessNbrs(current, stats.descentDistances, nearest);
        }
        return entered;
    };
    uint64_t expansionDistances = 0;
    const auto expand = [&](common::offset_t node) {
        stats.expansions++;
        stats.listsScanned++;
        uint64_t numNbrs = 0;
        uint64_t numSelectedNbrs = 0;
        auto itr = searchState.lowerGraph->scanFwd(common::nodeID_t{node, tableID},
            *searchState.nbrScanState);
        for (const auto chunk : itr) {
            chunk.forEach([&](auto neighbors, auto, auto i) {
                const auto nbr = neighbors[i].offset;
                stats.edgesScanned++;
                numNbrs++;
                if (visited.contains(nbr)) {
                    numSelectedNbrs++; // only selected nodes are ever marked visited
                    return;
                }
                if (!isNodeSelected(nbr)) {
                    return;
                }
                numSelectedNbrs++;
                visited.add(nbr);
                const auto dist = distanceTo(nbr);
                if (dist) {
                    expansionDistances++;
                    consider(nbr, *dist);
                }
            });
        }
        // Optionally skip the witness edges where the lower layer is already well connected
        // among selected nodes. The default threshold is above 1, so they are always read.
        if (numNbrs > 0 && static_cast<double>(numSelectedNbrs) / static_cast<double>(numNbrs) >=
                               queryConfig.witnessEdgeSel) {
            return;
        }
        NodeWithDistance unused;
        scanWitnessNbrs(node, expansionDistances, unused);
    };

    for (auto i = 0; i < queryConfig.initLeaves; i++) {
        const auto leaf = pullLeaf();
        if (leaf < 0) {
            break;
        }
        seedFromLeaf(leaf);
    }
    int64_t misses = 0;
    while (true) {
        if (!candidates.empty()) {
            const auto [candidate, candidateDist] = candidates.top();
            if (results.size() < ef || candidateDist <= results.top().distance) {
                candidates.pop();
                expand(candidate);
                continue;
            }
        }
        // The beam cannot continue. If it is not full yet, selected nodes may remain that it
        // could not reach, so take another start point. If it is full, take more start points
        // only while they keep paying off.
        const auto beamFull = results.size() >= ef;
        if (beamFull && misses >= queryConfig.leafPatience) {
            break;
        }
        const auto leaf = pullLeaf();
        if (leaf < 0) {
            break;
        }
        const auto entered = seedFromLeaf(leaf);
        if (beamFull) {
            misses = entered ? 0 : misses + 1;
        }
    }
    return popTopK(results, searchState.k);
}

} // namespace vector_extension
} // namespace lbug
