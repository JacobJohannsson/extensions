#include "binder/binder.h"
#include "function/hnsw_index_functions.h"
#include "function/table/bind_data.h"
#include "function/table/bind_input.h"
#include "function/table/simple_table_function.h"
#include "index/townsw.h"

using namespace lbug::common;
using namespace lbug::function;

namespace lbug {
namespace vector_extension {

namespace {

struct Counter {
    const char* name;
    uint64_t VectorSearchStats::* member;
};

// Column order after search_type, search_ms and num_results.
constexpr Counter COUNTERS[] = {
    {"distances", &VectorSearchStats::distances},
    {"lists_scanned", &VectorSearchStats::listsScanned},
    {"expansions", &VectorSearchStats::expansions},
    {"selected_witnesses", &VectorSearchStats::selectedWitnesses},
    {"alive_leaves", &VectorSearchStats::aliveLeaves},
    {"tree_nodes_marked", &VectorSearchStats::treeNodesMarked},
    {"tree_nodes_opened", &VectorSearchStats::treeNodesOpened},
    {"centroid_distances", &VectorSearchStats::centroidDistances},
    {"leaves_pulled", &VectorSearchStats::leavesPulled},
    {"entry_distances", &VectorSearchStats::entryDistances},
    {"descent_distances", &VectorSearchStats::descentDistances},
    {"witness_lists_scanned", &VectorSearchStats::witnessListsScanned},
    {"edges_scanned", &VectorSearchStats::edgesScanned},
    {"mask_checks", &VectorSearchStats::maskChecks},
};

} // namespace

static offset_t internalTableFunc(const TableFuncMorsel& /*morsel*/,
    const TableFuncInput& /*input*/, DataChunk& output) {
    const auto stats = VectorSearchStats::last();
    const auto pos = output.state->getSelVector()[0];
    output.getValueVectorMutable(0).setValue(pos, stats.searchType);
    output.getValueVectorMutable(1).setValue(pos, stats.searchMs);
    output.getValueVectorMutable(2).setValue(pos, static_cast<int64_t>(stats.numResults));
    auto column = 3u;
    for (const auto& counter : COUNTERS) {
        output.getValueVectorMutable(column++).setValue(pos,
            static_cast<int64_t>(stats.*counter.member));
    }
    return 1;
}

static std::unique_ptr<TableFuncBindData> bindFunc(const main::ClientContext*,
    const TableFuncBindInput* input) {
    std::vector<std::string> returnColumnNames{"search_type", "search_ms", "num_results"};
    std::vector<LogicalType> returnTypes;
    returnTypes.emplace_back(LogicalType::STRING());
    returnTypes.emplace_back(LogicalType::DOUBLE());
    returnTypes.emplace_back(LogicalType::INT64());
    for (const auto& counter : COUNTERS) {
        returnColumnNames.emplace_back(counter.name);
        returnTypes.emplace_back(LogicalType::INT64());
    }
    returnColumnNames =
        TableFunction::extractYieldVariables(returnColumnNames, input->yieldVariables);
    auto columns = input->binder->createVariables(returnColumnNames, returnTypes);
    return std::make_unique<TableFuncBindData>(std::move(columns), 1 /* one row result */);
}

function_set VectorSearchStatsFunction::getFunctionSet() {
    function_set functionSet;
    auto function = std::make_unique<TableFunction>(name, std::vector<LogicalTypeID>{});
    function->tableFunc = SimpleTableFunc::getTableFunc(internalTableFunc);
    function->bindFunc = bindFunc;
    function->initSharedStateFunc = SimpleTableFunc::initSharedState;
    function->initLocalStateFunc = TableFunction::initEmptyLocalState;
    functionSet.push_back(std::move(function));
    return functionSet;
}

} // namespace vector_extension
} // namespace lbug
