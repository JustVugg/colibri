//===----- graph.hpp ----------------------------*- C++ -*-----------------===//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/sycl.hpp>
#include <unordered_map>

namespace dpct {
namespace experimental {

typedef sycl::ext::oneapi::experimental::command_graph<
    sycl::ext::oneapi::experimental::graph_state::modifiable>
    *command_graph_ptr;

typedef sycl::ext::oneapi::experimental::command_graph<
    sycl::ext::oneapi::experimental::graph_state::executable>
    *command_graph_exec_ptr;

typedef sycl::ext::oneapi::experimental::node *node_ptr;

namespace detail {
class graph_mgr {
public:
  graph_mgr() = default;
  graph_mgr(const graph_mgr &) = delete;
  graph_mgr &operator=(const graph_mgr &) = delete;
  graph_mgr(graph_mgr &&) = delete;
  graph_mgr &operator=(graph_mgr &&) = delete;

  static graph_mgr &instance() {
    static graph_mgr instance;
    return instance;
  }

  void begin_recording(sycl::queue *queue_ptr) {
    // Calling begin_recording on an already recording queue is a no-op in SYCL
    if (queue_graph_map.find(queue_ptr) != queue_graph_map.end()) {
      return;
    }
    auto graph = new sycl::ext::oneapi::experimental::command_graph<
        sycl::ext::oneapi::experimental::graph_state::modifiable>(
        queue_ptr->get_context(), queue_ptr->get_device());
    auto result = queue_graph_map.insert({queue_ptr, graph});
    if (!result.second) {
      delete graph;
      return;
    }
    graph->begin_recording(*queue_ptr);
  }

  void end_recording(sycl::queue *queue_ptr,
                     dpct::experimental::command_graph_ptr *graph) {
    auto it = queue_graph_map.find(queue_ptr);
    if (it == queue_graph_map.end()) {
      return;
    }
    *graph = it->second;
    queue_graph_map.erase(it);
    (*graph)->end_recording();
  }

  void get_nodes(dpct::experimental::command_graph_ptr graph,
                 dpct::experimental::node_ptr *nodesArray,
                 std::size_t *numberOfNodes) {
    auto nodes = graph->get_nodes();
    nodes_map[graph] = nodes;
    *numberOfNodes = nodes.size();
    if (!nodesArray) {
      return;
    }
    for (std::size_t i = 0; i < *numberOfNodes; i++) {
      nodesArray[i] = &nodes_map[graph][i];
    }
  }

  void get_root_nodes(dpct::experimental::command_graph_ptr graph,
                      dpct::experimental::node_ptr *nodesArray,
                      std::size_t *numberOfNodes) {
    auto root_nodes = graph->get_root_nodes();
    root_nodes_map[graph] = root_nodes;
    *numberOfNodes = root_nodes.size();
    if (!nodesArray) {
      return;
    }
    for (std::size_t i = 0; i < *numberOfNodes; i++) {
      nodesArray[i] = &root_nodes_map[graph][i];
    }
  }

private:
  std::unordered_map<sycl::queue *, command_graph_ptr> queue_graph_map;
  std::unordered_map<dpct::experimental::command_graph_ptr,
                     std::vector<sycl::ext::oneapi::experimental::node>>
      nodes_map;
  std::unordered_map<dpct::experimental::command_graph_ptr,
                     std::vector<sycl::ext::oneapi::experimental::node>>
      root_nodes_map;
};
} // namespace detail

/// Begins recording commands into a command graph for a specific SYCL queue.
/// \param [in] queue_ptr A pointer to the SYCL queue on which the commands
/// will be recorded.
static inline void begin_recording(sycl::queue *queue_ptr) {
  detail::graph_mgr::instance().begin_recording(queue_ptr);
}

/// Ends the recording of commands into a command graph for a specific SYCL
/// queue.
/// \param [in] queue_ptr A pointer to the SYCL queue on which the commands
/// were recorded.
/// \param [out] graph A pointer to a command_graph_ptr pointer where the
/// command graph will be assigned.
static inline void end_recording(sycl::queue *queue_ptr,
                                 dpct::experimental::command_graph_ptr *graph) {
  detail::graph_mgr::instance().end_recording(queue_ptr, graph);
}

/// Adds an empty node to the command graph with optional
/// dependencies.
/// \param [out] newNode A pointer to the node_ptr that will be
/// added to the graph.
/// \param [in] graph A pointer to the command graph.
/// \param [in] dependenciesArray An array of node pointers
/// representing the dependencies of the new node.
/// \param [in] numberOfDependencies The number of dependencies in
/// the dependenciesArray.
static void
add_empty_node(dpct::experimental::node_ptr *newNode,
               dpct::experimental::command_graph_ptr graph,
               const dpct::experimental::node_ptr *dependenciesArray,
               std::size_t numberOfDependencies) {
  if (numberOfDependencies == 0) {
    *newNode = new sycl::ext::oneapi::experimental::node(graph->add());
    return;
  }
  std::vector<sycl::ext::oneapi::experimental::node> dependencies;
  for (std::size_t i = 0; i < numberOfDependencies; i++) {
    dependencies.push_back(*dependenciesArray[i]);
  }
  *newNode =
      new sycl::ext::oneapi::experimental::node(graph->add(sycl::property_list{
          sycl::ext::oneapi::experimental::property::node::depends_on(
              dependencies)}));
}

/// Adds dependencies between nodes in the command graph.
/// \param [in] graph A pointer to the command graph.
/// \param [in] fromNodes An array of node pointers representing
/// the source nodes.
/// \param [in] toNodes An array of node pointers representing
/// the destination nodes.
/// \param [in] numberOfDependencies The number of dependencies
/// to be added.
static void add_dependencies(dpct::experimental::command_graph_ptr graph,
                             const dpct::experimental::node_ptr *fromNodes,
                             const dpct::experimental::node_ptr *toNodes,
                             std::size_t numberOfDependencies) {
  for (std::size_t i = 0; i < numberOfDependencies; i++) {
    graph->make_edge(*fromNodes[i], *toNodes[i]);
  }
}

/// Gets the nodes in the command graph.
/// \param [in] graph A pointer to the command graph.
/// \param [out] nodesArray An array of node pointers where the
/// nodes will be assigned.
/// \param [out] numberOfNodes The number of nodes in the graph.
static void get_nodes(dpct::experimental::command_graph_ptr graph,
                      dpct::experimental::node_ptr *nodesArray,
                      std::size_t *numberOfNodes) {
  detail::graph_mgr::instance().get_nodes(graph, nodesArray, numberOfNodes);
}

/// Gets the root nodes in the command graph.
/// \param [in] graph A pointer to the command graph.
/// \param [out] nodesArray An array of node pointers where the
/// root nodes will be assigned.
/// \param [out] numberOfNodes The number of root nodes in the graph.
static void get_root_nodes(dpct::experimental::command_graph_ptr graph,
                           dpct::experimental::node_ptr *nodesArray,
                           std::size_t *numberOfNodes) {
  detail::graph_mgr::instance().get_root_nodes(graph, nodesArray,
                                               numberOfNodes);
}

} // namespace experimental
} // namespace dpct
