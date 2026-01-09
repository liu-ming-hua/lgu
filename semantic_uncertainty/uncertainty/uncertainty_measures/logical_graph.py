import logging
import numpy as np
import torch
from collections import defaultdict
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def log_to_probs_sum_normalized(log_likelihoods):
    # sum_normalized
    log_lik_norm = log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
    probs = np.exp(log_lik_norm)
    return probs

class LogicalGraph():
    def __init__(self, strings, class_probs, entail_model):
        self.nodes = {i: class_probs[i] for i in range(len(class_probs))}
        self.entail_model = entail_model
        self.strings_list = strings
        self.nli_matrix = np.zeros((len(self.nodes), len(self.nodes)), dtype=int)
        self.probs_matrix = np.zeros((len(self.nodes), len(self.nodes)))
        self.alpha = 1
        self.edges = {list(self.nodes.keys())[i]: [] for i in range(len(self.nodes))}
        self.build_edges()

    def build_edges(self):
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if i == j:
                    continue
                entail, prob = self.entail_model.check_implication(self.strings_list[i], self.strings_list[j])
                self.nli_matrix[i, j] = entail
                if entail == 2:
                    self.edges[i].append(j)
                if entail == 0:                    
                    self.probs_matrix[i, j] = prob

    def find_cycle(self):
        """Return a list of nodes in a cycle, or [] if no cycle."""
        visited = set()
        rec_stack = []

        def dfs(node):
            visited.add(node)
            rec_stack.append(node)
            for neighbor in self.edges.get(node, []):
                if neighbor not in visited:
                    cycle = dfs(neighbor)
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = rec_stack.index(neighbor)
                    return rec_stack[cycle_start:]
            rec_stack.pop()
            return []

        for node in self.nodes.keys():
            if node not in visited:
                cycle = dfs(node)
                if cycle:
                    return cycle
        return []

    def fix_cycle(self):
        """Detect and fix one cycle by keeping the node with most external edges.
        The probabilities of removed nodes are merged into the kept node.
        """
        cycle_nodes = self.find_cycle()
        if not cycle_nodes:
            return False  # No cycle found

        logging.warning(f"Cycle detected: {cycle_nodes}")

        # Count external edges for each cycle node
        external_edge_counts = {}
        for node in cycle_nodes:
            count = 0
            # outgoing edges to outside
            for nb in self.edges.get(node, []):
                if nb not in cycle_nodes:
                    count += 1
            # incoming edges from outside
            for other, nbs in self.edges.items():
                if other not in cycle_nodes and node in nbs:
                    count += 1
            external_edge_counts[node] = count

        # Choose node to keep
        keep_node = max(external_edge_counts, key=external_edge_counts.get)
        logging.info(f"Keeping node {keep_node} (max external edges: {external_edge_counts[keep_node]})")

        # Merge probabilities from deleted nodes into keep_node
        for node in cycle_nodes:
            if node == keep_node:
                continue
            # add probability
            self.nodes[keep_node] += self.nodes[node]
            # remove node
            del self.nodes[node]
            # remove outgoing edges
            self.edges[node] = []
            # remove incoming edges
            for other, nbs in self.edges.items():
                if node in nbs:
                    nbs.remove(node)

        return True
    
    def get_roots(self):
        """Return all nodes with no incoming edges (roots)."""
        all_nodes = set(self.nodes.keys())
        non_roots = {nb for nbs in self.edges.values() for nb in nbs}
        roots = list(all_nodes - non_roots)
        return roots

    def compute_root_probabilities(self):
        """Compute probabilities only for root nodes by summing over their leaf descendants * alpha."""
        root_probs = {}

        def dfs(node):
            
            if not self.edges.get(node):
                return self.nodes[node]
            total = self.nodes[node]
            for child in self.edges[node]:
                total += self.alpha * dfs(child)
            return total

        for root in self.get_roots():
            root_probs[root] = dfs(root)

        return root_probs
    
    def get_reachable_from_roots(self):
        """Return a dict: {root_node: set of reachable nodes (including itself)}."""
        roots = self.get_roots()
        reachable = {}
        # DFS from each root
        def dfs(node, visited):
            visited.add(node)
            for child in self.edges.get(node, []):
                if child not in visited:
                    dfs(child, visited)

        for root in roots:
            visited = set()
            dfs(root, visited)
            reachable[root] = visited

        return reachable
    
def construct_logical_graph(semantic_ids, log_likelihood_per_semantic_id, strings_list, model):
    # Compute probabilities from log likelihoods
    probs = log_to_probs_sum_normalized(np.array(log_likelihood_per_semantic_id))
    logging.info(f"Probabilities per semantic cluster: {probs}")
    unique_ids = sorted(set(semantic_ids))
    cluster_string_list = []
    id_to_string = {}
    for idx, sid in enumerate(semantic_ids):
        if sid not in id_to_string:
            id_to_string[sid] = strings_list[idx]
    for sid in unique_ids:
        cluster_string_list.append(id_to_string[sid])
    logging.info(f"Representative strings per semantic cluster: {cluster_string_list}")

    graph = LogicalGraph(cluster_string_list, probs, model)
    logging.info(f"Initial graph nodes: {graph.nodes}")
    logging.info(f"Initial graph edges: {graph.edges}")
    while graph.fix_cycle(): 
        pass

    return graph

def construct_discrete_logical_graph(semantic_ids, strings_list, model):
    # Count occurrences for each cluster id
    unique_ids = sorted(set(semantic_ids))
    probs = [0.0 for _ in unique_ids]
    cluster_string_list = []
    id_to_string = {}
    for idx, sid in enumerate(semantic_ids):
        probs[sid] += 1 / len(semantic_ids)
        if sid not in id_to_string:
            id_to_string[sid] = strings_list[idx]
    # Build cluster_string_list in order of unique_ids
    for sid in unique_ids:
        cluster_string_list.append(id_to_string[sid])
    
    logging.info(f"Representative strings per semantic cluster: {cluster_string_list}")
    logging.info(f"Probabilities per semantic cluster: {probs}")

    graph = LogicalGraph(cluster_string_list, probs, model)
    logging.info(f"Initial graph nodes: {graph.nodes}")
    logging.info(f"Initial graph edges: {graph.edges}")
    while graph.fix_cycle(): 
         pass

    return graph