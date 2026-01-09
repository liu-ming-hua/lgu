from collections import defaultdict
import logging
import pickle
import numpy as np

from uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT35
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4Turbo
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentLlama

from uncertainty.uncertainty_measures.logical_graph import *



from uncertainty.utils import utils

utils.setup_logger()

def compute_root_entropy(lge_graph):
    # Adjust alpha to 0 to compute root entropy
    lge_graph.alpha = 0.0
    logging.info(f"Set alpha to {lge_graph.alpha} for root entropy computation.")

    # Compute root probabilities and entropy
    root_probs = lge_graph.compute_root_probabilities()
    probs_array = np.array(list(root_probs.values()))
    normalized_probs = probs_array / probs_array.sum()
    logging.info(f"Root probabilities: {normalized_probs}")

    root_entropy = -np.sum(normalized_probs * np.log(normalized_probs + 1e-12))
    logging.info(f"Computed root entropy: {root_entropy:.4f}")

    return root_entropy

def compute_ige(lge_graph):
    # Adjust alpha to 0 to compute root entropy
    lge_graph.alpha = 1.0
    logging.info(f"Set alpha to {lge_graph.alpha} for root entropy computation.")

    # Compute root probabilities and entropy
    root_probs = lge_graph.compute_root_probabilities()
    probs_array = np.array(list(root_probs.values()))
    normalized_probs = probs_array / probs_array.sum()
    logging.info(f"Root probabilities: {normalized_probs}")

    ige = -np.sum(normalized_probs * np.log(normalized_probs + 1e-12))
    logging.info(f"Computed root entropy: {ige:.4f}")

    return ige

def compute_lgu_edges_density(ige, lge_graph):
    root_node = lge_graph.get_roots()
    if len(root_node) == 1:
        lgu_edges_density = ige
        lgu_edges_density_weighted = ige
        logging.info("Only one root node, no contradictions to consider.")
    else:
        logging.info(f"Root nodes: {root_node}")
        contradiction_list = []

        # Calculate contradictions edges between root nodes
        for rn in root_node:
            for ln in root_node:
                if rn == ln:
                    continue
                if lge_graph.nli_matrix[rn][ln] == 0:
                    contradiction_list.append( lge_graph.probs_matrix[rn, ln] )
        correlation = len(contradiction_list) / (len(root_node) * (len(root_node) - 1))
        weighted_correlation = np.sum(contradiction_list) / (len(root_node) * (len(root_node) - 1))
        logging.info(f"Correlation: {correlation:.4f}, Weighted correlation: {weighted_correlation:.4f}")

        # Adjusted IGE with edges density
        lgu_edges_density = ige * (1 + correlation)
        lgu_edges_density_weighted = ige * (1 + weighted_correlation)
        logging.info(f"LGU with edges density: {lgu_edges_density:.4f}, LGU with edges density weighted: {lgu_edges_density_weighted:.4f}")

    return lgu_edges_density, lgu_edges_density_weighted

def compute_lgu_average_degree(ige, lge_graph):
    root_node = lge_graph.get_roots()
    # Excalibrate root entropy based on contradictions between root nodes
    if len(root_node) == 1:
        lgu_avg_degree = ige
        logging.info("Only one root node, no contradictions to consider.")
    else:
        # Calculate in-degree and out-degree correlations based on contradictions
        logging.info(f"Root nodes: {root_node}")
        edges_num = 0
        for rn in root_node:
            for ln in root_node:
                if rn == ln:
                    continue
                if lge_graph.nli_matrix[rn][ln] == 0:
                    edges_num += 1
        avg_degree_correlation = edges_num / len(root_node)
        logging.info(f"Average degree correlation: {avg_degree_correlation:.4f}")
        lgu_avg_degree = ige * (1 + avg_degree_correlation)
        logging.info(f"LGU with average degree: {lgu_avg_degree:.4f}")

    return lgu_avg_degree

def compute_lgu_estrada(ige, lge_graph):
    root_node = lge_graph.get_roots()
    # Excalibrate root entropy based on contradictions between root nodes
    if len(root_node) == 1:
        lgu_estrada = ige
        logging.info("Only one root node, no contradictions to consider.")
    else:
        # Calculate Estrada index based on contradictions
        logging.info(f"Root nodes: {root_node}")
        adjacency_matrix = np.zeros((len(root_node), len(root_node)))
        for i in range(len(root_node)):
            for j in range(i, len(root_node)):
                if i == j:
                    continue
                rn = root_node[i]
                ln = root_node[j]
                if lge_graph.nli_matrix[rn][ln] == 0:
                    adjacency_matrix[i][j] = 1
                    adjacency_matrix[j][i] = 1
        logging.info(f"Adjacency matrix for Estrada index:\n{adjacency_matrix}")

        # Compute Estrada index
        eigenvalues = np.linalg.eigvals(adjacency_matrix)
        logging.info(f"Eigenvalues: {eigenvalues}")
        eigenvalues = np.real(eigenvalues)
        estrada_index = np.sum(np.exp(eigenvalues))
        logging.info(f"Estrada index: {estrada_index:.4f}")
        lgu_estrada = ige * (1 + estrada_index)
        logging.info(f"LGU with Estrada index: {lgu_estrada:.4f}")

    return lgu_estrada

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, required=True, help='Run ID for the experiment')
    parser.add_argument('--entailment_model', type=str, default='deberta', help='Entailment model to use: deberta, gpt-4, gpt-3.5, gpt-4-turbo, llama-xxx')
    parser.add_argument('--compute_predictive_entropy', default=True)
    parser.add_argument('--num_eval_samples', type=int, default=1000)
    parser.add_argument('--use_all_generations', default=True,)
    parser.add_argument('--use_num_generations', type=int, default=-1)
    parser.add_argument('--condition_on_question', action='store_true')
    parser.add_argument('--strict_entailment', default=True)
    args = parser.parse_args()

    logging.info("Args: %s", args)

    result_dict = dict()
    result_dict['semantic_ids'] = []

    validation_path = f"/root/lmh/LGU/result/10_answer/{args.run_id}/files/validation_generations.pkl"
    result_path = f"/root/lmh/LGU/result/10_answer/{args.run_id}/files/lgu_measures.pkl"
    with open(validation_path, 'rb') as infile:
        validation_generations = pickle.load(infile)

    entropies = defaultdict(list)
    validation_is_true, validation_answerable = [], []
    count = 0
    
    if args.entailment_model == 'deberta':
        entailment_model = EntailmentDeberta(None, False)
    elif args.entailment_model == 'gpt-4':
        entailment_model = EntailmentGPT4(None, False)
    elif args.entailment_model == 'gpt-3.5':
        entailment_model = EntailmentGPT35(None, False)
    elif args.entailment_model == 'gpt-4-turbo':
        entailment_model = EntailmentGPT4Turbo(None, False)
    elif 'llama' in args.entailment_model.lower():
        entailment_model = EntailmentLlama(None, False, args.entailment_model)
    else:
        raise ValueError
    logging.info('Entailment model loading complete.')
    logical_question_number = 0

    for idx, tid in enumerate(validation_generations):

        example = validation_generations[tid]
        question = example['question']
        context = example['context']
        full_responses = example["responses"]
        most_likely_answer = example['most_likely_answer']

        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            responses = [fr[0] for fr in full_responses[:args.use_num_generations]]
        else:
            responses = [fr[0] for fr in full_responses]
        validation_is_true.append(most_likely_answer['accuracy'])

        logging.info('validation_is_true: %f', validation_is_true[-1])

        if args.compute_predictive_entropy:
            # Token log likelihoods. Shape = (n_sample, n_tokens)
            if not args.use_all_generations:
                log_liks = [r[1] for r in full_responses[:args.use_num_generations]]
            else:
                log_liks = [r[1] for r in full_responses]

            for i in log_liks:
                assert i

            if args.condition_on_question and args.entailment_model == 'deberta':
                responses = [f'{question} {r}' for r in responses]

            # Compute semantic ids
            semantic_ids = get_semantic_ids(
                responses, model=entailment_model,
                strict_entailment=args.strict_entailment, example=example)

            result_dict['semantic_ids'].append(semantic_ids)

            # Length normalization of generation probabilities.
            log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]

            # Compute semantic entropy.
            unique_ids, log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')

            # Construct Discrete LGE graph
            discrete_lgu_graph = construct_discrete_logical_graph(semantic_ids, responses, entailment_model)
            logging.info(f"Discrete LGE graph nodes: {discrete_lgu_graph.nodes}")
            logging.info(f"Discrete LGE graph edges: {discrete_lgu_graph.edges}")
            
            # Count logical questions (with more than 1 edge in the discrete LGE graph)
            if len(discrete_lgu_graph.edges) > 1:
                logical_question_number += 1

            # Compute Discrete LGE graph entropy series
            discrete_root_entropy = compute_root_entropy(discrete_lgu_graph)
            entropies['discrete_root_entropy'].append(discrete_root_entropy)

            discrete_ige = compute_ige(discrete_lgu_graph)
            entropies['discrete_ige'].append(discrete_ige)

            discrete_lgu_edges_density, discrete_lgu_edges_density_weighted = compute_lgu_edges_density(discrete_ige, discrete_lgu_graph)
            entropies['discrete_lgu_edges_density'].append(discrete_lgu_edges_density)
            entropies['discrete_lgu_edges_density_weighted'].append(discrete_lgu_edges_density_weighted)

            discrete_lgu_avg_degree = compute_lgu_average_degree(discrete_ige, discrete_lgu_graph)
            entropies['discrete_lgu_avg_degree'].append(discrete_lgu_avg_degree)

            discrete_lgu_estrada = compute_lgu_estrada(discrete_ige, discrete_lgu_graph)
            entropies['discrete_lgu_estrada'].append(discrete_lgu_estrada)
  
            
            # Constract LGE graph 
            lgu_graph = construct_logical_graph(semantic_ids, log_likelihood_per_semantic_id, responses, entailment_model)

            root_entropy = compute_root_entropy(lgu_graph)
            entropies['root_entropy'].append(root_entropy)

            ige = compute_ige(lgu_graph)
            entropies['ige'].append(ige)

            lgu_edges_density, lgu_edges_density_weighted = compute_lgu_edges_density(ige, lgu_graph)
            entropies['lgu_edges_density'].append(lgu_edges_density)
            entropies['lgu_edges_density_weighted'].append(lgu_edges_density_weighted)

            lgu_avg_degree = compute_lgu_average_degree(ige, lgu_graph)
            entropies['lgu_avg_degree'].append(lgu_avg_degree)

            lgu_estrada = compute_lgu_estrada(ige, lgu_graph)
            entropies['lgu_estrada'].append(lgu_estrada)

        count += 1
        if count >= args.num_eval_samples:
            break

    validation_is_false = [1.0 - is_t for is_t in validation_is_true]
    result_dict['validation_is_false'] = validation_is_false
    validation_unanswerable = [1.0 - is_a for is_a in validation_answerable]
    result_dict['validation_unanswerable'] = validation_unanswerable
    result_dict['logical_question_number'] = logical_question_number

    # Save results
    if 'uncertainty_measures' not in result_dict:
        result_dict['uncertainty_measures'] = dict()
    if args.compute_predictive_entropy:
        result_dict['uncertainty_measures'].update(entropies)

    with open(result_path, 'wb') as f:
        pickle.dump(result_dict, f)
    print(f'Saved to {result_path}')

if __name__ == '__main__':
    main()
