import numpy as np

def mean_token_entropy(token_log_likelihoods):
    """
    Calculate the mean token entropy for a single answer (Mean Token Entropy).
    Args:
        token_log_likelihoods: List[float] or np.ndarray
            Each row is the log probability distribution (log probabilities after log_softmax) for a token.
    Returns:
        float, the average value of all token entropies
    """
    token_log_likelihoods = np.array(token_log_likelihoods)  
    probs = np.exp(token_log_likelihoods)
    entropy_per_token = -np.sum(probs * token_log_likelihoods) 
    mean_entropy = entropy_per_token / len(token_log_likelihoods)
    return mean_entropy

def avg_neg_logp(token_log_likelihoods):
    """
    Calculate Avg(-logp): average negative log probability per token.
    Args:
        token_log_likelihoods: List[float] or np.ndarray
            Log probability for each token (shape=[number of tokens])
    Returns:
        float, average negative log probability
    """
    token_log_likelihoods = np.array(token_log_likelihoods)
    avg_nll = -np.mean(token_log_likelihoods)
    return avg_nll

def max_neg_logp(token_log_likelihoods):
    """
    Calculate Max(-logp): maximum negative log probability among tokens.
    Args:
        token_log_likelihoods: List[float] or np.ndarray
            Log probability for each token (shape=[number of tokens])
    Returns:
        float, maximum negative log probability
    """
    token_log_likelihoods = np.array(token_log_likelihoods)
    max_nll = -np.min(token_log_likelihoods)
    return max_nll

def avg_entropy(token_log_likelihoods):
    """
    Calculate Avg(H): average entropy per token.
    Args:
        token_log_likelihoods: np.ndarray, shape=[num_tokens, vocab_size]
            Each row is the log probability distribution for a token.
    Returns:
        float, average entropy
    """
    token_log_likelihoods = np.array(token_log_likelihoods)
    probs = np.exp(token_log_likelihoods)
    # entropy for each token: -sum(p * logp) over vocab
    entropies = -probs * token_log_likelihoods
    avg_H = np.mean(entropies)
    return avg_H

def max_entropy(token_log_likelihoods):
    """
    Calculate Max(H): maximum entropy among tokens.
    Args:
        token_log_likelihoods: np.ndarray, shape=[num_tokens, vocab_size]
            Each row is the log probability distribution for a token.
    Returns:
        float, maximum entropy
    """
    token_log_likelihoods = np.array(token_log_likelihoods)
    probs = np.exp(token_log_likelihoods)
    entropies = -probs * token_log_likelihoods
    max_H = np.max(entropies)
    return max_H


def maximum_sequence_probability(token_log_likelihoods):
    """
    Calculate the Maximum Sequence Probability (MSP) for a single answer.
    Args:
        token_log_likelihoods: List[float] or np.ndarray
            Log probability for each token (usually the log probability of the generated sequence, shape=[number of tokens])
    Returns:
        float, U_MSP = 1 - P(y|x)
    """
    # Sum all token log probabilities to get log P(y|x)
    log_p_y_given_x = np.sum(token_log_likelihoods)
    p_y_given_x = np.exp(log_p_y_given_x)
    u_msp = 1 - p_y_given_x
    return u_msp



def perplexity(token_log_likelihoods):
    """
    Calculate the Perplexity (PPL) for a single answer.
    Args:
        token_log_likelihoods: List[float] or np.ndarray
            Log probability for each token (shape=[number of tokens])
    Returns:
        float, PPL(y|x)
    """
    log_p_y_given_x = np.sum(token_log_likelihoods)
    L = len(token_log_likelihoods)
    avg_nll = -log_p_y_given_x / L
    ppl = np.exp(avg_nll)
    return ppl


