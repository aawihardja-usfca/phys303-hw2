# PHYS303 Homework 2: Bayesian Monte Carlo Sampling &amp; Uncertainty

1. I generated the prior array and likelihood array in the following manner:
   ```python
   prior_arr = np.append(np.repeat(False, 990), np.repeat(True, 10))
   likelihood_arr = np.append(np.repeat(True, 95), np.repeat(False, 5))
   ```
2. For Q3, I created a function that samples from the prior_arr and likelihood_arr.Below is the signature:
   ```python
   def simulate_pop(prior, likelihood, likelihood_alt=None, count=1000) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a sample population by picking a random value of (ground truth label and predicted label) for each individual.
    
    Args:
        prior: the prior to draw population from
        likelihood: likelihood of testing positive, given the person has the disease
        likelihood_alt: likelihood of testing negative, given the person doesn't have the disease
        count: sample count
        
    Returns:
        tuple of (ground_truth, predicted_label)
    """
   ```
4. I made another function to count the amount of true positive & false positive, like so:
   ```python
   def count_tp_fp(ground_truth: np.ndarray, pred: np.ndarray) -> tuple[int, int]:
    """
    Count the true_positive and false_positive, given ground truth and predicted label
    
    Args:
        ground_truth: the ground truth label
        pred: predicted label
        
    Returns:
        tuple of (true positive, false positive)
    """
   ```
5. For Q4 I picked a new sample population (of size 1000) and counted tp & fp at each iteration of k:
   ```python
   def repeat_sampling(k, prior, likelihood, likelihood_alt=None, sample_count=1000):
    priors = np.array([])

    for i in range(k):
        samples, test_results = simulate_pop(prior, likelihood, likelihood_alt, sample_count)
        tp, fp = count_tp_fp(samples, test_results)
        new_prior = tp / (tp + fp)
        priors = np.append(priors, new_prior)
        
    return priors
   ```
   I assume this is where the uncertainty is introduced; the value of the prior, tpr, and fpr is different each time.
