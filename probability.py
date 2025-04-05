import numpy as np
from scipy import stats

class Probability:
    @staticmethod
    def calculate_probability(favorable_outcomes, total_outcomes):
        """Calculate simple probability"""
        return favorable_outcomes / total_outcomes

    @staticmethod
    def calculate_conditional_probability(joint_prob, marginal_prob):
        """Calculate conditional probability P(A|B) = P(A∩B)/P(B)"""
        return joint_prob / marginal_prob

    @staticmethod
    def binomial_probability(n, k, p):
        """Calculate binomial probability"""
        return stats.binom.pmf(k, n, p)

    @staticmethod
    def normal_probability(x, mean, std_dev):
        """Calculate normal distribution probability"""
        return stats.norm.pdf(x, mean, std_dev)

    @staticmethod
    def poisson_probability(k, lambda_param):
        """Calculate Poisson probability"""
        return stats.poisson.pmf(k, lambda_param)

    @staticmethod
    def generate_random_probability(size=1):
    
    @staticmethod
    def confidence_interval(data, confidence=0.95):
        """Calculate confidence interval for a dataset"""
        mean = np.mean(data)
        std_err = stats.sem(data)
        interval = stats.t.interval(confidence, len(data)-1, loc=mean, scale=std_err)
        return interval
    def confidence_interval(data, confidence=0.95):
        """Calculate confidence interval for a dataset"""
        mean = np.mean(data)
        
        