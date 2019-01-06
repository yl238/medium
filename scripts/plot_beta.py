import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
import seaborn as sns
sns.set(style='white', font='consolas', font_scale=1.2)


def bernoulli_entropy_fig():
    """Simple illustration of entropy for the Bernoulli distribution"""
    x = np.linspace(0.001, 0.999, 100)
    y = -x*np.log2(x) - (1-x)*np.log2(1-x)
    
    f, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, y)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticklabels(['0', '0.5', '1'])
    ax.set_yticklabels(['0', '0.5', '1'])
    ax.set_xlabel("P(X)=1")
    ax.set_ylabel("H(X)")
                    
def beta_distribution():
    """Examples of Beta distribution with varying alpha and beta parameters."""
    x = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(6, 4))
    rv = beta.pdf(x, 0.5, 0.5)
    ax.plot(x, rv, label='a=0.5, b=0.5')
    rv = beta.pdf(x, 1, 1)
    ax.plot(x, rv, label='a=1, b=1')
    rv = beta.pdf(x, 2, 3)
    ax.plot(x, rv, label='a=2, b=3')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3)
    ax.set_title('beta distributions')
    plt.legend()
    f.savefig('../figures/beta_distribution.png')

if __name__ == '__main__':
    beta_distribution()
