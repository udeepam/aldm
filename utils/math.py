import numpy as np
import torch
import pycave


def explained_variance(y_pred, y):
    """
    Taken from: https://github.com/openai/baselines

    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    var_y = torch.var(y)
    return np.nan if var_y==0 else 1 - torch.var(y-y_pred)/var_y


def safe_mean(xs):
    """
    Taken from: https://github.com/openai/baselines

    Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
    """
    return np.nan if len(xs) == 0 else np.mean(xs)


def safe_divide(top, bottom):
    """
    Avoid division error.
    """
    return np.nan if bottom == 0 else top/bottom


def kl_divergence(p, q, num_samples=10**3):
    """
    TODO: fix pycave sampling to keep gradients

    Reference: https://stackoverflow.com/questions/26079881/kl-divergence-of-two-gmms

    KL(p||q) = int p(x) log[p(x) / q(x)] dx = E_p[ log(p(x) / q(x)) ]

    Parameters:
    -----------
    p : `torch.distributions` or `pycave.bayes.GMM`
    q : `torch.distributions` or `pycave.bayes.GMM`
    """
    if isinstance(p, pycave.bayes.GMM):
        x = p.sample(num_samples)
        log_p_x = -p.evaluate(x, reduction='none')
        log_q_x = -q.evaluate(x, reduction='none')
    else:
        # torch.distributions object
        x = p.rsample((num_samples,))
        log_p_x = p.log_prob(x)
        log_q_x = q.log_prob(x)
    return log_p_x.mean() - log_q_x.mean()


def js_divergence(p, q, num_samples=10**3):
    """
    TODO: fix pycave sampling to keep gradients

    Reference: https://stackoverflow.com/questions/26079881/kl-divergence-of-two-gmms

    JS(p||q) = 0.5 * [ KL(p||M) + KL(q||M) ]
    where M = 0.5 * ( p + q )

    Parameters:
    -----------
    p : `torch.distributions` or `pycave.bayes.GMM`
    q : `torch.distributions` or `pycave.bayes.GMM`
    """
    if isinstance(p, pycave.bayes.GMM):
        x = p.sample(num_samples)
        log_p_x = -p.evaluate(x, reduction='none')
        log_q_x = -q.evaluate(x, reduction='none')

        x = q.sample(num_samples)
        log_p_y = -p.evaluate(x, reduction='none')
        log_q_y = -q.evaluate(x, reduction='none')
    else:
        # torch.distributions object
        x = p.rsample((num_samples,))
        log_p_x = p.log_prob(x)
        log_q_x = q.log_prob(x)

        x = q.rsample((num_samples,))
        log_p_y = p.log_prob(x)
        log_q_y = q.log_prob(x)

    log_m_x = torch.logsumexp(torch.stack([log_p_x, log_q_x]), dim=0)
    log_m_y = torch.logsumexp(torch.stack([log_p_y, log_q_y]), dim=0)
    return (log_p_x.mean() - (log_m_x.mean()-torch.tensor(2.).log()) + log_q_y.mean() - (log_m_y.mean()-torch.tensor(2.).log())) / 2
