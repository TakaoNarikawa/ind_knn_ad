from scipy.stats import norm

def norm_ppf(mu, var, percent=0.95):
    return norm.ppf(q=percent, loc=mu, scale=var**0.5)

def norm_cdf(mu, var, x):
    return norm.cdf(x=x, loc=mu, scale=var)

if __name__ == '__main__':

    out = norm_ppf(7.033514976501465, 0.10828708857297897)
    per = norm_cdf(9.476301193237305, 1.8827488422393799, out)
    print("out", out, per)