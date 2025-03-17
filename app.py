import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Title
st.title("Central Limit Theorem Demonstration")

# Sidebar settings
distribution = st.sidebar.selectbox('Distribution', ['Uniform', 'Exponential', 'Binomial', 'Poisson'])
n = st.sidebar.slider('Sample Size (n)', 1, 100, 30)
N = st.sidebar.slider('Number of Samples (N)', 100, 5000, 1000)

# Distribution parameters
if distribution == 'Uniform':
    param1 = st.sidebar.number_input('Lower Bound', value=0.0)
    param2 = st.sidebar.number_input('Upper Bound', value=1.0)
    data = np.random.uniform(param1, param2, size=(N, n))
    mu, sigma = (param1 + param2)/2, (param2 - param1)/np.sqrt(12)

elif distribution == 'Exponential':
    param1 = st.sidebar.number_input('Scale', value=1.0)
    data = np.random.exponential(param1, size=(N, n))
    mu, sigma = param1, param1

elif distribution == 'Binomial':
    param1 = st.sidebar.number_input('Number of Trials', value=10, step=1)
    param2 = st.sidebar.slider('Probability', 0.0, 1.0, 0.5)
    data = np.random.binomial(param1, param2, size=(N, n))
    mu, sigma = param1 * param2, np.sqrt(param1 * param2 * (1-param2))

elif distribution == 'Poisson':
    param1 = st.sidebar.number_input('Lambda (Rate)', value=3.0)
    data = np.random.poisson(param1, size=(N, n))
    mu, sigma = param1, np.sqrt(param1)

sample_means = data.mean(axis=1)

# Plot population and sample means
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Population histogram
ax1 = plt.subplot(1,2,1)
plt.hist(data.flatten(), bins=30, density=True, alpha=0.6, color='skyblue')
plt.title('Population Distribution')
plt.xlabel('Value')
plt.ylabel('Density')

# Sample mean distribution
plt.subplot(1,2,2)
plt.hist(sample_means, bins=30, density=True, alpha=0.6, color='lightgreen', label='Sample Mean')

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, sigma/np.sqrt(n))
plt.plot(x, p, 'r--', linewidth=2, label='Normal Distribution')

plt.title('Sample Mean Distribution')
plt.xlabel('Sample Mean')
plt.ylabel('Density')
plt.legend()

st.pyplot(fig=plt)

st.write('### Statistical Summary')
st.write(f'Theoretical Mean: {mu:.4f}, Sample Mean: {sample_means.mean():.4f}')
st.write(f'Theoretical Std. Dev.: {sigma/np.sqrt(n):.4f}, Sample Std Dev: {sample_means.std():.4f}')
