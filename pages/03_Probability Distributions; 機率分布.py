import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, poisson

# Streamlit page configuration
st.set_page_config(
    page_title="Probability Distributions", 
    page_icon="📊",
    layout="wide", 
    initial_sidebar_state="auto",
    menu_items={
        'About': "This work authored by Teddy (Yen-Hsiang Huang) from the NCHU, Taiwan.",
        'Get Help': 'https://github.com/TeddYenn/stat2vis',
        'Report a bug': "https://github.com/TeddYenn/stat2vis/issues"
    })

# App header and intro
st.title("Ch. 3: Probability Distributions  |  機率分布")
st.caption("**Developed by Teddy (Yen-Hsiang Huang)**")

st.write("Through EDA techniques, we have gained a foundational understanding of basic data analysis. In this chapter, we will explore probability distributions, which serve as the foundation for statistical modeling.")
st.write("透過探索式資料分析技術，我們已了解基礎的資料分析。此章節探索不同的機率分布，將是後續建構統計模型的基礎。")

# Section: Distribution Selector
st.write("")
st.write("")
st.write("### 1️⃣ Select a Distribution  |  選擇分布")

dist_category = st.radio("Distribution Type:", ["Discrete Distribution （離散分布)", "Continuous Distribution (連續分布)"])

if dist_category == "Discrete Distribution （離散分布)":
    dist_type = st.selectbox("Discrete Distributions:", 
        ["Binomial Distribution (二項分布)", 
         "Hypergeometric Distribution (超幾何分布)",
         "Geometric Distribution (幾何分布)",
         "Negative Binomial Distribution (負二項分布)",
         "Poisson Distribution (卜瓦松分布)",
         "Multinomial Distribution (多項分布)"])

elif dist_category == "Continuous Distribution (連續分布)":
    dist_type = st.selectbox("Continuous Distributions:", 
        ["Uniform Distribution (均勻分布)", #
         "Normal Distribution (常態分布)", #
         "Exponential Distribution (指數分布)", #
         "Gamma Distribution (Gamma 分布)",
         "Chi-square Distribution (卡方分布)",
         "Student's t-distribution (t 分布)",
         "F Distribution (F 分布)",
         "Beta Distribution (Beta 分布)"
         ])

st.write("")

# ==========================================
# Normal Distribution
# ==========================================

if dist_type == "Normal Distribution (常態分布)":
    st.subheader("📈 Normal Distribution | 常態分布")

    # --- Description and formula ---
    st.markdown("""
                The normal distribution is the most common continuous probability distribution.  
                It is characterized by its symmetry, unimodal shape, and bell-like curve.
                
                Its PDF is given by:
                """)

    st.latex(r'''
             f(x \mid \mu, \sigma) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
             ''')

    # --- Symbol explanation ---
    st.markdown("""
                - $x$: random variable  
                - $\\mu$: mean  
                - $\\sigma$: standard deviation  
                """)

    st.markdown("---")
    st.write("### 2️⃣ Make a Plot  |  作圖")

    # --- User input for parameters ---
    col1, col2 = st.columns(2)
    with col1:
        mean = st.slider("Mean (μ)", -100.0, 100.0, 10.0, 10.0)
        std = st.slider("Standard Deviation（σ）", 0.1, 10.0, 1.0, 0.1)

    # --- PDF calculation and plotting values ---
    x = np.linspace(mean - 4*std, mean + 4*std, 500)
    y = norm.pdf(x, mean, std)

    # --- Simulate data points (samples) for visualization ---
    data_points = np.random.normal(loc=mean, scale=std, size=300)
    data_y = norm.pdf(data_points, mean, std)

    # --- Plotting PDF and elements ---
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(x, y, label="PDF", color="blue")
    ax.scatter(data_points, data_y, color='grey', zorder=5, label="Data Points", s=8)

    # --- Mean and 95% interval ---
    ax.axvline(mean, color="red", linestyle="--", label=f"Mean = {mean}")
    z = 1.96
    lower = mean - z * std
    upper = mean + z * std
    ax.axvline(lower, color="green", linestyle=":", label=f"Lower 95% ≈ {lower:.2f}")
    ax.axvline(upper, color="green", linestyle=":", label=f"Upper 95% ≈ {upper:.2f}")

    # --- Highlight 95% area ---
    x_fill = np.linspace(lower, upper, 300)
    y_fill = norm.pdf(x_fill, mean, std)
    ax.fill_between(x_fill, y_fill, alpha=0.3, color='gray', label="~95% Area")

    ax.set_title("Normal Distribution")
    ax.legend(fontsize="small")
    st.pyplot(fig, use_container_width=False)

# ==========================================
# Uniform Distribution
# ==========================================
elif dist_type == "Uniform Distribution (均勻分布)":
    st.subheader("📈 Uniform Distribution | 均勻分布")

    # --- Description and PDF formula ---
    st.markdown("""
                The continuous uniform distribution models equal probability across an interval \\([a, b]\\).  
                It is flat and non-peaked, meaning each value in the interval is equally likely.
                
                Its PDF is given by:
                """)

    st.latex(r'''
             f(x) = 
            \begin{cases}
            \frac{1}{b - a}, & a \leq x \leq b \\
            0, & \text{otherwise}
            \end{cases}
            ''')

    # --- Symbol explanation ---
    st.markdown("""
                - $x$: random variable  
                - $a$: lower bound  
                - $b$: upper bound  
                """)

    # --- Mean and Standard Deviation ---
    st.markdown("The mean and standard deviation are:")
    st.latex(r'''
             \mu = \frac{a + b}{2}, \quad \sigma = \sqrt{\frac{(b - a)^2}{12}}
             ''')

    st.markdown("---")
    st.write("### 2️⃣ Make a Plot  |  作圖")

    # --- User input for interval [a, b] ---
    col1, col2 = st.columns(2)
    with col1:
        a = st.slider("Lower Bound (a)", -100.0, 100.0, 0.0, 10.0)
        b = st.slider("Upper Bound (b)", a + 10.0, a + 200.0, a + 10.0, 10.0)  # Ensure b > a

    # --- PDF data generation ---
    x = np.linspace(a - (b - a) * 0.2, b + (b - a) * 0.2, 500)
    y = np.where((x >= a) & (x <= b), 1 / (b - a), 0)

    # --- Calculate mean and std ---
    mean = (a + b) / 2
    std = np.sqrt((b - a) ** 2 / 12)

    # --- Plotting PDF ---
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hlines(1 / (b - a), xmin=a, xmax=b, colors='blue', label="PDF", linewidth=2)

    # --- Vertical reference lines for a, b, and mean ---
    ax.axvline(a, color="green", linestyle=":", label=f"a = {a}")
    ax.axvline(b, color="green", linestyle=":", label=f"b = {b}")
    ax.axvline(mean, color="red", linestyle="--", label=f"Mean = {mean:.2f}")

    # --- Plot adjustments ---
    ax.set_ylim(0, (1 / (b - a)) * 1.2)
    ax.set_title("Uniform Distribution")
    ax.legend(fontsize="small")

    st.pyplot(fig, use_container_width=False)


# ==========================================
# Exponential Distribution
# ==========================================

elif dist_type == "Exponential Distribution (指數分布)":
    st.subheader("📈 Exponential Distribution | 指數分布")
    
    # --- Description and PDF formula ---
    st.markdown("""
                The exponential distribution models the time between events in a Poisson process.  
                It is right-skewed, non-negative, and memoryless.
                
                Its PDF is given by:
                """)

    st.latex(r'''
             f(x \mid \lambda) = \lambda e^{-\lambda x}, \quad x \geq 0
             ''')

    # --- Symbol explanation ---
    st.markdown("""
                - $x$: random variable  
                - $\\lambda$: rate parameter (events per unit time), where $\\lambda > 0$
                """)

    # --- Mean and standard deviation ---
    st.markdown("The mean and standard deviation are:")
    st.latex(r'''
             \mu = \frac{1}{\lambda}, \quad \sigma = \frac{1}{\lambda}
             ''')

    st.markdown("---")
    st.write("### 2️⃣ Make a Plot  |  作圖")

    # --- Parameter inputs ---
    col1, col2 = st.columns(2)
    with col1:
        lam = st.slider("Rate (λ)", 0.1, 10.0, 1.0, 0.1)  # λ > 0

    # --- PDF calculation ---
    x_max = 20
    x = np.linspace(0, x_max, 500)
    y = lam * np.exp(-lam * x)

    mean = 1 / lam
    std = 1 / lam
    lower = 0
    upper = mean + 2 * std  # Approximate range that covers ~95% for exponential

    # --- Plotting the PDF ---
    fig, ax = plt.subplots(figsize=(7, 3))

    ax.plot(x, y, label="PDF", color="blue")  # Plot the exponential curve

    # --- Reference lines for mean and ~95% range ---
    ax.axvline(mean, color="red", linestyle="--", label=f"Mean = {mean:.2f}")
    ax.axvline(upper, color="green", linestyle=":", label=f"Upper ≈ μ + 2σ ≈ {upper:.2f}")

    # --- Fill between for ~95% area ---
    x_fill = np.linspace(lower, upper, 300)
    y_fill = lam * np.exp(-lam * x_fill)
    ax.fill_between(x_fill, y_fill, alpha=0.3, color='gray', label="~95% Area")

    # --- Final layout ---
    ax.set_title("Exponential Distribution")
    ax.legend(fontsize="small")

    st.pyplot(fig, use_container_width=False)

# ==========================================
# Gamma Distribution
# ==========================================

elif dist_type == "Gamma Distribution (Gamma 分布)":
    st.subheader("📈 Gamma Distribution | Gamma 分布")
    st.markdown("""
                The gamma distribution is a two-parameter continuous distribution that models waiting times and lifetimes of processes.  
                It is right-skewed, flexible in shape, and often used in queuing theory and Bayesian statistics.
                
                Its PDF is given by:
                """)

    st.latex(r'''
             f(x \mid \alpha, \beta) = 
             \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha - 1} e^{-\beta x}, \quad x > 0
             ''')

    st.markdown("""
                - $x$: random variable  
                - $\\alpha$: shape parameter (形狀參數)  
                - $\\beta$: rate parameter (速率參數；$\\theta = 1/\\beta$ 表示為尺度參數)  
                """)

    st.markdown("The mean and standard deviation are:")
    st.latex(r"\mu = \frac{\alpha}{\beta}, \quad \sigma = \frac{\sqrt{\alpha}}{\beta}")

    st.markdown("---")
    st.write("### 2️⃣ Make a Plot  |  作圖")

    # -- User Inputs for shape (α) and rate (β) parameters
    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider("Shape (α)", 0.1, 20.0, 2.0, 1.0)
        beta = st.slider("Rate (β)", 0.1, 10.0, 1.0, 1.0)

    # -- Generate x values and corresponding PDF values
    x = np.linspace(0, 5 * alpha / beta, 500)
    from scipy.stats import gamma
    y = gamma.pdf(x, a=alpha, scale=1/beta)

    # -- Compute mean and standard deviation
    mean = alpha / beta
    std = np.sqrt(alpha) / beta
    lower = mean - 2 * std
    upper = mean + 2 * std

    # -- Plotting
    fig, ax = plt.subplots(figsize=(7, 3))

    ax.plot(x, y, label="PDF", color="blue")

    ax.axvline(mean, color="red", linestyle="--", label=f"Mean ≈ {mean:.2f}")
    ax.axvline(lower, color="green", linestyle=":", label=f"μ - 2σ ≈ {lower:.2f}")
    ax.axvline(upper, color="green", linestyle=":", label=f"μ + 2σ ≈ {upper:.2f}")

    x_fill = np.linspace(lower, upper, 300)
    y_fill = gamma.pdf(x_fill, a=alpha, scale=1/beta)
    ax.fill_between(x_fill, y_fill, alpha=0.3, color='gray', label="~95% Area")

    ax.set_title("Gamma Distribution")
    ax.legend(fontsize="small")

    st.pyplot(fig, use_container_width=False)

# ==========================================
# Chi-square Distribution
# ==========================================

elif dist_type == "Chi-square Distribution (卡方分布)":
    st.subheader("📈 Chi-square Distribution | 卡方分布")

    # --- Description and formula ---
    st.markdown("""
                The chi-square distribution is a continuous distribution defined for non-negative values.  
                It is a special case of the gamma distribution with shape = degrees of freedom / 2, and is commonly used in statistical hypothesis testing.
                """)
    
    st.latex(r'''
             \chi^2_k = \sum_{i=1}^{k} Z_i^2
             \quad \text{where } Z_i \sim \mathcal{N}(0, 1)
             ''')
    
    st.latex(r'''
             \chi^2_k \sim \text{Gamma}\left( \frac{k}{2}, \frac{1}{2} \right)
             ''')
    
    st.markdown("""
                Its PDF is given by:
                """)
    
    st.latex(r'''
             f(x \mid k) = \frac{1}{2^{k/2} \Gamma(k/2)} x^{k/2 - 1} e^{-x/2}, \quad x > 0
             ''')

    # --- Symbol explanation ---
    st.markdown("""
                - $x$: random variable  
                - $k$: degrees of freedom  
                - $\\Gamma$: gamma function  
                """)

    st.markdown("The mean and standard deviation are:")
    st.latex(r'''
             \mu = k, \quad \sigma = \sqrt{2k}
             ''')

    st.markdown("---")
    st.write("### 2️⃣ Make a Plot  |  作圖")

    # --- User input for degrees of freedom ---
    col1, col2 = st.columns(2)
    with col1:
        df = st.slider("Degrees of Freedom (k)", 1, 50, 5, 1)

    # --- PDF calculation and plotting values ---
    x = np.linspace(0, df+50, 500)
    from scipy.stats import chi2
    y = chi2.pdf(x, df)

    mean = df
    std = np.sqrt(2 * df)
    lower = mean - 2 *std
    upper = mean + 2 *std

    # --- Plotting PDF and elements ---
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(x, y, label="PDF", color="blue")

    # --- Mean and ±1σ interval ---
    ax.axvline(mean, color="red", linestyle="--", label=f"Mean = {mean}")
    ax.axvline(lower, color="green", linestyle=":", label=f"μ - 2σ ≈ {lower:.2f}")
    ax.axvline(upper, color="green", linestyle=":", label=f"μ + 2σ ≈ {upper:.2f}")

    # --- Highlight ~95% area around the mean ---
    x_fill = np.linspace(lower, upper, 300)
    y_fill = chi2.pdf(x_fill, df)
    ax.fill_between(x_fill, y_fill, alpha=0.3, color='gray', label="~95% Area")

    ax.set_title("Chi-square Distribution")
    ax.legend(fontsize="small")
    st.pyplot(fig, use_container_width=False)

# --- Student's t-distribution ---

elif dist_type == "Student's t-distribution (t 分布)":
    st.subheader("📈 Student's t-distribution | t 分布")

    # --- Description and formula ---
    st.markdown("""
                The Student’s t-distribution is a symmetric, bell-shaped distribution like the normal distribution, but with heavier tails.  
                It is commonly used in statistical inference, especially for small sample sizes.
                """)
    
    st.latex(r'''
             t = \frac{Z}{\sqrt{ \frac{V}{\nu} }}
             \quad \text{where } Z \sim \mathcal{N}(0, 1),\ V \sim \chi^2(\nu),\ Z \perp V
             ''')
    
    st.markdown("""
                Its PDF is given by:
                """)

    st.latex(r'''
             f(x \mid \nu) = \frac{\Gamma\left(\frac{\nu + 1}{2}\right)}{\sqrt{\nu \pi}\, \Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{x^2}{\nu}\right)^{-\frac{\nu + 1}{2}}
             ''')

    # --- Symbol explanation ---
    st.markdown("""
                - $x$: random variable  
                - $\\nu$: degrees of freedom (自由度)  
                - $\\Gamma$: gamma function  
                """)

    # --- Mean and Standard Deviation ---
    st.markdown("The mean and standard deviation are:")
    st.latex(r'''
             \mu = 0, \quad \sigma = \sqrt{\frac{\nu}{\nu - 2}}, \quad \text{for } \nu > 2
             ''')

    st.markdown("---")
    st.write("### 2️⃣ Make a Plot  |  作圖")

    # --- User input for degrees of freedom ---
    col1, col2 = st.columns(2)
    with col1:
        df = st.slider("Degrees of Freedom (ν)", 1, 50, 5, 1)

    # --- PDF calculation ---
    from scipy.stats import t
    x_range = 5
    x = np.linspace(-x_range, x_range, 500)
    y = t.pdf(x, df)

    # --- Calculate mean and std (if df > 2) ---
    mean = 0
    std = np.sqrt(df / (df - 2)) if df > 2 else np.inf
    lower = mean - 2 * std if df > 2 else -x_range / 2
    upper = mean + 2 * std if df > 2 else x_range / 2

    # --- Plotting PDF and elements ---
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(x, y, label="PDF", color="blue")

    # --- Mean and standard deviation reference lines ---
    ax.axvline(mean, color="red", linestyle="--", label=f"Mean = {mean}")
    if df > 2:
        ax.axvline(lower, color="green", linestyle=":", label=f"μ - 2σ ≈ {lower:.2f}")
        ax.axvline(upper, color="green", linestyle=":", label=f"μ + 2σ ≈ {upper:.2f}")

        # --- Highlight ~95% area ---
        x_fill = np.linspace(lower, upper, 300)
        y_fill = t.pdf(x_fill, df)
        ax.fill_between(x_fill, y_fill, alpha=0.3, color='gray', label="~95% Area")
    else:
        st.warning("Standard deviation is undefined for ν ≤ 2")

    ax.set_title("Student's t-distribution")
    ax.legend(fontsize="small")
    st.pyplot(fig, use_container_width=False)

# --- F-distribution ---

elif dist_type == "F Distribution (F 分布)":
    st.subheader("📈 F-distribution | F 分布")

    # --- Description and formula ---
    st.markdown("""
                The F-distribution is a continuous probability distribution that arises frequently in the context of hypothesis testing,  especially in ANOVA and regression analysi.  
                It is used to compare two variances.
                """)

    st.latex(r'''
             F = \frac{\left( \frac{X_1}{d_1} \right)}{\left( \frac{X_2}{d_2} \right)}
             \quad \text{where } X_1 \sim \chi^2(d_1),\ X_2 \sim \chi^2(d_2)
             ''')
    
    st.markdown("""
                Its PDF is given by:
                """)
    
    st.latex(r'''
             f(x \mid d_1, d_2) = \frac{\sqrt{ \left( \frac{d_1 x}{d_1 x + d_2} \right)^{d_1} \left( \frac{d_2}{d_1 x + d_2} \right)^{d_2} }} {x B\left( \frac{d_1}{2}, \frac{d_2}{2} \right)}, \quad x > 0
             ''')


    # --- Symbol explanation ---
    st.markdown("""
                - $x$: random variable  
                - $d_1$: degrees of freedom for the numerator  
                - $d_2$: degrees of freedom for the denominator  
                - $B$: beta function  
                """)

    # --- Mean and Standard Deviation ---
    st.markdown("The mean and standard deviation are (if $d_2 > 2$):")
    st.latex(r'''
             \mu = \frac{d_2}{d_2 - 2}, \quad 
             \sigma = \sqrt{ \frac{2 d_2^2 (d_1 + d_2 - 2)}{d_1 (d_2 - 2)^2 (d_2 - 4)} }
             \quad \text{for } d_2 > 4
             ''')

    st.markdown("---")
    st.write("### 2️⃣ Make a Plot  |  作圖")

    # --- User input for degrees of freedom ---
    col1, col2 = st.columns(2)
    with col1:
        d1 = st.slider("Numerator df (d₁)", 1, 100, 15, 1)
        d2 = st.slider("Denominator df (d₂)", 1, 100, 20, 1)

    # --- PDF calculation ---
    from scipy.stats import f
    x_max = f.ppf(0.995, d1, d2) if d2 > 2 else 10
    x = np.linspace(0.01, x_max, 500)
    y = f.pdf(x, d1, d2)

    # --- Mean and std computation (if defined) ---
    mean = d2 / (d2 - 2) if d2 > 2 else None
    std = np.sqrt(
        (2 * d2**2 * (d1 + d2 - 2)) / (d1 * (d2 - 2)**2 * (d2 - 4))
    ) if d2 > 4 else None

    # --- Plotting PDF and elements ---
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(x, y, label="PDF", color="blue")

    # --- Annotate mean and ~95% area (if std is defined) ---
    if mean is not None:
        ax.axvline(mean, color="red", linestyle="--", label=f"Mean ≈ {mean:.2f}")
    if mean is not None and std is not None:
        lower = mean - 2 * std
        upper = mean + 2 * std
        ax.axvline(lower, color="green", linestyle=":", label=f"μ - 2σ ≈ {lower:.2f}")
        ax.axvline(upper, color="green", linestyle=":", label=f"μ + 2σ ≈ {upper:.2f}")

        x_fill = np.linspace(lower, upper, 300)
        y_fill = f.pdf(x_fill, d1, d2)
        ax.fill_between(x_fill, y_fill, alpha=0.3, color='gray', label="~95% Area")
    else:
        st.warning("Standard deviation is undefined for d₂ ≤ 4")

    ax.set_title("F-distribution")
    ax.legend(fontsize="small")
    st.pyplot(fig, use_container_width=False)

# --- Beta Distribution ---

elif dist_type == "Beta Distribution (Beta 分布)":
    st.subheader("📈 Beta Distribution | Beta 分布")

    # --- Description and formula ---
    st.markdown("""
                The Beta distribution is a continuous probability distribution defined on the interval [0, 1].  
                It is commonly used to model proportions, probabilities, and uncertainty in Bayesian analysis.
                
                Its PDF is given by:
                """)

    st.latex(r'''
             f(x \mid \alpha, \beta) = \frac{1}{B(\alpha, \beta)} x^{\alpha - 1} (1 - x)^{\beta - 1}, \quad 0 < x < 1
             ''')

    # --- Symbol explanation ---
    st.markdown("""
                - $x$: random variable  
                - $\\alpha$: shape parameter (left shape)  
                - $\\beta$: shape parameter (right shape)  
                - $B(\\alpha, \\beta)$: Beta function (normalizing constant)  
                """)

    # --- Mean and Standard Deviation ---
    st.markdown("The mean and standard deviation are:")
    st.latex(r'''
             \mu = \frac{\alpha}{\alpha + \beta}, \quad 
             \sigma = \sqrt{ \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)} }
             ''')

    st.markdown("---")
    st.write("### 2️⃣ Make a Plot  |  作圖")

    # --- User input for alpha and beta ---
    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider("Alpha (α)", 0.1, 10.0, 5.0, 0.1)
        beta = st.slider("Beta (β)", 0.1, 10.0, 5.0, 0.1)

    # --- PDF calculation ---
    from scipy.stats import beta as beta_dist
    x = np.linspace(0, 1, 500)
    y = beta_dist.pdf(x, alpha, beta)

    # --- Mean and std computation ---
    mean = alpha / (alpha + beta)
    std = np.sqrt((alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1)))
    lower = mean - 2 * std
    upper = mean + 2 * std

    # --- Plotting PDF and elements ---
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(x, y, label="PDF", color="blue")

    # --- Annotate mean and ~95% area ---
    ax.axvline(mean, color="red", linestyle="--", label=f"Mean ≈ {mean:.2f}")
    if lower > 0 and upper < 1:
        ax.axvline(lower, color="green", linestyle=":", label=f"μ - 2σ ≈ {lower:.2f}")
        ax.axvline(upper, color="green", linestyle=":", label=f"μ + 2σ ≈ {upper:.2f}")

        x_fill = np.linspace(lower, upper, 300)
        y_fill = beta_dist.pdf(x_fill, alpha, beta)
        ax.fill_between(x_fill, y_fill, alpha=0.3, color='gray', label="~95% Area")
    else:
        st.warning("The 95% range exceeds the domain [0, 1], and cannot be shown fully.")

    ax.set_title("Beta Distribution")
    ax.legend(fontsize="small")
    st.pyplot(fig, use_container_width=False)

# Footer
st.markdown("---")
st.write("stat2vis: Collection of Applications for Visualizing Statistics")
st.write("GitHub: https://github.com/TeddYenn/stat2vis")