

## 📊 Descriptive Statistics 

Descriptive statistics is all about **summarizing and describing data** so you can understand what's going on before building any model.

It splits into **4 big ideas:**

---

### 1️⃣ Measures of Central Tendency
*"Where is the center of my data?"*

| Measure | Formula | Best Used When |
|---|---|---|
| **Mean** | Sum / Count | Data has no outliers |
| **Median** | Middle value | Data has outliers/skew |
| **Mode** | Most frequent value | Categorical data |

**Example — Salaries of 5 employees:**
`[30k, 32k, 31k, 29k, 200k]`

- **Mean** = 64.4k → *Misleading! One outlier drags it up*
- **Median** = 31k → *More honest representation*
- **Mode** = No mode here, but useful in e.g. shoe sizes

> 💡 **Data Science tip:** Always check mean vs median. A big gap = skewed data = be careful with linear models.

---

### 2️⃣ Measures of Spread / Dispersion
*"How scattered is my data?"*

**Variance** — Average of squared differences from the mean.
- Population: σ² = Σ(x - μ)² / N
- Sample: s² = Σ(x - x̄)² / (n-1) ← uses n-1 (Bessel's correction) to avoid underestimating

**Standard Deviation (SD)** — Square root of variance. Same unit as data, easier to interpret.

**Example:**
Two student groups both scored a mean of 70/100:
- Group A scores: `[68, 70, 71, 69, 72]` → SD = ~1.5 (very consistent)
- Group B scores: `[40, 90, 55, 95, 50]` → SD = ~24 (wildly spread)

Same mean, totally different story!

**Range** = Max - Min → Simple but sensitive to outliers.

**IQR (Interquartile Range)** = Q3 - Q1 → Spread of the middle 50%. Outlier-resistant.

> 💡 **Data Science tip:** SD is used in Z-score normalization. IQR is used in box plots and outlier detection (1.5×IQR rule).

---

### 3️⃣ Measures of Shape
*"What does the distribution look like?"*

**Skewness** — Measures asymmetry of the distribution.

- **Positive skew (Right skew):** Tail is on the right → Mean > Median (e.g. income data)
- **Negative skew (Left skew):** Tail is on the left → Mean < Median (e.g. age at retirement)
- **Symmetric:** Mean ≈ Median ≈ Mode (e.g. height)

**Kurtosis** — Measures the "peakedness" or heaviness of tails.

- **Leptokurtic (Kurtosis > 3):** Sharp peak, heavy tails → more outliers (e.g. stock returns)
- **Platykurtic (Kurtosis < 3):** Flat peak, thin tails → fewer outliers
- **Mesokurtic (Kurtosis = 3):** Normal distribution

> 💡 **Data Science tip:** High kurtosis = more extreme values = your model might struggle. Many ML algorithms assume normality — always check skewness and kurtosis first.

---

### 4️⃣ Measures of Position
*"Where does a specific value stand?"*

**Percentiles & Quartiles:**
- Q1 = 25th percentile, Q2 = 50th (Median), Q3 = 75th
- Example: If your exam score is at the **90th percentile**, you scored better than 90% of students.

**Z-Score** = (x - μ) / σ
- Tells you how many standard deviations a value is from the mean.
- Example: Height mean = 170cm, SD = 10cm. Person is 190cm → Z = (190-170)/10 = **+2** → unusually tall.

> 💡 **Data Science tip:** Z-scores are used in feature scaling (StandardScaler), anomaly detection, and comparing values across different scales.

---

### 🔗 How They All Connect in Data Science

```
Raw Data
   ↓
Central Tendency  → Understand the "typical" value
   ↓
Spread            → Understand variability / risk
   ↓
Shape             → Check assumptions before modeling
   ↓
Position          → Detect outliers, scale features
   ↓
Ready for EDA & Modeling ✅
```

---

