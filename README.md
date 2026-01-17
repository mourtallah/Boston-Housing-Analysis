# Boston Housing Analysis

A linear regression analysis of the celebrated Boston Housing dataset, tracing the steps from data inspection through model assumptions, feature selection, and evaluation.

> **⚠️ Ethical Note:** The `load_boston()` function has been deprecated and removed from scikit-learn (version 1.2+) due to ethical concerns. The dataset contains a variable (`B`) that encodes racial composition, raising fairness issues in housing price prediction. This analysis is presented for educational purposes only. For production work, consider the California Housing or Ames Housing datasets.

---

## Introduction

Linear Regression is a fundamental machine learning technique in data science: here we trace the steps analyzing a dataset, testing model assumptions, and evaluating the model's effectiveness.

The Boston Housing Prices dataset is one of the most famous machine learning datasets. The original dataset was collected by David Harrison, Jr. of the Department of City and Regional Planning at Harvard and Daniel L. Rubinfield of The National Bureau of Economic Research. Its first publication was in the *Journal of Environmental Economics and Management* in 1978 and has been reprised in subsequent research in that journal, as well as in the *Journal of Real Estate Finance and Economics* throughout the 90's.

It is comprised of observations of unique Boston suburbs measured on 13 variables including the proportion of the suburb zoned for lots over 25,000 sq.ft., the proportion of owner-occupied units that pre-date 1940, the average number of rooms per house, and weighted distances to the five Boston employment centers, just to name a few. This project explores the predictive values of these features toward developing a model to predict the median value of owner-occupied homes.

---

## Dataset

The data is sourced from the [original CMU repository](http://lib.stat.cmu.edu/datasets/boston) with 506 observations and 14 attributes:

| Variable | Description |
|----------|-------------|
| CRIM | Per capita crime rate by town |
| ZN | Proportion of residential land zoned for lots over 25,000 sq.ft. |
| INDUS | Proportion of non-retail business acres per town |
| CHAS | Charles River dummy variable (1 if tract bounds river; 0 otherwise) |
| NOX | Nitric oxides concentration (parts per 10 million) |
| RM | Average number of rooms per dwelling |
| AGE | Proportion of owner-occupied units built prior to 1940 |
| DIS | Weighted distances to five Boston employment centres |
| RAD | Index of accessibility to radial highways |
| TAX | Full-value property-tax rate per $10,000 |
| PTRATIO | Pupil-teacher ratio by town |
| B | 1000(Bk - 0.63)² where Bk is the proportion of Black residents by town |
| LSTAT | Percentage lower status of the population |
| MEDV | Median value of owner-occupied homes in $1000's (target) |

---

## Preprocessing

It is good practice to check for missing or invalid values, both of which there are none in this data. However, this is not necessarily the entire story. Upon inspection, CHAS and ZN have an inordinate amount of zeros—indeed over half of each measure are 0.

I have observed other researchers drop these two variables altogether after relabelling "0" as NA. While conventional wisdom advises we drop measures for which over 3/4's of observations are invalid, that application of wisdom in this case is unjustified. If we inspect the meaning of these two measures—CHAS represents a spatial relationship to the Charles River, and ZN represents the proportion of residential land zoned for lots larger than 25,000 square feet—both are plausible to be mostly comprised of zeros. Common sense is often the most robust algorithm.

---

## Analysis

### Collinearity

Correlation analysis reveals the strongest relationship is between RAD and TAX (~0.91), indicating that though one may be used in the model, using both would be inappropriate. Comparing their correlations with MEDV:

- RAD with MEDV: **-0.381**
- TAX with MEDV: **-0.469**

The magnitude of correlation between median home value and full-value property tax is greater than its correlation with access to radial highways. This indicates TAX likely has greater predictive power than RAD.

### Feature Selection

Through correlation analysis, the most promising predictors for median home value are:

- **LSTAT** (r ≈ -0.74): Percentage lower status population
- **RM** (r ≈ 0.70): Average number of rooms
- **PTRATIO** (r ≈ -0.51): Pupil-teacher ratio

Between RM and LSTAT, the absolute value of the highest correlation amongst other explanatory variables is around 0.6, indicating they are not only correlated with the response but adequately linearly independent from other features.

### Linearity Assessment

The relationship between LSTAT and MEDV appears somewhat curvilinear, while RM and PTRATIO seem to suffer from moderate to severe heteroscedasticity, respectively. To address the curvilinearity, a log transformation was applied to LSTAT.

---

## Model

A linear regression model was fitted using three features:
- `RM` (rooms)
- `PTRATIO` (pupil-teacher ratio)  
- `log_LSTAT` (log-transformed lower status percentage)

Data was split 80/20 for training and testing with `random_state=5`.

### Performance

| Metric | Training Set | Test Set |
|--------|--------------|----------|
| RMSE | ~4.84 | ~4.40 |
| R² | ~0.73 | ~0.75 |

### Cross-Validation

5-fold cross-validation provides more robust estimates:
- **Mean R²:** ~0.72 (+/- 0.10)
- **Mean RMSE:** ~4.68 (+/- 0.86)

### Residual Diagnostics

To validate linear regression assumptions, residuals were examined:
1. **Residuals vs Fitted Values:** Some evidence of heteroscedasticity
2. **Q-Q Plot:** Slight deviation from normality at the tails
3. **Shapiro-Wilk Test:** Formal test for normality of residuals

---

## Conclusion

When the dust settled, the strongest signals for median value came from `RM`, `LSTAT`, and `PTRATIO`. The correlation heatmaps made clear that `RAD` and `TAX` were nearly interchangeable; between the two I kept `TAX` so the model is not double-counting the same story. To tame the slight curve in `LSTAT`'s relationship to `MEDV`, I log-transformed it before fitting.

This three-feature linear model explains roughly three quarters of the variation in prices: R² sits between 0.73 and 0.75 across the training and held-out splits. RMSE hovers around $4,400–$4,800, which means even the good predictions are still off by a few thousand 1970s dollars—useful, but not clairvoyant.

### Limitations

This entire exercise is anchored in a 1978 snapshot of Boston; it cannot speak to current market dynamics. The `B` variable encodes racial composition, so any contemporary, ethical analysis would drop it. Residual checks hint at heteroscedasticity and some deviation from normality, reminding us that the linear assumptions are only approximately satisfied. And many real-world drivers—interest rates, neighborhood amenities, broader economic tides—sit outside this dataset altogether.

### Future Work

- **Regularization:** Ridge or Lasso regression to address multicollinearity
- **Non-linearity:** Polynomial terms or tree-based models to capture complex relationships
- **Modern data:** Repeat this workflow on an ethically vetted housing dataset

---

## Repository Structure

```
├── notebooks/
│   ├── BostonHousingDraft.ipynb    # Main analysis notebook
│   └── GC_LinearRegression.ipynb   # Quick reference implementation
├── drafts/
├── research/
│   └── BostonHousingMetadata.html
├── .gitignore
├── BostonHousingDataAnalysis.Rproj
└── README.md
```

---

## Getting Started

```bash
# Create virtual environment
uv venv .venv

# Activate environment
source .venv/bin/activate

# Install dependencies
uv pip install numpy pandas matplotlib seaborn scikit-learn scipy ipykernel

# Register Jupyter kernel
python -m ipykernel install --user --name=boston-housing --display-name="Boston Housing (Python)"
```

---

## References

- Harrison Jr, David, and Daniel L. Rubinfeld. "Hedonic housing prices and the demand for clean air." *Journal of Environmental Economics and Management* 5.1 (1978): 81-102.
- Carlisle, M. "Racist data destruction?" [Medium](https://medium.com/@docintangible/racist-data-destruction-113e3eff54a8)
