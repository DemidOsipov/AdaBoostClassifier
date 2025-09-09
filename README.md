# AdaBoost SAMME.R Implementation

Custom AdaBoost classifier implementing the SAMME.R algorithm for multiclass classification problems. It works better than classic SAMME for multiclass problems. Unlike sklearn's implementation, this one lets you see exactly what's happening under the hood.

## Key Features

- **Multiclass support** - handles more than 2 classes out of the box
- **SAMME.R algorithm** - uses probability estimates for smoother weight updates  
- **Flexible base estimators** - works with any sklearn-compatible classifier
- **Training monitoring** - tracks weighted error for each estimator
- **Proper seeding** - ensures reproducible results across different base estimators

## Usage

```python
from sklearn.tree import DecisionTreeClassifier

# Basic usage
clf = MyAdaBoostClassifier(
    n_estimators=50,
    base_estimator=DecisionTreeClassifier,
    max_depth=3,
    seed=42
)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

# Check training progress
plt.plot(clf.error_history)
```

## How It Works

The algorithm sequentially trains weak classifiers, reweighting training samples after each iteration to focus on previously misclassified examples. SAMME.R uses probability estimates rather than discrete predictions for smoother convergence.

This was implemented as part of YSDA ML course assignments. The math follows the original SAMME.R paper pretty closely.
