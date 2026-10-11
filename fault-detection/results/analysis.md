# Task 6 - Analysis

All numbers come from `results/table*.csv` (test set = 2,401 rows, split seed 42).

| Model | Test acc. | False alarms | Missed faults |
|---|---|---|---|
| Logistic regression, Set R | 72.64 % | 0 | 657 |
| Logistic regression, Set Q | 97.88 % | 6 | 45 |
| Linear SVM (C = 0.1) | 73.59 % | 0 | 634 |
| RBF SVM (C = 100, gamma = 1) | 99.58 % | 4 | 6 |

## 6.1 Why do logistic regression (Set R) and the linear SVM score almost the same?
They are both **linear classifiers**: each can only draw a single flat boundary (a hyperplane) in the
six-dimensional feature space. They only differ in the loss they minimise (log-loss vs hinge loss), so
when the best possible straight line is poor, both find roughly the same poor line (72.6 % vs 73.6 %).
Both predict "no fault" for every no-fault row (0 false alarms) and still miss 630-660 real faults.
This tells us the problem is **not the optimiser or the hyper-parameters (the linear SVM's accuracy was
identical, 72.58 %, for all four values of C) - the data are not linearly separable**. The limit is the
model family, so a better algorithm of the same type cannot fix it; we must change the features or the kernel.

## 6.2 Signs of the squared-term weights (Set Q)
| Weight | Value |
|---|---|
| Ia^2 | +18.89 |
| Ib^2 | +4.75 |
| Ic^2 | +13.96 |
| Va^2 | -6.03 |
| Vb^2 | -6.46 |
| Vc^2 | -6.14 |

* **I^2 weights are positive.** A short circuit gives the current a very low-impedance path, so current
  magnitude jumps from about 40 A (mean |Ia| for no-fault rows) to about 460 A (fault rows). Because the
  line currents are sinusoidal, the *sign* of Ia is positive or negative at random, but Ia^2 is always large
  in a fault. Large I^2 therefore pushes the score toward "fault".
* **V^2 weights are negative.** A fault drags the line voltage toward zero at the fault point, so
  voltage magnitudes sag. Large V^2 means a healthy voltage, so it pushes the score toward "no fault".
* The linear terms (Ia, Ib, ...) get small weights because a signed instantaneous value carries almost
  no class information on its own; the energy-like squared terms do.

## 6.3 Support vectors: linear vs RBF
The best linear SVM uses **5,849** support vectors (81 % of the 7,200 training rows); the best RBF SVM
uses **266** (3.7 %). Support vectors are the points on or inside the margin or misclassified. A very
large number means the model cannot separate the classes: most points violate the margin, so the
boundary depends on almost every training point. This is the signature of underfitting / a wrong model
class. The RBF SVM can bend around the blue cluster, so only a small band of border points matters.

## 6.4 Effect of C on accuracy and support vectors
* **Linear:** accuracy does not change (72.58 % for every C) and the support-vector count stays near
  5,840, because no amount of extra flexibility lets a straight line separate the classes.
* **RBF:** accuracy rises with C (gamma = 1: 98.04 % -> 99.17 % -> 99.62 % -> 99.67 %) and the number of
  support vectors falls (848 -> 482 -> 342 -> 266; gamma = 0.1: 1347 -> 174).
* C is the penalty for margin violations. A small C tolerates many violations (soft, wide margin), so many
  points sit inside the margin and become support vectors. A large C punishes violations heavily, giving a
  narrower margin that hugs the true boundary, so fewer points end up on or inside it. The trade-off
  is that very large C can over-fit noise, so we should still choose C on the validation set.
  (Here the best validation score is C = 100, gamma = 1, and the gain from C = 10 to 100 is only 0.05 %.)

## 6.5 Is the test accuracy optimistic or pessimistic?
**Optimistic.** The rows are consecutive time samples, so a test row almost always has near-identical
neighbours (one sample earlier and later) in the training set. The model can score well just by
"remembering" the neighbours - this is data leakage through time. A real relay has to handle a *new* fault
event it has never seen (different fault location, resistance, inception angle, load), and the numbers
above do not measure that. A fairer evaluation would split by blocks of time or by whole fault events so
that the test events are completely unseen.

## 6.6 Which model would I deploy?
The **RBF SVM (C = 100, gamma = 1)**: only **6 missed faults** and 4 false alarms out of 2,401 rows (99.58 %).
In a protection system a missed fault means the breaker does not trip, which can damage equipment, so
missed faults matter far more than false alarms. The Set R models are unacceptable (630+ missed faults,
roughly 58-60 % of all faults). Logistic regression with squares (Set Q) is a reasonable cheap backup
(45 missed) and is easy to implement in hardware, but it misses 7x more faults than the RBF SVM. Before
real deployment I would also (a) lower the decision threshold to trade a few more false alarms for fewer
missed faults, (b) use a window of several samples rather than one instant (a sinusoid passes through zero,
which is where single-sample misses occur) and (c) re-test with an event-based split as in 6.5.
