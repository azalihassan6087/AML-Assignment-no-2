⚡ Can a straight line detect a fault on a power line?

For our Applied Machine Learning assignment we built a fault detector for a three-phase transmission line using only six instantaneous measurements (3 currents, 3 voltages).

What we found:
📉 Logistic regression (written by hand in NumPy) and a linear SVM both stalled at ~73% accuracy and missed 630+ faults. A fault changes the *magnitude* of the current, not its sign, so no straight line separates the classes.
📈 Fix 1, new features: adding squared terms lifted logistic regression to 97.9%.
🚀 Fix 2, new classifier: an RBF-kernel SVM reached 99.6% with only 6 missed faults out of 2,401 test rows.

In protection systems, a missed fault is worse than a false alarm, so we judged models by missed faults, not accuracy alone. We also discuss why our test accuracy is probably optimistic (neighbouring time samples leak across the split).

📝 Blog: [Add Medium link here]
💻 Code + Streamlit demo: [Add GitHub repo link here]

#MachineLearning #SVM #LogisticRegression #PowerSystems #FaultDetection #Python
