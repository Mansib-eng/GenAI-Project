# Day 6 — Classical Baseline Error Analysis

## Evaluation summary

The TF-IDF + Logistic Regression baseline was evaluated on an untouched test set of 6 texts: 3 factual and 3 clickbait. It correctly classified 5 of 6 examples, producing an accuracy of 0.833 and a Macro-F1 of 0.829. Macro-F1 is slightly lower because it gives equal weight to both classes and the factual class performed less strongly. Because this test set is extremely small, these scores are suitable for workflow practice but not for a reliable real-world performance claim.

## Confusion-matrix findings

The confusion matrix contains 2 true negatives, 1 false positive, 0 false negatives, and 3 true positives. The model detected all 3 clickbait examples, so clickbait recall was 1.000. However, one factual example was incorrectly flagged as clickbait. Consequently, factual recall was 0.667, while clickbait precision fell to 0.750. In this sample, the model was therefore slightly more willing to predict clickbait than to miss it.

## Misclassified example

The only error was: **â€œResearchers publish results of health studyâ€**. Its true class was factual, but the model assigned a clickbait probability of 0.508, only slightly above the 0.50 decision threshold. This is a borderline prediction rather than a confident failure. Of the sentence's meaningful words, most were absent from the small training vocabulary. The recognized term was `of`. That term had a small positive contribution toward clickbait, leaving the classifier without enough learned factual evidence to make the correct decision.

## Likely cause and next actions

The main limitation is data sparsity: only 28 training texts and 6 test texts were used. The model cannot learn dependable weights for varied factual vocabulary such as research, health, and study when those terms do not occur in training. The next experiment should use a larger, more diverse dataset; retain stratified splitting; inspect false positives and false negatives separately; and compare word n-grams with character n-grams. Performance should then be reported with class-level precision, recall, F1, support, and Macro-F1â€”not accuracy alone.