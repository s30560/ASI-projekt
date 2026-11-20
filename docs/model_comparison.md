# Porównanie modeli (Sprint 3, krok 4)

Wynik `kedro run` z 20.11.2026 (pełne dane, split 80/20, `random_state=42`).
Tabelę generuje node `build_model_comparison_node` do `data/08_reporting/model_comparison.md`.



| Model | Najlepszy model wewnętrzny | F1 (główna) | Precision | Recall | Accuracy | Predykcja zbioru test [ms] | Predykcja 1 wiersza [ms] | Rozmiar artefaktu [MB] | Pamięć po wczytaniu [MB] | Wczytanie + 1. predykcja [s] | Czas treningu [s] |
|-|-|-|-|-|-|-|-|-|-|-|-|
| baseline (logistic_regression) | - | 0.3852 | 0.2593 | 0.7487 | 0.6298 | 15.38 | 7.09 | 0.01 | 64.79 | 1.88 | - |
| ag_gbm_cat_medium | WeightedEnsemble_L2 | 0.3962 | 0.2953 | 0.6021 | 0.7157 | 21.24 | 9.29 | 0.87 | 95.64 | 3.61 | 65.80 |
| ag_good_quality | LightGBMXT_BAG_L1_FULL | 0.3882 | 0.2948 | 0.5681 | 0.7226 | 99.71 | 18.95 | 3.23 | 96.38 | 5.16 | 319.82 |
