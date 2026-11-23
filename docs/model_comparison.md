# Porównanie modeli (Sprint 3, krok 4)

Wynik `kedro run` z 20.11.2026 (pełne dane, split 80/20, `random_state=42`).
Tabelę generuje node `build_model_comparison_node` do `data/08_reporting/model_comparison.md`.



| Model | Najlepszy model wewnętrzny | F1 (główna) | Precision | Recall | Accuracy | Predykcja zbioru test [ms] | Predykcja 1 wiersza [ms] | Rozmiar artefaktu [MB] | Pamięć po wczytaniu [MB] | Wczytanie + 1. predykcja [s] | Czas treningu [s] |
|-|-|-|-|-|-|-|-|-|-|-|-|
| baseline (logistic_regression) | - | 0.3852 | 0.2593 | 0.7487 | 0.6298 | 15.38 | 7.09 | 0.01 | 64.79 | 1.88 | - |
| ag_gbm_cat_medium | WeightedEnsemble_L2 | 0.3962 | 0.2953 | 0.6021 | 0.7157 | 21.24 | 9.29 | 0.87 | 95.64 | 3.61 | 65.80 |
| ag_good_quality | LightGBMXT_BAG_L1_FULL | 0.3882 | 0.2948 | 0.5681 | 0.7226 | 99.71 | 18.95 | 3.23 | 96.38 | 5.16 | 319.82 |

## Wybór modelu produkcyjnego (Sprint 3, krok 5)

**Decyzja: do produkcji trafia `ag_gbm_cat_medium`** (AutoGluon, preset `medium_quality`,
tylko LightGBM + CatBoost, najlepszy model wewnętrzny: `WeightedEnsemble_L2`).

### Kryteria

1. F1 na zbiorze testowym (metryka główna z kontraktu, klasa pozytywna = zakup).
2. Koszt wdrożenia w API (Sprint 4): czas predykcji 1 wiersza, rozmiar artefaktu,
   pamięć po wczytaniu, czas zimnego startu.
3. Czas treningu: istotny tylko pomocniczo (trening nie jest na ścieżce API).

### Porównanie z wybranym modelem

| Wariant | F1 vs wybrany | Predykcja 1 wiersza | Rozmiar | Pamięć | Zimny start | Wniosek |
|-|-|-|-|-|-|-|
| **ag_gbm_cat_medium** | - | 9.29 ms | 0.87 MB | 95.6 MB | 3.61 s | **wybrany** |
| ag_good_quality | -0.0080 | 18.95 ms (2x) | 3.23 MB (3.7x) | 96.4 MB | 5.16 s | odrzucony - gorszy we wszystkim |
| baseline (logistic_regression) | -0.0110 | 7.09 ms | 0.01 MB | 64.8 MB | 1.88 s | odrzucony - najniższe F1 |

### Uzasadnienie

- `ag_gbm_cat_medium` ma najwyższe F1 (0.3962) i najwyższą precision (0.2953).
- `ag_good_quality` jest **zdominowany**: niższe F1, 2x wolniejsza predykcja wiersza,
  4.7x wolniejsza predykcja zbioru testowego (99.7 vs 21.2 ms), 3.7x większy artefakt,
  wolniejszy zimny start i 5x dłuższy trening. Droższy preset nie dał zysku jakości.
- Baseline jest najtańszy, ale różnice w kosztach są dla API pomijalne:
  +2.2 ms na predykcję wiersza, +0.86 MB artefaktu, +31 MB pamięci, +1.7 s zimnego
  startu (jednorazowo przy starcie usługi). W zamian dostajemy +0.011 F1
  (+2.9% względnie), +0.036 precision i +0.086 accuracy, czyli mniej fałszywych alarmów.
- Koszt wyboru: niższy recall niż baseline (0.6021 vs 0.7487). Akceptujemy to,
  bo metryką główną jest F1, a nie recall.

### Ryzyka i zastrzeżenia

- Różnica F1 względem baseline jest niewielka i pochodzi z jednego splitu 80/20
  (`random_state=42`) - bez CV nie wiemy, czy jest stabilna. Jeżeli w kolejnych runach
  przewaga zniknie, wracamy do baseline jako tańszego modelu (aktualizacja ADR-2).
- Model AutoGluona to katalog (`TabularPredictor`), a nie pojedynczy estymator sklearn.
  Pakiet w kroku 6 musi zawierać wersję po `clone_for_deployment` i wczytywać ją przez
  `TabularPredictor.load`, a nie przez `joblib` jak obecnie w `model_package.py`.
- Pakiet produkcyjny wymaga `autogluon.tabular` w obrazie API - większy obraz Dockera
  niż przy baseline (do uwzględnienia w ADR-3, Sprint 4).

Pełne uzasadnienie decyzji trafi do ADR-2 (krok 11).
