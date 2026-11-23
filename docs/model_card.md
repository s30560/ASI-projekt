# Model Card: `ag_gbm_cat_medium` (`model-package:production`)

> Minimalna Model Card ze Sprintu 3. Aktualizacja w Sprincie 6.
> Decyzja o wyborze modelu: [ADR-2](adr/0002-wybor-modelu-produkcyjnego.md).

## Model

| | |
|---|---|
| Model produkcyjny | `ag_gbm_cat_medium`: AutoGluon `TabularPredictor`, preset `medium_quality`, tylko LightGBM i CatBoost, limit czasu 120 s |
| Model wewnętrzny | `WeightedEnsemble_L2` (ensemble LightGBM + CatBoost) |
| Artefakt | W&B `model-package`, alias **`production`** |
| Postać | katalog `model/` (predyktor po `clone_for_deployment`) + `metadata.json`; wczytanie przez `load_model_package` z `src/asi_project/model_package.py` |
| Utworzony | `2026-11-23`, pipeline Kedro `data_science`, node `package_production_model_node` |

## Przeznaczenie

- Zadanie: klasyfikacja binarna, czy sesja w sklepie internetowym zakończy się zakupem.
- Target: **`Revenue`**, klasa pozytywna: **`true`** (sesja zakończona zakupem). Model zwraca
  `1` dla klasy pozytywnej i `0` dla pozostałych.
- Moment predykcji: dla sesji na podstawie jej zagregowanych statystyk (liczby i czasy
  odwiedzonych stron, `BounceRates`, `ExitRates`, kontekst wizyty), zanim wiadomo, czy
  nastąpił zakup.
- Zastosowanie: wskazanie sesji z podwyższonym prawdopodobieństwem zakupu, np. do analizy
  ruchu lub priorytetyzacji działań marketingowych.
- Poza zakresem: decyzje dotyczące konkretnych osób (ceny, odmowa usługi), sklepy o innej
  strukturze ruchu niż zbiór źródłowy, predykcja w trakcie sesji na częściowych danych.

## Dane

- Dataset: `online_shoppers`, Online Shoppers Purchasing Intention (UCI, DOI
  10.24432/C5F88Q, licencja CC BY 4.0). 12 330 sesji, klasa pozytywna 1 908 (15,5%).
- Fingerprint danych (SHA-256 pliku): `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`
- Cechy wejściowe (16): `Administrative`, `Administrative_Duration`, `Informational`,
  `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`,
  `ExitRates`, `SpecialDay`, `Month`, `OperatingSystems`, `Browser`, `Region`,
  `TrafficType`, `VisitorType`, `Weekend`. Pełna lista w `metadata.json` (`features`).
  `PageValues` jest wykluczona polityką kursu (niejasna dostępność w momencie predykcji).
- Split: `stratified_random`, `test_size=0.2`, `random_state=42`;
  trening 9 864, test 2 466 sesji.

## Metryki (zbiór testowy, 2 466 sesji)

Metryka główna: **F1 dla klasy pozytywnej**. Próg decyzji: domyślny AutoGluona (0.5).

| Model | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|
| DummyClassifier (zawsze `0`) | 0.0000 | 0.0000 | 0.0000 | 0.8451 |
| baseline (`LogisticRegression`) | 0.3852 | 0.2593 | 0.7487 | 0.6298 |
| **`ag_gbm_cat_medium` (production)** | **0.3962** | **0.2953** | **0.6021** | **0.7157** |

Macierz pomyłek modelu produkcyjnego:

| | przewidziano `0` | przewidziano `1` |
|---|---|---|
| **faktycznie `0`** | 1535 (TN) | 549 (FP) |
| **faktycznie `1`** | 152 (FN) | 230 (TP) |

Koszt obsługi (W&B, 20.11.2026): predykcja 1 wiersza ok. 9 ms, artefakt ok. 0.9 MB,
pamięć po wczytaniu ok. 96 MB, wczytanie + 1. predykcja ok. 3.6 s.

## Ograniczenia i ryzyka

- Niska precision: ok. 7 na 10 sesji oznaczonych jako zakup to fałszywe alarmy, a recall
  0.60 oznacza, że model pomija ok. 40% zakupów. Nadaje się do rankingu sesji, nie do
  automatycznych decyzji.
- Mała przewaga nad baseline: ok. +0.01 F1 z jednego splitu 80/20, bez walidacji krzyżowej.
  Jeśli nie utrzyma się w kolejnych runach, wracamy do tańszego baseline (ADR-2).
- Niezbalansowane klasy (15,5%) i domyślny próg 0.5, który nie był strojony.
- Dane z jednego sklepu z okresu jednego roku. Sezonowość (`Month`, `SpecialDay`) i struktura
  ruchu mogą nie przenosić się na inne sklepy ani okresy.
- Obraz API musi zawierać `autogluon.tabular` w tej samej wersji co przy treningu (niżej);
  predyktor nie wczyta się na innej wersji AutoGluona.

## Wersje

| | |
|---|---|
| Git SHA treningu | `5fd421b` |
| Biblioteki | `python` 3.11.18, `pandas` 2.3.3, `numpy` 2.2.6, `scikit-learn` 1.7.2, `joblib` 1.6.0, `wandb` 0.30.0, `autogluon.tabular` 1.6.3 |
