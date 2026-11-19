# Porównanie modeli (Sprint 3, krok 4)

Tabela powstaje automatycznie w node `build_model_comparison_node` (pipeline `data_science`)
i jest zapisywana do `data/08_reporting/model_comparison.csv` oraz `model_comparison.md`.
Gdy W&B jest włączone, ta sama tabela trafia do runu `model-comparison` jako `wandb.Table`.

Odtworzenie:

```bash
WANDB_MODE=offline kedro run
```

`WANDB_MODE=offline` zapisuje runy lokalnie w `wandb/`; po założeniu konta można je wysłać
komendą `wandb sync wandb/offline-run-*`.

## Wyniki

Run z 19.11.2026, pełne dane (`online_shoppers`, SHA-256 `b3055ee3…e0d9`), split
`stratified_random` 80/20, `random_state=42`, zbiór testowy 2466 sesji. Pomiary na laptopie
z Windows 11 (CPU), więc liczy się porządek wielkości, a nie dokładne milisekundy.

| Model | Najlepszy model wewnętrzny | F1 (główna) | Precision | Recall | Accuracy | Predykcja zbioru test [ms] | Predykcja 1 wiersza [ms] | Rozmiar artefaktu [MB] | Pamięć po wczytaniu [MB] | Wczytanie + 1. predykcja [s] | Czas treningu [s] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (logistic_regression) | — | 0.3852 | 0.2593 | 0.7487 | 0.6298 | 8.75 | 3.40 | 0.01 | 65.27 | 1.44 | — |
| ag_gbm_cat_medium | WeightedEnsemble_L2 | 0.3962 | 0.2953 | 0.6021 | 0.7157 | 15.07 | 7.46 | 0.87 | 95.33 | 3.10 | 37.11 |
| ag_good_quality | LightGBMXT_BAG_L2_FULL | 0.3829 | 0.3045 | 0.5157 | 0.7425 | 501.92 | 358.93 | 133.90 | 115.89 | 4.46 | 317.31 |

Konfiguracje AutoGluona (`conf/base/parameters.yml`):

- `ag_gbm_cat_medium`: `presets=medium_quality`, tylko LightGBM i CatBoost, limit 120 s;
- `ag_good_quality`: `presets=good_quality` (bagging + stacking), limit 300 s.

## Jak mierzymy

| Kolumna | Metoda |
|---|---|
| F1 i pozostałe metryki | `compute_metrics` na zbiorze testowym, klasa pozytywna `Revenue=true` (1) |
| Predykcja zbioru test | mediana z 5 wywołań `predict` na całym `X_test` |
| Predykcja 1 wiersza | mediana z 20 wywołań `predict` dla pojedynczej sesji; tak będzie wołało API w S4 |
| Rozmiar artefaktu | rozmiar na dysku: pakiet baseline'u (`model.pkl` + `metadata.json`) oraz katalog `clone_for_deployment` AutoGluona |
| Pamięć po wczytaniu | przyrost RSS w **osobnym, świeżym procesie**: import bibliotek modelu + wczytanie artefaktu + jedna predykcja 5 wierszy (punkt odniesienia: interpreter + pandas) |
| Wczytanie + 1. predykcja | czas tego samego kroku w świeżym procesie (zimny start usługi) |

Pamięć mierzymy w osobnym procesie, bo w procesie treningowym biblioteki są już wczytane.
Predykcja jest częścią pomiaru, bo AutoGluon wczytuje modele leniwie dopiero przy pierwszym
`predict`. Funkcje pomiarowe: `measure_prediction_time` i `measure_load_memory`
w `src/asi_project/tracking.py`.

## Obserwacje (wejście do kroku 5)

- F1 wszystkich trzech modeli jest zbliżone (0.383–0.396). Najlepsza konfiguracja
  `ag_gbm_cat_medium` jest lepsza od baseline'u o ok. 0.011 F1. Wyniki pochodzą
  z jednego splitu, więc tak mała różnica może mieścić się w losowej zmienności.
- `ag_good_quality` nie poprawia F1, a predykcja jednego wiersza trwa ~105× dłużej niż
  w baseline'ie (~359 ms vs ~3.4 ms), artefakt jest ~150× większy niż w `ag_gbm_cat_medium`
  i ~18 000× większy niż w baseline'ie (133.9 MB vs 7.5 KB). Pełny katalog treningowy przed `clone_for_deployment`
  zajmuje ~386 MB.
- `ag_gbm_cat_medium` jest kompromisem: niewiele lepsze F1 przy dwukrotnie dłuższej
  predykcji jednego wiersza i +30 MB pamięci względem baseline'u.
- Baseline ma najwyższy recall (0.75), ale najniższą precision. Modele AutoGluona są
  bardziej zachowawcze. Wszystkie modele używają domyślnego progu 0.5.
