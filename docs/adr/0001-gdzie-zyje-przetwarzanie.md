# ADR-1: Gdzie żyje przetwarzanie danych

- **Status:** proponowany
- **Sprint:** S2
- **Autor:** Tadeusz Kucza-Kuczyński
- **Recenzent:**

## Kontekst

W Sprincie 2 przenosimy trening z notebooka do pipeline'u Kedro (`data_processing` → `data_science`). Od Sprintu 4 ten sam model będzie obsługiwał pojedyncze rekordy w API. Musimy więc ustalić, w którym miejscu systemu wykonuje się każdy z trzech rodzajów przetwarzania. Chodzi o to, żeby:

- predykcja w API widziała dane w tej samej postaci co trening (brak training/serving skew),
- żadna statystyka zbioru testowego nie przeciekała do treningu (brak data leakage),
- logika nie była zduplikowana między notebookiem, pipeline'em i API.

Dane: `online_shoppers` (UCI, 12 330 sesji, 18 kolumn, target `Revenue`, klasa pozytywna `true` stanowi 15,47% rekordów). Split `stratified_random`, `test_size=0.2`, `random_state=42`.

## Rozważone opcje

1. **Trzy warstwy w trzech miejscach.** Input normalization jako funkcja w `data.py` używana przez node Kedro i później przez API. Training-only filtering jako osobny node tylko w pipeline'ie treningowym. Model preprocessing wewnątrz sklearn `Pipeline`.
2. **Całe przetwarzanie w nodach Kedro.** Imputacja, skalowanie i one-hot jako osobne nody przed treningiem, a dopasowane transformatory zapisywane jako osobne artefakty.
3. **Całe przetwarzanie w sklearn `Pipeline`.** Normalizacja, filtrowanie i preprocessing jako kroki jednego obiektu modelu.

## Decyzja

Wybieramy **opcję 1**:

| Rodzaj przetwarzania | Gdzie żyje | Kiedy działa |
|---|---|---|
| Input normalization | `asi_project.data.normalize_input`, node `input_normalization_node` | trening **i** predykcja (API od S4) |
| Training-only filtering | `asi_project.data.filter_training_rows`, node `training_only_filtering_node` | wyłącznie przy przygotowaniu danych treningowych |
| Model preprocessing | `asi_project.model.build_baseline_pipeline` (`ColumnTransformer`: imputacja, `StandardScaler`, `OneHotEncoder`) | wewnątrz modelu: `fit` na `X_train`, `transform` przy każdej predykcji |

Dlaczego:

- **Input normalization** działa na pojedynczym rekordzie, nie używa targetu ani statystyk zbioru i jest idempotentna (sprawdzają to testy w `tests/test_project.py`). Dzięki temu jedną funkcję można bezpiecznie wywołać zarówno w pipeline'ie, jak i w API, bez kopii logiki.
- **Training-only filtering** usuwa rekordy, np. bez targetu. Przy predykcji targetu nie ma, a rekordu użytkownika nie wolno „odfiltrować”, więc ten krok nie może należeć ani do API, ani do modelu.
- **Model preprocessing** wymaga `fit`. Gdy jest częścią sklearn `Pipeline`, dopasowuje się wyłącznie na `X_train` (brak leakage), a model zapisuje się jako jeden artefakt (`trained_model`), który przy predykcji wykonuje dokładnie te same transformacje.

Liczby z naszego systemu, na których opiera się decyzja:

| Wielkość | Wartość | Skąd pochodzi |
|---|---|---|
| Rekordy w pełnych danych | 12 330 | `online_shoppers_intention.csv` |
| Rekordy bez targetu usuwane przez filtering | 0 | pełne dane, `Revenue.isna().sum()` |
| Dokładne duplikaty w danych | 125 (zachowane) | pełne dane, `df.duplicated().sum()` |
| Rozmiar train / test po filteringu | 9 864 / 2 466 | `metrics.json` → `split` |
| Odsetek klasy pozytywnej | 15,47% | pełne dane, `Revenue.mean()` |
| F1 LogisticRegression (z preprocessingiem w `Pipeline`) | 0,385 | `metrics.json` |
| F1 DummyClassifier | 0,000 | `metrics.json` |

Filtering nie usuwa obecnie żadnego rekordu (0 braków targetu, 9 864 + 2 466 = 12 330), ale zostaje jako jawny krok. Kontrakt danych (`filter_training_rows`) gwarantuje, że rekordy bez targetu nigdy nie trafią do treningu, także przy nowych wersjach danych.

## Odrzucone alternatywy

- **Opcja 2 (preprocessing w nodach Kedro)** — odrzucona. Dopasowane imputery, skalery i enkodery byłyby osobnymi artefaktami, które API musiałoby wczytać i wywołać w tej samej kolejności co pipeline. To prosta droga do training/serving skew. Łatwo też przypadkiem dopasować transformator na całym zbiorze przed splitem (leakage).
- **Opcja 3 (wszystko w sklearn `Pipeline`)** — odrzucona. Transformatory sklearn nie mogą usuwać wierszy ani zmieniać `y`, więc training-only filtering nie da się tam poprawnie umieścić. Dodatkowo filtrowanie wykonywane przy predykcji mogłoby odrzucić rekord użytkownika zamiast zwrócić wynik.

## Konsekwencje

- Ułatwia API (S4) wywołanie `normalize_input` na rekordzie, a potem `model.predict_proba`. Nie potrzebuje żadnej dodatkowej logiki preprocessingu.
- Ułatwia w S3 model produkcyjny pozostaje jednym artefaktem (pakiet modelu + `metadata.json`).
- Wymaga dyscypliny: każda nowa reguła w `normalize_input` musi działać na pojedynczym rekordzie i nie może używać statystyk zbioru. W przeciwnym razie należy ją przenieść do `Pipeline` (jeśli wymaga `fit`) albo do filteringu (jeśli usuwa rekordy).
- Do rozważenia: 125 duplikatów obecnie zostaje w danych. Ich ewentualne usunięcie to zmiana w `filter_training_rows` i aktualizacja tego ADR.



## Historia

| Data | Zmiana |
|---|---|
| 2026-11-07 | utworzenie |
