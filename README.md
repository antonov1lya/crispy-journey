# crispy-journey

Высокопроизводительная реализация **HNSW** для приближённого поиска *k* ближайших соседей. Ориентирована на вычислительные эксперименты: построение индекса и замеры recall / числа вычислений расстояния / времени поиска на стандартных ANN-датасетах.

Низкоуровневые оптимизации: SIMD-векторизация (AVX), программный префетчинг, непрерывное хранение точек, выравнивание памяти, huge pages. Опциональные (флаги в `config.h`): перенумерация графа (*reordering*), пакетное вычисление расстояний (*batching*), оптимизированная продуктовая квантизация (*OPQ*).

## Структура

| Файл / каталог       | Назначение |
|----------------------|------------|
| `main.cpp`           | Точка входа: задачи `Create` / `Benchmark` / `Reorder`. |
| `hnsw.h`             | `HNSW<Space>` — построение графа, перенумерация, сохранение. |
| `hnsw_inference.h`   | `HNSWInference<Space>` — оптимизированный поиск (batching, PQ). |
| `primitives.h`       | Метрики `SpaceL2` / `SpaceCosine`, выделение памяти. |
| `config.h`           | Вся конфигурация запуска. |
| `scripts/`           | Python-скрипты подготовки данных и графиков. |

## Сборка

```bash
git clone --recurse-submodules https://github.com/antonov1lya/crispy-journey.git
cd crispy-journey && mkdir build && cd build
cmake .. && cmake --build .
```

Требуется компилятор C++ с AVX2 и OpenMP, CMake ≥ 3.5, Linux с huge pages. Флаги (включая `-march=native`) заданы в `CMakeLists.txt`; собирать нужно на целевой машине. Результат — бинарь `anns`.

## Конфигурация

Вся настройка — через `#define` в `config.h`, **на этапе компиляции** (после правки нужна пересборка; аргументов командной строки нет).

- **Датасет** (ровно один): `SIFT1M` / `GLOVE100` / `GIST1M` / `DEEP1B` / `FASHION_MNIST` — задаёт `SIZE` и метрику `SPACE`.
- **Задача** (ровно одна): `CREATE_TASK` / `BENCHMARK_TASK` / `REORDER_TASK`.
- **Тип перенумерации** (для `REORDER_TASK`): `REORDERING_TYPE_LOCAL_SEARCH` / `_BFS` / `_MST`.

Прочие опции: `PQ` (квантизация при инференсе), `BATCH` (пакетные расстояния), `REORDER` (возврат исходных id), `FIND_EF` (подбор `ef` под recall, граница `EF_R`), `SUBSPACES`/`BITS` (параметры PQ, по умолчанию 32 / 256).

Параметры построения (`M`, `efConstruction`, `n`) зашиты в `Create()` в `main.cpp` для каждого датасета.

## Данные

Программа ожидает раскладку относительно рабочего каталога:

```
datasets/<dataset>/data.bin, query.bin, groundtruth.bin
datasets/<dataset>/data_pq<S>.bin, centroids<S>.bin, matrix<S>.bin   # только при PQ
indexes/<dataset>/base.bin
logs/<dataset>/base.csv
```

`<dataset>`: `sift1m`, `glove100`, `gist1m`, `deep1b`, `fashion_mnist`. `<S>` = `SUBSPACES`. Каталоги `indexes/` и `logs/` нужно создать заранее.

Исходные датасеты — HDF5 из [erikbern/ann-benchmarks](https://github.com/erikbern/ann-benchmarks):

| `config.h`      | HDF5-файл                          | `--angular` |
|-----------------|------------------------------------|-------------|
| `SIFT1M`        | `sift-128-euclidean.hdf5`          | нет         |
| `GLOVE100`      | `glove-100-angular.hdf5`           | да          |
| `GIST1M`        | `gist-960-euclidean.hdf5`          | нет         |
| `DEEP1B`        | `deep-image-96-angular.hdf5`       | да          |
| `FASHION_MNIST` | `fashion-mnist-784-euclidean.hdf5` | нет         |

Конвертация и (для `PQ`) обучение квантователя:

```bash
wget http://ann-benchmarks.com/sift-128-euclidean.hdf5
python scripts/convert_hdf5.py sift-128-euclidean.hdf5 -o datasets/sift1m
python scripts/train_pq.py --input_file sift-128-euclidean.hdf5 \
    --output_dir datasets/sift1m --M <SUBSPACES> [--angular]
```

`--M` в `train_pq.py` должно совпадать с `SUBSPACES` в `config.h`. `--angular` — для косинусных датасетов. Зависимости скриптов: `numpy`, `pandas`, `matplotlib`, `h5py`, `scikit-learn`, `faiss`. Скрипты `draw_graphs.py` / `draw_graphs1.py` строят графики QPS-vs-recall по CSV-логам (конфиг — JSON через `--json-path`).

## Запуск

```bash
sudo sysctl vm.nr_hugepages=2048
taskset -c 1 ./build/anns
```

Типичный цикл: `CREATE_TASK` → построить `base.bin`; опционально `REORDER_TASK` → `reordering.bin`; `BENCHMARK_TASK` → результаты в `logs/<dataset>/base.csv`.

CSV-лог: `recall,ef,dst,time` (recall@10, ширина поиска, среднее число вычислений расстояния на запрос, время в нс). Для каждого `ef` из сетки в `Benchmark()` — 10 повторов после прогрева.

## API

```cpp
// hnsw.h — построение
HNSW(IntType M, IntType ef_construction, IntType max_elements, std::ifstream& file_data);
HNSW(std::ifstream& file, std::ifstream& file_data);
void Add(int level);
std::vector<IntType> Search(FloatType* query, IntType K, IntType ef);
void ReOrdering();
void Save(std::ofstream& file);

// hnsw_inference.h — инференс
HNSWInference(std::ifstream& file, std::ifstream& file_data);
std::vector<IntType> Search(FloatType* query, IntType K, IntType ef);
std::vector<IntType> SearchPQ(FloatType* query, IntType K, IntType ef);
void LoadPQ(std::ifstream& file_data_pq, std::ifstream& file_centroids);
void LoadPQMatrix(std::ifstream& file_matrix);
```

