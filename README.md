# crispy-journey

Высокопроизводительная реализация алгоритма **HNSW** для приближённого поиска *k* ближайших соседей. Проект предназначен для проведения вычислительных экспериментов: построения индекса и измерения точности (recall), числа вычислений расстояния и времени поиска на стандартных ANN-датасетах.

В реализации применяются низкоуровневые оптимизации: SIMD-векторизация (AVX), программный префетчинг, непрерывное хранение точек, выравнивание памяти и использование huge pages. Дополнительно доступны опциональные оптимизации, включаемые флагами в `config.h`: перенумерация навигационного графа (*reordering*), пакетное вычисление расстояний (*batching*) и оптимизированная продуктовая квантизация (*OPQ*).

## Структура проекта

| Файл / каталог      | Назначение |
|---------------------|------------|
| `main.cpp`          | Точка входа; выполняет одну из задач `Create` / `Benchmark` / `Reorder`. |
| `hnsw.h`            | `HNSW<Space>` — построение графа, перенумерация, сохранение индекса. |
| `hnsw_inference.h`  | `HNSWInference<Space>` — оптимизированный поиск (batching, PQ). |
| `primitives.h`      | Метрики `SpaceL2` и `SpaceCosine`, выделение памяти. |
| `config.h`          | Конфигурация запуска. |
| `scripts/`          | Python-скрипты подготовки данных и построения графиков. |

## Сборка

```bash
git clone --recurse-submodules https://github.com/antonov1lya/crispy-journey.git
cd crispy-journey && mkdir build && cd build
cmake .. && cmake --build .
```

Для сборки необходимы компилятор C++ с поддержкой AVX2, CMake версии не ниже 3.5 и операционная система Linux с поддержкой huge pages. Флаги компиляции, включая `-march=native`, заданы в `CMakeLists.txt`, поэтому сборку следует выполнять на целевой машине. Результатом сборки является исполняемый файл `anns`.

## Конфигурация

Вся настройка выполняется через директивы `#define` в файле `config.h` на этапе компиляции; после внесения изменений проект необходимо пересобрать. Аргументы командной строки не используются.

- **Датасет** (выбирается ровно один): `SIFT1M`, `GLOVE100`, `GIST1M`, `DEEP1B` или `FASHION_MNIST`. Выбор датасета определяет размерность `SIZE` и метрику `SPACE`.
- **Задача** (выбирается ровно одна): `CREATE_TASK`, `BENCHMARK_TASK` или `REORDER_TASK`.
- **Тип перенумерации** (для `REORDER_TASK`): `REORDERING_TYPE_LOCAL_SEARCH`, `REORDERING_TYPE_BFS` или `REORDERING_TYPE_MST`.

Прочие опции: `PQ` (продуктовая квантизация при инференсе), `BATCH` (пакетное вычисление расстояний), `REORDER` (возврат исходных идентификаторов точек), `FIND_EF` (подбор `ef` под заданную полноту, верхняя граница задаётся `EF_R`), `SUBSPACES` и `BITS` (параметры PQ, по умолчанию 32 и 256 соответственно).

Параметры построения индекса (`M`, `efConstruction`, число точек `n`) определены в функции `Create()` в `main.cpp` отдельно для каждого датасета.

Имена файлов индекса и лога также задаются в `config.h` и могут быть изменены: `CREATE_TASK_INDEX_FILE` (создаваемый индекс), `BENCHMARK_TASK_INDEX_FILE` (индекс для бенчмарка), `BENCHMARK_TASK_LOG_FILE` (файл результатов), `REORDERING_TASK_INDEX_FILE` (перенумерованный индекс). По умолчанию используются `base.bin`, `base.csv` и `reordering.bin`.
## Данные

Программа ожидает следующую раскладку файлов относительно рабочего каталога:

```
datasets/<dataset>/data.bin, query.bin, groundtruth.bin
datasets/<dataset>/data_pq<S>.bin, centroids<S>.bin, matrix<S>.bin   # только при PQ
indexes/<dataset>/base.bin
logs/<dataset>/base.csv
```

Здесь `<dataset>` принимает значения `sift1m`, `glove100`, `gist1m`, `deep1b` или `fashion_mnist`, а `<S>` соответствует значению `SUBSPACES`. Каталоги `indexes/` и `logs/` необходимо создать заранее.

Исходные датасеты используются в формате HDF5 из репозитория [erikbern/ann-benchmarks](https://github.com/erikbern/ann-benchmarks):

| `config.h`      | HDF5-файл                          | `--angular` |
|-----------------|------------------------------------|-------------|
| `SIFT1M`        | `sift-128-euclidean.hdf5`          | нет         |
| `GLOVE100`      | `glove-100-angular.hdf5`           | да          |
| `GIST1M`        | `gist-960-euclidean.hdf5`          | нет         |
| `DEEP1B`        | `deep-image-96-angular.hdf5`       | да          |
| `FASHION_MNIST` | `fashion-mnist-784-euclidean.hdf5` | нет         |

Конвертация датасета и, при использовании `PQ`, обучение квантователя выполняются следующим образом:

```bash
wget http://ann-benchmarks.com/sift-128-euclidean.hdf5
python scripts/convert_hdf5.py sift-128-euclidean.hdf5 -o datasets/sift1m
python scripts/train_pq.py --input_file sift-128-euclidean.hdf5 \
    --output_dir datasets/sift1m --M <SUBSPACES> [--angular]
```

Значение `--M` в `train_pq.py` должно совпадать с `SUBSPACES` в `config.h`; флаг `--angular` указывается для косинусных датасетов. Зависимости скриптов: `numpy`, `pandas`, `matplotlib`, `h5py`, `scikit-learn`, `faiss`. Скрипты `draw_graphs.py` и `draw_graphs1.py` строят графики зависимости QPS от полноты по CSV-логам; конфигурация передаётся в формате JSON через аргумент `--json-path`.

## Запуск

```bash
sudo sysctl vm.nr_hugepages=2048
./build/anns
```

Типичный порядок работы: задача `CREATE_TASK` строит индекс `base.bin`; при необходимости `REORDER_TASK` формирует перенумерованный индекс `reordering.bin`; задача `BENCHMARK_TASK` записывает результаты в `logs/<dataset>/base.csv`.

Формат CSV-лога — `recall,ef,dst,time`: полнота recall@10, ширина поиска, среднее число вычислений расстояния на один запрос и время в наносекундах. Для каждого значения `ef` из сетки, заданной в `Benchmark()`, выполняется 10 повторов.

## Программный интерфейс

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
