# Проект для учебной практики

## Подготовка датасета

Положить датасет GTSDB в папку `data/raw/`.

Подходит такая структура:

```text
data/raw/
  FullIJCNN2013/
    00000.ppm
    00001.ppm
    ...
    gt.txt
```

Запуск подготовки:

```bash
python main.py prepare --config configs/default.yaml
```

После выполнения появится папка `data/processed/` с подготовленными данными.

Анализ датасета;

```bash
python main.py analyze --config configs/default.yaml
```

Команда строит таблицы и графики по датасету:

- распределение объектов по классам;
- анализ размеров bounding box.

Результаты сохраняются в:

```text
results/metrics/
results/plots/
```

## Обучение моделей

Обучение одной модели:

```bash
python main.py train --config configs/default.yaml --model yolov8n
python main.py train --config configs/default.yaml --model fasterrcnn_mobilenet
python main.py train --config configs/default.yaml --model ssdlite
python main.py train --config configs/default.yaml --model efficientdet_d0
python main.py train --config configs/default.yaml --model detr_resnet50
```

Обучение всех моделей:

```bash
python main.py train-all --config configs/default.yaml
```

Во время обучения каждая модель оценивается на `val`-выборке. Лучший checkpoint и история обучения сохраняются в:

```text
results/logs/<model_name>/
```

Общие метрики после обучения записываются в:

```text
results/metrics/experiments.csv
results/metrics/experiments.jsonl
```

## Графики результатов

После обучения можно построить графики:

```bash
python main.py plot --config configs/default.yaml
```

Они сохраняются в:

```text
results/plots/
```

Команда:

- сравнивает моделей по `mAP`, `mAP50`, `Precision`, `Recall`, `F1`;
- строит графики обучения по эпохам для каждой модели.

## Основные настройки

Главный файл конфигурации:

```text
configs/default.yaml
```

В нём задаются:

- пути к данным;
- количество классов;
- названия классов;
- параметры обучения;
- список моделей;
- параметры оценки;
- пути для сохранения результатов.

