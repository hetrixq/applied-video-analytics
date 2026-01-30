# Motion Detection Baseline

Сквозной baseline-пайплайн для задачи **обнаружения движения** в видеопотоке.  
Формулировка: **бинарная классификация по временным окнам** ( `motion` / `no_motion` ).

## Что входит в пайплайн

* загрузка видео из `data/raw/` по спискам `splits/*.txt`
* фиксированный экспериментальный протокол: разбиение по **видео**, а не по кадрам
* простая стандартная модель: **классификатор по порогу** для `motion_score`
* процедура обучения: подбор порога на train (grid search)
* процедура оценки: accuracy/precision/recall/f1 (+ TP/TN/FP/FN)
* сохранение результатов в `runs/<run_name>/`

## Структура

```
.
├── configs/baseline.yaml
├── data/
│   ├── raw/                 # положить сюда видео
│   └── annotations/annotations.csv
├── splits/train.txt
├── splits/val.txt
├── splits/test.txt
├── src/
│   ├── baseline.py
│   ├── data.py
│   ├── model.py
│   └── utils.py
└── runs/
```

## Аннотации (разметка)

Файл `data/annotations/annotations.csv` с интервалами движения:

* `video` — путь относительно `data/raw/` (должен совпадать со строками в splits)
* `t_start_sec` — начало интервала движения (сек)
* `t_end_sec` — конец интервала движения (сек)

Пример:

```csv
video,t_start_sec,t_end_sec
cam1/scene_001.mp4,3.2,8.9
cam1/scene_001.mp4,15.0,16.4
```

Если видео целиком без движения — его можно не указывать в CSV (оно будет считаться all-negative).

## Как запустить

1) Добавь видео в `data/raw/` (можно с подпапками).
2) Заполни `splits/train.txt` , `splits/val.txt` , `splits/test.txt` путями к видео (по одному на строку).
3) Заполни `data/annotations/annotations.csv` .
4) Запусти:

```bash
python -m src.baseline --config configs/baseline.yaml
```

Результаты будут в `runs/<run_name>/` :
* `params.yaml` — использованный конфиг
* `model.json` — найденный порог
* `metrics.json` — метрики train/val/test
* `predictions.csv` — предикты по каждому окну времени

## Воспроизводимость

Обеспечивается:
* фиксированными split-файлами
* фиксированным конфигом (fps/resize/окна/пороги/метрики)
* фиксированным `random_seed`
