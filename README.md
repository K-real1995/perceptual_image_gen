# perceptual-image-gen

Генерация изображений через градиентную оптимизацию с **перцептивным лоссом** на признаках VGG19.

Утилита принимает исходную картинку и итеративно «восстанавливает» её из случайного шума, сравнивая не пиксели напрямую, а **признаки нейросети** — так же, как это делают современные подходы к переносу стиля и суперразрешению.

---

## Какую задачу решает проект

Представьте, что нужно получить изображение, похожее на оригинал, методом оптимизации: стартуем со случайного шума и на каждом шаге подстраиваем пиксели так, чтобы результат становился ближе к цели.

Самый простой способ — минимизировать **попиксельную разницу** (MSE между пикселями). На практике это даёт размытые, «мыльные» результаты: пространство RGB плохо отражает визуальное сходство для человека.

**Перцептивный лосс** решает эту проблему иначе:

1. Пропускаем оригинал и сгенерированное изображение через предобученную **VGG19** (ImageNet).
2. Сравниваем **карты признаков** на выбранном свёрточном слое.
3. Оптимизируем только тензор изображения — веса VGG остаются замороженными.

Так оптимизатор учитывает структуру, текстуры и контуры, а не только совпадение цветов по пикселям.

Подход описан в работе [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) (Johnson et al., 2016).

---

## Что делает утилита

| Возможность | Описание |
|---|---|
| Загрузка изображения | Чтение файла с диска, приведение к нужному размеру |
| Извлечение признаков | VGG19, слои `block2_conv2`, `block4_conv1`, `block5_conv3` |
| Оптимизация | Градиентный спуск (Adam) по тензору изображения |
| Регуляризация | Total Variation — сглаживает шум, стабилизирует результат |
| Сравнение слоёв | Флаг `--layer all` — прогон по трём слоям и сводка |
| Артефакты | `result.png`, `comparison.png`, `config.json` |

После запуска в папке результатов появляется **side-by-side сравнение** «оригинал | сгенерированное» — основной наглядный результат.

---

## Как это работает

```
Исходное изображение
        │
        ▼
   VGG19 (frozen) ──► target features (один раз)
        │
        ▼
Случайный шум ──► tf.Variable (оптимизируемый тензор)
        │
        ▼
   Цикл оптимизации (N шагов):
     1. Пропуск через VGG19 → current features
     2. Loss = MSE(features) + λ · TotalVariation
     3. Градиент по изображению → шаг Adam
     4. Clip значений в [0, 1]
        │
        ▼
   result.png + comparison.png
```

**Важно:** обучается не нейросеть, а **само изображение**. VGG19 выступает как фиксированный «измеритель сходства».

### Влияние выбора слоя

| Слой | Уровень абстракции | Характер результата |
|---|---|---|
| `block2_conv2` | Низкий (ранние свёртки) | Больше мелких деталей и текстур |
| `block4_conv1` | Средний | Баланс структуры и деталей |
| `block5_conv3` | Высокий (глубокие слои) | Крупные формы, меньше мелочи |

---

## Быстрый старт

### Требования

- Python 3.10+
- ~2 GB свободного места (TensorFlow + веса VGG19)

### Установка

```bash
git clone <repository-url>
cd perceptual-image-gen

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

При первом запуске TensorFlow автоматически скачает веса VGG19 (~550 MB).

### Запуск

```bash
python -m perceptual_gen --input data/sample.jpg --output-dir outputs/default
```

В консоли отобразится прогресс-бар оптимизации. По завершении:

```
outputs/default/
├── result.png       # итоговое изображение
├── comparison.png   # оригинал | результат
└── config.json      # параметры запуска и метрики
```

---

## Примеры использования

```bash
# Базовый запуск (слой block4_conv1, 500 шагов)
python -m perceptual_gen -i data/sample.jpg -o outputs/default

# Ранний слой — акцент на текстурах
python -m perceptual_gen -i data/sample.jpg -o outputs/block2 -l block2_conv2

# Глубокий слой — акцент на крупной структуре
python -m perceptual_gen -i data/sample.jpg -o outputs/block5 -l block5_conv3

# Промежуточные снимки каждые 100 шагов
python -m perceptual_gen -i data/sample.jpg -o outputs/block5 -l block5_conv3 --save-every 100

# Сравнение всех трёх слоёв за один запуск
python -m perceptual_gen -i data/sample.jpg -o outputs/compare_all -l all
```

Структура при `--layer all`:

```
outputs/compare_all/
├── block2_conv2/
│   ├── result.png
│   ├── comparison.png
│   └── config.json
├── block4_conv1/
│   └── ...
├── block5_conv3/
│   └── ...
└── summary.json
```

---

## Параметры CLI

| Флаг | Короткий | По умолчанию | Описание |
|---|---|---|---|
| `--input` | `-i` | — | Путь к исходному изображению |
| `--output-dir` | `-o` | — | Папка для результатов |
| `--layer` | `-l` | `block4_conv1` | Слой VGG19 или `all` |
| `--steps` | `-s` | `500` | Число шагов оптимизации |
| `--lr` | | `0.2` | Learning rate (Adam) |
| `--max-dim` | | `256` | Макс. размер длинной стороны (px) |
| `--tv-weight` | | `0.01` | Вес total variation |
| `--seed` | | `42` | Seed для воспроизводимости |
| `--save-every` | | `0` | Сохранять снимок каждые N шагов (`0` — выкл.) |

---

## Стек технологий

| Компонент | Технология |
|---|---|
| ML framework | TensorFlow 2.x |
| Feature extractor | VGG19 (Keras Applications, ImageNet) |
| Optimizer | Adam |
| Loss | MSE по feature maps + Total Variation |
| CLI | argparse |
| Визуализация | matplotlib |
| Тесты | pytest |

---

## Структура проекта

```
perceptual-image-gen/
├── perceptual_gen/
│   ├── cli.py              # парсинг аргументов, точка входа CLI
│   ├── config.py           # GenerationConfig, валидация параметров
│   ├── image_io.py         # загрузка, resize, сохранение
│   ├── vgg_extractor.py    # FeatureExtractor, VGG19
│   ├── loss.py             # перцептивный лосс + TV
│   ├── optimizer.py        # цикл градиентной оптимизации
│   ├── pipeline.py         # оркестрация полного прогона
│   └── visualization.py    # side-by-side сравнение
├── data/
│   └── sample.jpg          # пример для быстрого старта
├── tests/
├── requirements.txt
└── README.md
```

---

## Тесты

```bash
pytest
```

Покрытие включает загрузку изображений, вычисление лосса с градиентом и smoke-тест CLI (`--steps 2`).

---

## Производительность

| Условия | Ориентир |
|---|---|
| CPU, `max_dim=256`, `steps=500` | несколько минут |
| GPU | значительно быстрее |
| Первый запуск | + время на загрузку весов VGG19 |

Для ускорения на слабом железе уменьшите `--max-dim` или `--steps`.

---

## Ссылки

- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) — оригинальная статья
- [TensorFlow VGG19](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg19/VGG19) — документация модели
- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) — классический neural style transfer (Gatys et al.)
