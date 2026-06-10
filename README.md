# perceptual-image-gen

Генерация изображений через градиентную оптимизацию признаков **VGG19**. Два режима работы:

- **content** — восстановление изображения по перцептивному лоссу (MSE между feature maps)
- **texture** — синтез новой текстуры по Gram-матрицам признаков (style loss)

Оба режима стартуют со случайного шума и оптимизируют **только тензор изображения** — веса VGG19 заморожены.

---

## Какую задачу решает проект

### Режим `content` — перцептивный лосс

Нужно получить изображение, визуально похожее на оригинал, методом градиентной оптимизации.

Попиксельная MSE даёт размытые результаты: RGB-плохо отражает воспринимаемое сходство. **Перцептивный лосс** сравнивает карты признаков VGG19 на выбранном слое — так сохраняются структура, контуры и текстуры.

Основа: [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) (Johnson et al., 2016).

### Режим `texture` — генерация текстуры

Нужно сгенерировать **новое** изображение, которое перенимает **текстуру** референса, а не копирует его содержимое.

Для каждого слоя VGG вычисляется **Gram-матрица** — она описывает статистику корреляций между каналами признаков, то есть характер текстуры. Оптимизатор подбирает пиксели так, чтобы Gram-матрицы сгенерированного изображения совпали с референсом.

Основа: [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) (Gatys et al., 2015).

---

## Режимы работы

| | `content` | `texture` |
|---|---|---|
| **Цель** | Похоже на оригинал | Новая картинка с текстурой референса |
| **Лосс** | MSE между feature maps | MSE между Gram-матрицами |
| **Слои VGG** | 1 слой (`block2_conv2`, `block4_conv1`, `block5_conv3`) | 5 слоёв (`block1_conv1` … `block5_conv1`) |
| **Шаги** | 500 | 2500 |
| **Learning rate** | 0.2 | 0.05 |
| **TV weight** | 0.01 | 10000 |
| **Сравнение** | Original \| Generated | Style reference \| Generated texture |

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

При первом запуске TensorFlow скачает веса VGG19 (~550 MB).

### Запуск

```bash
# Восстановление изображения (content)
python -m perceptual_gen -i data/sample.jpg -o outputs/content --mode content

# Генерация текстуры (texture)
python -m perceptual_gen -i data/sample.jpg -o outputs/texture --mode texture
```

Результат каждого запуска:

```
outputs/<run>/
├── result.png       # сгенерированное изображение
├── comparison.png   # референс | результат
└── config.json      # параметры и метрики
```

---

## Примеры

### Content — перцептивный лосс

```bash
# Базовый запуск
python -m perceptual_gen -i data/sample.jpg -o outputs/default --mode content

# Ранний слой — больше мелких деталей
python -m perceptual_gen -i data/sample.jpg -o outputs/block2 -l block2_conv2

# Глубокий слой — крупная структура
python -m perceptual_gen -i data/sample.jpg -o outputs/block5 -l block5_conv3

# Сравнение всех трёх слоёв
python -m perceptual_gen -i data/sample.jpg -o outputs/compare_all -l all
```

### Texture — генерация текстуры

```bash
# Базовый запуск (дефолты из ноутбука: 2500 шагов, lr=0.05, tv=10000)
python -m perceptual_gen -i data/sample.jpg -o outputs/texture --mode texture

# Другие гиперпараметры — для сравнения 2–3 вариантов
python -m perceptual_gen -i data/sample.jpg -o outputs/texture_fast --mode texture --steps 1000 --lr 0.1
python -m perceptual_gen -i data/sample.jpg -o outputs/texture_smooth --mode texture --tv-weight 5000

# Пакетный прогон нескольких текстур из папки
python -m perceptual_gen --input-dir data/textures -o outputs/textures_batch --mode texture
```

При `--input-dir` каждое изображение сохраняется в отдельную подпапку:

```
outputs/textures_batch/
├── pebbles/
│   ├── result.png
│   ├── comparison.png
│   └── config.json
└── ...
```

---

## Как это работает

### Content mode

```
Исходное изображение → VGG19 → target features
Случайный шум → оптимизация → MSE(features) + λ·TV → result.png
```

### Texture mode

```
Референс текстуры → VGG19 → Gram-матрицы (target)
Случайный шум → оптимизация → MSE(Gram) + λ·TV → result.png
```

**Важно:** в обоих режимах обучается не нейросеть, а **само изображение**.

### Влияние слоя (content mode)

| Слой | Характер результата |
|---|---|
| `block2_conv2` | Мелкие детали и текстуры |
| `block4_conv1` | Баланс структуры и деталей |
| `block5_conv3` | Крупные формы, меньше мелочи |

---

## Параметры CLI

| Флаг | Короткий | Content | Texture | Описание |
|---|---|---|---|---|
| `--input` | `-i` | ✓ | ✓ | Путь к одному изображению |
| `--input-dir` | | | ✓ | Папка с несколькими текстурами |
| `--output-dir` | `-o` | ✓ | ✓ | Папка для результатов |
| `--mode` | `-m` | `content` | `texture` | Режим генерации |
| `--layer` | `-l` | `block4_conv1` | — | Слой VGG или `all` (только content) |
| `--steps` | `-s` | `500` | `2500` | Число шагов оптимизации |
| `--lr` | | `0.2` | `0.05` | Learning rate (Adam) |
| `--max-dim` | | `256` | `256` | Макс. размер длинной стороны |
| `--tv-weight` | | `0.01` | `10000` | Вес total variation |
| `--seed` | | `42` | `42` | Seed для воспроизводимости |
| `--save-every` | | `0` | `0` | Снимок каждые N шагов (`0` — выкл.) |

---

## Стек технологий

| Компонент | Технология |
|---|---|
| ML framework | TensorFlow 2.x |
| Feature model | VGG19 (Keras Applications, ImageNet) |
| Content loss | MSE по feature maps |
| Style loss | MSE по Gram-матрицам |
| Optimizer | Adam |
| CLI | argparse |
| Тесты | pytest |

---

## Структура проекта

```
perceptual-image-gen/
├── perceptual_gen/
│   ├── cli.py
│   ├── config.py
│   ├── image_io.py
│   ├── vgg_model.py
│   ├── vgg_extractor.py      # content features
│   ├── style_extractor.py    # Gram-матрицы
│   ├── loss.py               # perceptual loss
│   ├── style_loss.py         # style loss
│   ├── optimizer.py
│   ├── pipeline.py
│   └── visualization.py
├── data/
│   ├── sample.jpg
│   └── textures/             # примеры для batch texture
├── tests/
├── requirements.txt
└── README.md
```

---

## Тесты

```bash
pytest
```

Покрытие: загрузка изображений, content/style loss с градиентом, smoke-тесты CLI для обоих режимов.

---

## Производительность

| Режим | Условия | Ориентир |
|---|---|---|
| content | CPU, 256px, 500 шагов | несколько минут |
| texture | CPU, 256px, 2500 шагов | 15–30+ минут |
| оба | GPU | значительно быстрее |

Для ускорения уменьшите `--max-dim` или `--steps`.

---

## Ссылки

- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
- [TensorFlow VGG19](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg19/VGG19)
