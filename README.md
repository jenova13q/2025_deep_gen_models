# ДЗ 3 — Sampling в латентном пространстве StyleGAN

---

## Датасет и подготовка
- Кадры выровнены под FFHQ-пайплайн (**Align images**) с детектором 68 landmark (dlib).
- Модель: **StyleGAN2-ADA (FFHQ)**.
- Латент: **W+** (18×512).

Папки с результатами:
```
reports/
    table1/                     # инверсий
    style_transfer/             # стили
    expression_transfer/        # эмоции
    face_swap/                  # свап
```

---

## Инверсия (Projection)

Описание:
Отразила изображения в W+ двумя способами: энкодером (e4e) и градиентной оптимизацией.

**Оригинал | e4e | Optimization** 
![Table1](/reports/table1/AJoly_0_table1.jpg)
![Table2](/reports/table1/EMerphy_0_table1.jpg)
![Table3](/reports/table1/KJunJin_0_table1.jpg)
![Table4](/reports/table1/LDCaprio_0_table1.jpg)
![Table5](/reports/table1/Stethem_0_table1.jpg)

---

## Style Transfer

Описание:
Стили брались из трёх картинок в /content/style (каждая предварительно проецировалась в W+ с e4e).
Для переноса цвета/света/текстуры при сохранении личности смешивали вектора на FINE-уровнях (W+ индексы 8–17).

![StyleMix](/reports/style_transfer/__style_inputs.jpg)
![AJoly](/reports/style_transfer/AJoly_2_stylemix.jpg)
![EMerphy](/reports/style_transfer/EMerphy_2_stylemix.jpg)
![KJunJin](/reports/style_transfer/KJunJin_2_stylemix.jpg)

Вывод:
Стиль (цвет/свет/текстура) переносится, личность сохраняется.

---

## Expression Transfer

Описание:
Использовали три edit-направления из /editing: smile, pose, age.
Применяли их к W+ одной базовой фотографии как аддитивные правки с коэффициентом α≈1.5.

**Оригинал | smile | pose | age**
![Expr1](/reports/expression_transfer/AJoly_2_expressions.jpg)
![Expr2](/reports/expression_transfer/EMerphy_2_expressions.jpg)
![Expr3](/reports/expression_transfer/KJunJin_2_expressions.jpg)

---

## Face Swap (ArcFace)

Описание:
Добавили ArcFace loss в пайплайн оптимизации W+ таргета (старт — e4e инверсия таргета).
Оптимизировали по комбинации: LPIPS + Rec + NoiseReg + ArcFace.
Делали прогон «каждый с каждым» (self-swap пропускали).
Добавили логирование лоссов (каждые N шагов) и early stop по метрике ArcFace.

steps=200, arc_w=10.0, rec_w=0.1, lpips_w=0.5

![SwapTriplet](/reports/face_swap/AJoly_to_EMerphy.jpg) ![SwapTriplet](/reports/face_swap/EMerphy_to_AJoly.jpg)
![SwapTriplet](/reports/face_swap/LDCaprio_to_Stethem.jpg) ![SwapTriplet](/reports/face_swap/Stethem_to_LDCaprio.jpg)
![SwapTriplet](/reports/face_swap/KJunJin_to_LDCaprio.jpg) ![SwapTriplet](/reports/face_swap/LDCaprio_to_KJunJin.jpg)
![SwapTriplet](/reports/face_swap/Stethem_to_AJoly.jpg) ![SwapTriplet](/reports/face_swap/AJoly_to_Stethem.jpg)

