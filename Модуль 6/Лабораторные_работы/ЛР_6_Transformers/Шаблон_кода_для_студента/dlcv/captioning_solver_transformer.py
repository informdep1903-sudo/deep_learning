import numpy as np

from . import optim
from .coco_utils import sample_coco_minibatch, decode_captions

import torch


class CaptioningSolverTransformer(object):
    """
    CaptioningSolverTransformer инкапсулирует всю логику, необходимую для обучения моделей генерации
    подписей к изображениям на основе архитектуры Transformer.

    Для обучения модели необходимо сначала создать экземпляр CaptioningSolver, передав в конструктор модель,
    датасет и различные параметры (скорость обучения, размер батча и т.д.). 
    Затем следует вызвать метод train() для запуска процедуры оптимизации и обучения модели.

    После завершения метода train() переменная экземпляра solver.loss_history будет содержать
    список всех значений функции потерь, полученных в процессе обучения.

    Пример использования может выглядеть следующим образом:

    data = load_coco_data()
    model = MyAwesomeTransformerModel(hidden_dim=100)
    solver = CaptioningSolver(model, data,
                optim_config={
                  'learning_rate': 1e-3,
                },
                num_epochs=10, batch_size=100,
                print_every=100)
    solver.train()

    CaptioningSolverTransformer работает с объектом модели, который должен соответствовать следующему API:

    Входные данные:
    features: Массив, содержащий признаки изображений для мини-батча, размерности (N, D),
    где N — размер батча, D — размерность признакового пространства.
    captions: Массив подписей для этих изображений, размерности (N, T),
    где каждый элемент находится в диапазоне [0, V] (V — размер словаря).

    Возвращаемое значение:
    loss: Скалярное значение функции потерь.
    grads: Словарь с теми же ключами, что и self.params, который отображает имена параметров
    на градиенты функции потерь по этим параметрам.
  
    """

    def __init__(self, model, data, idx_to_word, **kwargs):
        """
        Конструктор cоздает новый экземпляр CaptioningSolver.

        Обязательные аргументы:
            model: Объект модели, соответствующий API, описанному выше.
            data: Словарь с данными для обучения и валидации, полученный из функции load_coco_data.

        Необязательные аргументы:
            learning_rate: Скорость обучения оптимизатора.
            batch_size: Размер мини-батча, используемого для вычисления функции потерь и градиента в процессе обучения.
            num_epochs: Количество эпох для обучения.
            print_every: Целое число; значения функции потерь будут выводиться каждые print_every итераций.
            verbose: Логический флаг; если установлен в False, то в процессе обучения не будет выводиться никакой информации.  
        """
        self.model = model
        self.data = data

        # Распаковка ключевых аргументов
        self.learning_rate = kwargs.pop("learning_rate", 0.001)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)

        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)
        self.optim = torch.optim.Adam(self.model.parameters(), self.learning_rate)

        # Генерировать исключение, если обнаружены неожиданные именованные аргументы.
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        self._reset()

        self.idx_to_word = idx_to_word

    def _reset(self):
        """
        Инициализировать вспомогательные переменные для оптимизации.
        Не вызывайте этот метод вручную.
        """
        # Установка некоторых переменных
        self.epoch = 0
        self.loss_history = []


    def _step(self):
        """
        Выполнить одно обновление градиента. Этот метод вызывается из train()
        и не должен вызываться вручную.
        """
        # Создание мини-батча
        minibatch = sample_coco_minibatch(
            self.data, batch_size=self.batch_size, split="train"
        )
        captions, features, urls = minibatch

        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        mask = captions_out != self.model._null

        t_features = torch.Tensor(features)
        t_captions_in = torch.LongTensor(captions_in)
        t_captions_out = torch.LongTensor(captions_out)
        t_mask = torch.LongTensor(mask)
        logits = self.model(t_features, t_captions_in)

        loss = self.transformer_temporal_softmax_loss(logits, t_captions_out, t_mask)
        self.loss_history.append(loss.detach().numpy())
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def train(self):
        """
        Выполнение оптимизатора для обучения модели
        """
        num_train = self.data["train_captions"].shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # Печать потерь обучения
            if self.verbose and t % self.print_every == 0:
                print(
                    "(Iteration %d / %d) loss: %f"
                    % (t + 1, num_iterations, self.loss_history[-1])
                )

            # Счетчик эпох
            epoch_end = (t + 1) % iterations_per_epoch == 0

    def transformer_temporal_softmax_loss(self, x, y, mask):
        """
        Временная версия функции потерь softmax для использования в RNN. Мы предполагаем,
        что делаем предсказания для словаря размера V на каждом временном шаге временного 
        ряда длины T для мини-батча размера N. Входной тензор x содержит оценки (логиты) 
        для всех элементов словаря на всех временных шагах, а y содержит индексы правильных
        (эталонных) элементов на каждом временном шаге. Мы используем перекрестную энтропию 
        на каждом временном шаге, суммируя потери по всем временным шагам и усредняя по мини-батчу.

        В качестве дополнительного усложнения мы можем захотеть игнорировать выходные данные
        модели на некоторых временных шагах, поскольку последовательности разной длины могут 
        быть объединены в один мини-батч и дополнены специальными токенами (NULL). Необязательный
        аргумент mask указывает, какие элементы должны учитываться при вычислении потерь.

        Входные данные:
            x: Входные оценки (логиты), размерности (N, T, V).
            y: Индексы правильных ответов, размерности (N, T), где каждый элемент находится в
            диапазоне 0 <= y[i, t] < V.
            mask: Логический массив размерности (N, T), где mask[i, t] указывает, должны ли оценки x[i, t] 
            учитываться при вычислении потерь.

        Возвращает кортеж из:
        loss: Скалярное значение функции потерь.    
        """

        N, T, V = x.shape

        x_flat = x.reshape(N * T, V)
        y_flat = y.reshape(N * T)
        mask_flat = mask.reshape(N * T)

        loss = torch.nn.functional.cross_entropy(x_flat,  y_flat, reduction='none')
        loss = torch.mul(loss, mask_flat)
        loss = torch.mean(loss)

        return loss
