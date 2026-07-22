import numpy as np
import copy

import torch
import torch.nn as nn

from ..transformer_layers import *


class CaptioningTransformer(nn.Module):
    """
    CaptioningTransformer создает подписи по признакам изображений с помощью декодера трансформера.

    Трансформер получает входные векторы размера D, имеет размер словаря V,
    работает с последовательностями длины T, использует векторы слов размерности W и
    работает с мини-пакетами размера N.
    
    """
    def __init__(self, word_to_idx, input_dim, wordvec_dim, num_heads=4,
                 num_layers=2, max_length=50):
        """
        Конструктор класса CaptioningTransformer.

        Входы:
        - word_to_idx: Словарь, содержащий V слов, и 
          отображающий слово в  число в диапазоне [0, V).
        - input_dim: Размерность D вектора признаков входного изображения.
        - wordvec_dim: Размерность W вектора слов.
        - num_heads: Число голов внимания.
        - num_layers: Число слоев трансформера.
        - max_length: Максимально возможная длина последовательности.
        """
        super().__init__()

        vocab_size = len(word_to_idx)
        self.vocab_size = vocab_size
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)
        
        # Инициализация слоя преобразования изображения к размеру wordvec_dim
        self.visual_projection = nn.Linear(input_dim, wordvec_dim)
        # Инициализация слоя встраивания слова (матрицы векторов слов)
        self.embedding = nn.Embedding(vocab_size, wordvec_dim, padding_idx=self._null)
        # Инициализация слоя позиционного кодирования
        self.positional_encoding = PositionalEncoding(wordvec_dim, max_len=max_length)

        # Инициализация слоя декодера Трансформера
        decoder_layer = TransformerDecoderLayer(input_dim=wordvec_dim, num_heads=num_heads)
        # Инициализация декодера Трансформера из слоев TransformerDecoderLayer
        self.transformer = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.apply(self._init_weights)
        
        # Инициализация выходного линейного слоя
        self.output = nn.Linear(wordvec_dim, vocab_size)

    def _init_weights(self, module):
        """
       Инициализация весов сети.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, features, captions):
        """
        На основе признаков изображения и токенов подписей, возвращает распределение по
        возможным токенам для каждого временного шага. Обратите внимание, что поскольку вся
        последовательность подписей предоставляется сразу, мы маскируем будущие временные шаги.

        Входы:
         - features: признаки изображения, форма (N, D)
         - captions: правильные подписи, форма (N, T)

        Возвращает:
         - scores: рейтинг каждого токена на каждом временном, форма (N, T, V)
        """
        N, T = captions.shape
        # Создаём пустой тензор, который будет заполнен с помошью кода ниже.
        scores = torch.empty((N, T, self.vocab_size))
        ############################################################################
        # ЗАДАНИЕ: Реализовать функцию forward для CaptionTransformer.             #
        # Несколько подсказок:                                                     #
        #  1) Сначала необходимо встроить подпись и добавить позицонное кодирование#
        #  Затем необходимо спроецировать изображения в эту же размерность.        #
        #  2) Вам необходимо создать маску (tgt_mask) для маскирования будущих     #
        #  токенов в подписях. Для создания маски может быть использована функция  #
        #  torch.tril().                                                           #
        #  3) Окончательно, примените признаки декoдера к тексту и встроенному     #
        #  изображению вместе с tgt_mask. Отобразите выход на рейтинги             #
        #  scores каждого токена                                                   #
        ############################################################################
        # ***** НАЧАЛО ВАШЕГО КОДА *****
        
        pass

        # ***** КОНЕЦ ВАШЕГО КОДА *****
        ############################################################################
        #                             КОНЕЦ ВАШЕГО КОДА                            #
        ############################################################################

        return scores

    def sample(self, features, max_length=30):
        """
        Используя признаки изображения, применяет жадное декодирование, чтобы предсказать подпись к изображению.

        Входы:
         - features: признаки изображения, форма (N, D)
         - max_length: максимально возможная длина подписи

        Возвращает:
         - captions: подписи к каждому входному изображению, форма (N, max_length)
        """
        with torch.no_grad():
            features = torch.Tensor(features)
            N = features.shape[0]

           # Создание пустого тензора подписи (где все токены равны NULL).
            captions = self._null * np.ones((N, max_length), dtype=np.int32)

            # Создание частичной подписи только со стартовым токеном.
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption)
            # [N] -> [N, 1]
            partial_caption = partial_caption.unsqueeze(1)

            for t in range(max_length):

                # Предсказание следующего токена (игнорируя все остальные временные шаги).
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]

                # Выбор наиболее вероятного идентификатора слова из словаря.
                # [N, V] -> [N]
                word = torch.argmax(output_logits, axis=1)

                # Обновляем общую подпись и текущую частичную подпись.
                captions[:, t] = word.numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)

            return captions


class TransformerDecoderLayer(nn.Module):
    """
    Один слой декодера трансформера, предназначенный для использования в TransformerDecoder.
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        Конструктор класса TransformerDecoderLayer.

        Входы:
         - input_dim: Число входных признаков на входе.
         - num_heads: Число голов внимания.
         - dim_feedforward: Размер модели сети прямого распространения.
         - dropout: Значение для dropout.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.multihead_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()


    def forward(self, tgt, memory, tgt_mask=None):
        """
        Передает входные данные (и маску) через слой декодера.

        Входы:
        - tgt: последовательсность слоя декодера, форма (N, T, W)
        - memory: последовательсность на выходе последнего слоя энкодера,форма (N, S, D)
        - tgt_mask: части целевой последовательности для маскирования, форма (T, T)

        Возвращает:
        - out: признаки на выходе трансформера, форма (N, T, W)
        """
        # Определяем  самовнимание на целевой последовательности (вместе с dropout и
        #  слоем нормализации).
        tgt2 = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Внимание для целевой последовательности,  и для последовательности  последнего
        # слоя кодировщика.
        tgt2 = self.multihead_attn(query=tgt, key=memory, value=memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Передача
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

def clones(module, N):
    "Создание N идентичных слоёв"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask)

        return output
