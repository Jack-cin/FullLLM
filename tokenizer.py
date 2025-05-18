import os
import struct
import argparse
import torch
from typing import List, Dict

from sentencepiece import SentencePieceProcessor

TOKENIZER_MODEL = "tokenizer.model"  # Llama sentencepiece tokenizer 分词器模型路径

class Tokenizer:
    def __init__(self, tokenizer_model=None):
        # 加载 SentencePiece 模型文件，若没有提供路径则使用默认的 TOKENIZER_MODEL
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        assert os.path.isfile(model_path), model_path  # 确保文件存在
        self.sp_model = SentencePieceProcessor(model_file=model_path)  # 加载 SentencePiece 模型
        self.model_path = model_path  # 模型路径保存

        # 获取 BOS / EOS / PAD token 的 ID
        self.n_words: int = self.sp_model.vocab_size()  # 获取词汇表大小 N，表示模型可以处理的单词数
        self.bos_id: int = self.sp_model.bos_id()  # 获取 BOS (开始标记) token 的 ID
        self.eos_id: int = self.sp_model.eos_id()  # 获取 EOS (结束标记) token 的 ID
        self.pad_id: int = self.sp_model.pad_id()  # 获取 PAD (填充标记) token 的 ID
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()  # 校验词汇大小与模型词汇一致

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        将输入字符串 `s` 编码为一系列 token IDs，并可根据需要添加 BOS 和 EOS token。

        数学描述：
        1. 设 `s` 为输入字符串，模型输出的是一组 token IDs，记为 `{t1, t2, ..., tn}`。
        2. 若 `bos=True`，则在序列的开始处添加一个特殊的开始标记 BOS，记为 `BOS_ID`。
        3. 若 `eos=True`，则在序列的结束处添加一个特殊的结束标记 EOS，记为 `EOS_ID`。

        编码过程可以表示为：
        tokens = Encode(s) → [BOS_ID] + {t1, t2, ..., tn} + [EOS_ID]  # 若需要添加 BOS/EOS
        """
        assert type(s) is str  # 确保输入是字符串类型
        t = self.sp_model.encode(s)  # 使用 SentencePiece 将字符串编码为 token IDs
        if bos:
            t = [self.bos_id] + t  # 在列表前添加 BOS token ID
        if eos:
            t = t + [self.eos_id]  # 在列表末尾添加 EOS token ID
        return t

    def decode(self, t: List[int]) -> str:
        """
        将输入的 token ID 序列解码为字符串。

        数学描述：
        对于一个由 token IDs 组成的序列 `{t1, t2, ..., tn}`，解码过程通过查找每个 ID 对应的 token，
        将其转换回原始字符串 `s`：
        Decode({t1, t2, ..., tn}) → s
        """
        return self.sp_model.decode(t)  # 使用 SentencePiece 将 token ID 序列解码为原始字符串

    def pad(self, input_data: List[Dict]) -> Dict:
        """
        对输入数据进行填充，使其长度一致，适用于批量数据处理。

        数学描述：
        设 `X = {x1, x2, ..., xB}` 是 B 个样本，每个样本包含输入 ID 和注意力掩码 `{input_ids, attention_mask}`。
        填充操作通过计算最大长度 `L_max` 来确保每个输入的长度一致：
        Pad(X) → {input_ids: padded_ids, attention_mask: padded_mask}

        填充操作的目标是将所有输入序列 `x_i` 扩展到最大长度 `L_max`，并确保在填充位置的注意力掩码为 0。
        """
        max_len = max([len(x["input_ids"]) for x in input_data])  # 计算最大序列长度 L_max
        result = {"input_ids": torch.zeros(len(input_data), max_len, dtype=torch.long),
                  "attention_mask": torch.zeros(len(input_data), max_len, dtype=torch.long)}  # 初始化填充后的 tensor

        is_tensor_ele = isinstance(input_data[0]["input_ids"], torch.Tensor)  # 检查输入是否为 Tensor
        for i, data in enumerate(input_data):
            if not is_tensor_ele:
                result["input_ids"][i, :len(data["input_ids"])] = torch.tensor(data["input_ids"])  # 填充 input_ids
                result["attention_mask"][i, :len(data["attention_mask"])] = torch.tensor(data["attention_mask"])  # 填充 attention_mask
            else:
                result["input_ids"][i, :len(data["input_ids"])] = data["input_ids"]
                result["attention_mask"][i, :len(data["attention_mask"])] = data["attention_mask"]
        return result

    def export(self):
        """
        导出模型的所有 token 和对应的分数（后处理后）为二进制文件。

        数学描述：
        设 `tokens = {t1, t2, ..., tn}` 是模型的所有 token，`scores = {s1, s2, ..., sn}` 是每个 token 对应的得分。
        每个 token 在 SentencePiece 模型中都有一个对应的 ID 和得分，模型将每个 token 转换为字节流，并将其与得分一起保存为二进制格式。

        具体操作如下：
        1. 对每个 token `t_i`，获取其字节表示和分数 `s_i`。
        2. 将 token 的字节表示和分数打包成一个二进制格式。
        3. 写入二进制文件，其中每个 token 的格式为 `score (float) + length (int) + token (bytes)`。

        最终生成的文件可以用于后续的模型加载。
        """
        tokens, scores = [], []  # 存储所有 token 的字节流和得分
        for i in range(self.n_words):  # 遍历所有词汇表中的 token
            t = self.sp_model.id_to_piece(i)  # 获取当前 token 的字符表示
            s = self.sp_model.get_score(i)  # 获取当前 token 的得分
            if i == self.bos_id:
                t = '\n<s>\n'  # 对 BOS token 做特殊处理
            elif i == self.eos_id:
                t = '\n</s>\n'  # 对 EOS token 做特殊处理
            t = t.replace('▁', ' ')  # SentencePiece 使用特殊字符作为空格，进行替换
            b = t.encode('utf-8')  # 将 token 转换为 utf-8 编码的字节流

            tokens.append(b)  # 存储字节流
            scores.append(s)  # 存储得分

        max_token_length = max(len(t) for t in tokens)  # 计算最大 token 长度

        # 生成二进制文件路径，替换文件扩展名为 .bin
        tokenizer_bin = self.model_path.replace('.model', '.bin')
        with open(tokenizer_bin, 'wb') as f:
            f.write(struct.pack("I", max_token_length))  # 写入最大 token 长度
            for bytes, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes)))  # 写入每个 token 的得分和字节长度
                f.write(bytes)  # 写入 token 的字节流

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer-model", type=str, help="optional path to custom tokenizer ")
    args = parser.parse_args()

    t = Tokenizer(args.tokenizer_model)  # 创建 Tokenizer 实例
    t.export()  # 导出模型的二进制文件
