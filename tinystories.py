'''
第一章：数据下载与预处理
故事从数据下载开始。我们需要从指定的URL下载TinyStories数据集并解压到本地目录。
数学语言：设数据集的URL为 U，下载路径为 D，通过指定块大小 chunk_size 逐块下载数据，避免内存溢出。

下载文件并写入本地
使用requests库流式下载，避免一次性加载整个文件，逐块处理每个数据块
通过tqdm显示下载进度

下载完成后，解压文件到指定目录，生成多个分片文件（shard），每个文件包含多个TinyStories故事。

第二章：词表训练
接下来，我们训练一个自定义的分词器。使用SentencePiece方法训练一个BPE（字节对编码）分词器。
数学语言：通过训练，构建一个映射函数将故事文本转化为整数token。

训练过程中，将TinyStories文本合并成一个大文件（tiny.txt），并进行token化处理
最终生成词表，大小为用户指定的vocab_size，并保存训练好的tokenizer。

第三章：数据预处理与分词
在这部分，我们将数据分片中的每个故事通过训练好的分词器进行分词处理，生成token。
数学语言：每个分片中的文本通过分词器转化为tokens序列，存储为二进制文件（tok512.bin）。

每个数据分片将被处理并转化为token序列，生成对应的二进制文件。
存储格式为：tokens = [token_1, token_2, ..., token_m]，然后保存为文件。

第四章：数据集划分与训练
数据集分为训练集和测试集。训练集包含多个数据分片，测试集只包含一个数据分片。
每次训练时，我们从训练集随机选择数据进行处理并加载。数据会被打乱以增强训练多样性。

数学语言：设训练集为 D_train，测试集为 D_test，从 D_train 中抽取批次数据：
每个批次的数据是 tokens 序列的子集（token_i, token_{i+1}, ..., token_{i+k}），作为输入和输出进行训练。

训练过程中，通过DataLoader加载数据并通过模型进行前向和反向传播，优化目标是最小化损失函数 L_loss。

第五章：训练模型
数据集准备好后，开始训练模型。通过PyTorch的DataLoader实现批量数据加载和GPU计算。
数学语言：每个批次的数据被加载到GPU进行计算，通过反向传播更新模型参数 θ，目标是最小化损失函数：
L_loss = (1/N) * sum(loss(y_i, y_hat_i))，其中 y_i 是目标，y_hat_i 是模型预测。

故事结尾
最终，通过该过程构建了一个适合训练语言模型的数据集，包含了从下载数据、预处理、分词到数据划分和加载的完整步骤。
这为后续的深度学习模型训练提供了高效的数据源，为应用场景中的自然语言处理任务奠定了基础。
'''



"""
Download, preprocess and serve the TinyStories dataset as a DataLoader.
"""

import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"

def download_file(url: str, fname: str, chunk_size=1024):
    # url: 要下载文件的网址
    # fname: 文件保存到本地时的名字。
    # chunk_size: 每次从网络读取的数据块大小，默认是1024字节（即1KB）。这有助于处理大文件，避免一次性加载整个文件到内存中导致内存溢出


    """Helper function to download a file from a given url"""
    '''
    使用Python的requests库发送GET请求到指定的URL。
    这里的stream=True参数告诉requests不要立即下载整个文件，而是以流的形式下载，这样可以逐步处理数据，适合处理大文件
    '''
    resp = requests.get(url, stream=True)

    '''
    从响应头(resp.headers)获取content-length字段，它表示服务器返回的内容长度（以字节为单位）。
    如果该值不存在，则默认为0。这里将这个值转换成整数类型，用于后续进度条显示总进度
    '''
    total = int(resp.headers.get("content-length", 0))

    '''
    with open(fname, "wb") as file: 打开或创建一个文件用于写入二进制数据("wb"模式)。使用with语句确保文件在操作完成后正确关闭。
        tqdm() 是一个快速、可扩展的Python进度条库，用来显示下载进度。
        desc=fname: 在进度条前面显示的文字描述，这里是文件名。
        total=total: 进度条的最大值，也就是文件的总大小。
        unit="iB": 显示单位为字节(iB)。
        unit_scale=True: 自动调整单位(比如K, M, G等)。
        unit_divisor=1024: 定义了单位转换的基础，因为1KB=1024B。
    '''
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:

        '''
        resp.iter_content(chunk_size=chunk_size): 从HTTP响应中以指定的chunk_size(这里是1024字节)分块迭代内容。
        size = file.write(data): 将每一块数据写入到打开的文件中，并记录写入的字节数。
        bar.update(size): 更新tqdm进度条，增加已经下载的字节数量。
        '''
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

# ##################### 模块一：数据下载（download函数）#####################
def download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    # 创建缓存目录
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # 数据集URL
    # download the TinyStories dataset, unless it's already downloaded
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz") # 本地存储路径

    # 下载检查：避免重复下载
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # 解压检查：自动解压tar.gz文件
    # unpack the tar.gz file into all the data shards (json files)
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    # 数据验证：打印示例故事
    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    with open(shard_filenames[0], "r") as f:
        data = json.load(f)
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    print(f"Example story:\n{json.dumps(data[0:30], indent=2)}")

# ############ 模块二：词表(分词器)训练（train_vocab函数）：tok512.model###############
def train_vocab(vocab_size):
    """
    Trains a custom sentencepiece tokenizer on the TinyStories dataset.
    The custom tokenizer files will be saved in DATA_CACHE_DIR/tok{N} directories,
    where N is the vocab size. This is also where the pretok .bin files will go.
    """
    # 词表大小：决定模型能识别的独立词汇数量
    assert vocab_size > 0, "Vocab size must be positive"

    # 创建训练语料库临时文件
    # how many shards we'll use for vocab training, kept low for efficiency
    num_shards = 10
    # 1) export a large chunk of text as a single text file tiny.txt
    tiny_file = os.path.join(DATA_CACHE_DIR, "tiny.txt")
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    print(f"Writing temporary file {tiny_file} with {num_shards} shards...")
    with open(tiny_file, "w", encoding="utf-8") as of:
        for shard in tqdm(shard_filenames[:num_shards]): #取前10个分片json文件作为样本
            with open(shard, "r") as f:
                '''data结构
                [{},{},{"story": "Once upon a time, ...", 
                "instruction": {"prompt:": "Write a short story (3-5 paragraphs) ...", "words": ["hide", "time", "upset"], "features": []}, 
                "summary": "Lily played hide ....", 
                "source": "GPT-4"}]
                '''
                data = json.load(f)

            for example in data:
                text = example["story"]
                text = text.strip()
                of.write(text + "\n")
    print(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")

    # 【核心】训练BPE分词器：通过合并高频字符对构建词表
    # output file prefix path for sentencepiece
    prefix = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
    # 2) train the sentencepiece model
    print("Will now train the vocab...")
    spm.SentencePieceTrainer.train(input=tiny_file,
                                   model_prefix=prefix,
                                   model_type="bpe", # 使用字节对编码算法
                                   vocab_size=vocab_size, #用户指定词表大小
                                   self_test_sample_size=0,
                                   input_format="text",
                                   character_coverage=1.0,
                                   num_threads=os.cpu_count(),
                                   split_digits=True, # 将数字拆分为单独token
                                   allow_whitespace_only_pieces=True,
                                   byte_fallback=True, # 未知字符回退到字节表示
                                   unk_surface=r" \342\201\207 ",
                                   normalization_rule_name="identity")

    # 是否删除临时文件
    # 3) optional cleanup, ask the user if they'd like to delete tiny.txt
    dec = input(f"Delete the temporary file {tiny_file}? [y/N] ")
    if dec.lower() == "y":
        os.remove(tiny_file)
        print(f"Deleted {tiny_file}")

    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")


def process_shard(args, vocab_size):
    # 加载分词器
    shard_id, shard = args # executor.map(fun, enumerate(shard_filenames))
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)

    # 处理单个数据分片json文件
    with open(shard, "r") as f: #shard_filenames
        data = json.load(f)
    all_tokens = []
    for example in tqdm(data, position=shard_id):
        text = example["story"]
        text = text.strip()  # 清理空白字符
        tokens = enc.encode(text, bos=True, eos=False)  # 添加起始符<BOS>
        all_tokens.extend(tokens) # 拼接所有token

    # 存储为二进制格式
    # convert to uint16 nparray:每个token用2字节表示，节省空间
    all_tokens = np.array(all_tokens, dtype=np.uint16)

    # calculate the output filename
    if vocab_size == 0:
        # if we're using Llama 2, just save the tokenized file in the same dir
        tokenized_filename = shard.replace(".json", ".bin")
    else:
        # save .bin files into a new tok{N} directory
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)

    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes()) # 二进制写入

    # calculate the average sequence length 平均句长(they are separated by BOS=1)
    '''
    BOS标记：Begin of Sentence，标识句子开始
    '''
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum()) #all_tokens中值为1的元素数量
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")

# ### 模块三：数据预处理（pretokenize函数）：原始数据经过分词器后的tok512.bin文件########
def pretokenize(vocab_size):
    # 创建文件夹：保存从原始数据经过分词器后的bin文件
    if vocab_size > 0:
        # .bin files will be saved into tok{N} directory, create it once here
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)

    # 多进程预处理加速：并行处理分块json文件
    # iterate the shards and tokenize all of them one by one
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    # process all the shards in a process pool
    fun = partial(process_shard, vocab_size=vocab_size)
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))
    print("Done.")

# ################模块四：数据集划分（PretokDataset类）###################
#[磁盘上的.bin文件] → [内存映射读取] → [切块+打乱] → [构造训练样本] → [持续无限供应数据]

class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized samples from disk and yields them as PyTorch tensors."""
    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split  #选择划分模式：train/test datasets
        self.max_seq_len = max_seq_len #决定划分的单句最大长度
        self.vocab_size = vocab_size #
        self.vocab_source = vocab_source #选择不同分词器对应的分词结果

    def __iter__(self):
        # 0.确保各进程数据不同：获取当前进程ID->rng.shuffle()
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)

        print(f"Created a PretokDataset with rng seed {seed}")
        if self.vocab_source == "llama2":
            # the .bin files are right along the .json files
            bin_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        elif self.vocab_source == "custom":
            # the .bin files are in tok{N} directory
            bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{self.vocab_size}")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))

        # train/test split. let's use only shard 0 for test split, rest train
        # 一个分块json文件做测试，其他分块json文件做测试
        # 1.无限循环设计
        '''
        为什么需要无限循环？
        深度学习的训练往往需要多轮迭代（epoch），这个设计让数据可以无限重复供应，就像自助餐厅的旋转寿司传送带，吃完一轮后自动再来一轮新的。
        '''
        shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        assert len(shard_filenames)>0, f"No bin files found in {bin_dir}"
        while True:
            # 每次循环打乱文件顺序
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # 2.分片文件处理:内存映射方式读取，为了内存优化
                '''
                为什么需要内存映射技巧?
                假设你有一个100GB的大文件，普通读取会撑爆内存。内存映射就像给文件贴了个"便利贴"，只在需要时读取特定部分，如图书馆按需取书而不是搬走整个书架。
                '''
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r") #m:对应一个分块json文件 在分词化之后的 bin文件

                # 3.批次计算
                '''
                示例说明：
                假设文件包含1000个token，max_seq_len=256
                → 1000//256=3 完整批次（256*3=768）
                → 剩余232个token不足一批，直接丢弃
                就像把一根10米长的绳子，每2米剪一段，最后余下不足2米的部分不要
                '''
                num_batches = len(m) // self.max_seq_len # 计算总批次数
                num_batches -= 1  # 丢弃最后不完整的批次
                assert num_batches > 0, "this shard is way too small? investigate."

                # 4.索引打乱
                '''
                打乱的意义：
                就像洗牌后再发牌，防止模型记住数据顺序。假设原始数据是故事A→B→C，打乱后可能变成B→A→C，让模型学习更普适的语言规律。
                '''
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    # 5.数据切片m[start:end],数据转换:转为PyTorch张量
                    '''
                    +1的奥秘：
                    假设max_seq_len=3，需要截取4个token：
                    tokens: [A, B, C, D]
                    x: [A, B, C] → 模型输入
                    y: [B, C, D] → 预期输出
                    只有这样，每个y的(尤其是C)位置都是对应x位置的下一个token，
                    这就是语言模型的预测方式。
                    '''
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1 #【注意+1】
                    # calling .astype will copy the data into a new numpy array
                    #, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64)) # 截取数据块
                    x = chunk[:-1] # 输入序列
                    y = chunk[1:]  # 目标序列
                    '''
                    yield像流水线的传送带，逐个产出样本而不中断流程；
                    return像一次性交货，适合处理有限数据
                    '''
                    yield x, y

# -----------------------------------------------------------------------------
# public interface functions
def get_tokenizer_model_path(vocab_size):
    """
    Returns path to the sentencepiece tokenizer model for a given vocab size
    vocab_size = 0 designates the default Llama 2 tokenizer, in that case
    None is returned.
    """
    if vocab_size == 0:
        return None
    else:
        return os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}.model")

class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        # GPU数据传输优化
        '''
        pin_memory=True  # 启用锁页内存
        non_blocking=True # 异步传输
        '''

        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl: #x:(batch_size,max_seq_len) #y:(batch_size,max_seq_len)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y

# -----------------------------------------------------------------------------
# CLI for constructing the dataset
if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    python tinystories.py download
    python tinystories.py pretokenize

    To tokenize data with a custom tokenizer we train ourselves with sentencepiece, e.g.:
    python tinystories.py download
    python tinystories.py train_vocab --vocab_size=512
    python tinystories.py pretokenize --vocab_size=512
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "train_vocab", "pretokenize"])
    parser.add_argument("--vocab_size", type=int, default=0, help="pretokenization vocab size. 赋值0 = use Llama 2 tokenizer.")
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "download":
        download()
    elif args.stage == "train_vocab":
        train_vocab(vocab_size=args.vocab_size)
    elif args.stage == "pretokenize":
        pretokenize(vocab_size=args.vocab_size)
    else:
        raise ValueError(f"Unknown stage {args.stage}")