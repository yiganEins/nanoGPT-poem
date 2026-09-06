import sentencepiece as spm

def train():
    spm.SentencePieceTrainer.Train(
        input = ['./data/poems.ch'],
        model_prefix = './data/input_poems',
        vocab_size = 12000,
        model_type = 'bpe',
        character_coverage = 0.9995,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )

def test_text():
    with open('./data/poems.ch','r',encoding='utf-8') as f:
        text = f.read()
    
    print(len(text))
    vocab = sorted(list(set(text)))
    print(len(vocab))
    print(vocab[:100])

def text_model():
    sp = spm.SentencePieceProcessor()
    sp.load('./data/input_poems.model')

    text2 = '花有重开日，人无再少年。'
    print(sp.EncodeAsPieces(text2))
    print(sp.eos_id())


if __name__ == "__main__":
    # train()
    # test_text()
    text_model()