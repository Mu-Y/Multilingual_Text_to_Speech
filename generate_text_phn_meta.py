from dataset.dataset import TextToSpeechDataset



if __name__ == '__main__':
    TextToSpeechDataset.create_meta_file("css10", "./data/css10", "ge+fr+sp_w-ipa.txt", 22050, 80, spectrograms=False, phonemes=True)
