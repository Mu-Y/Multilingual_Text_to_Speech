from dataset.dataset import TextToSpeechDataset
import pdb



if __name__ == '__main__':


    # run the `create_meta_file` on raw data of css10
    # to obtain meta file with both text and phonemes
    # TextToSpeechDataset.create_meta_file("css10", "./data/css10", "ge+fr+sp_w-ipa.txt", 22050, 80, spectrograms=False, phonemes=True)

    # create the helper dict from the phoneme meta
    wavPath2Phns = {}
    with open("./data/css10/ge+fr+sp_w-ipa.txt", "r", encoding='utf-8') as f:
        for line in f:
            items = line.split("|")
            wav_path = items[3]
            phns = items[7]
            wavPath2Phns[wav_path] = phns


    # match the selected utterances (provided with the TTS author) with their
    # corresponding phonemes
    with open("./data/css10/val_ge+fr+sp.txt", "r", encoding='utf-8') as fr:
        with open("./data/css10/val_ge+fr+sp_w-ipa.txt", "w", encoding='utf-8') as fw:
            for line in fr:
                items = line.split("|")
                wav_path = items[3]
                try:
                    phns = wavPath2Phns[wav_path]
                    if items[-1] in ["", "\n"]:
                        items[-1] = phns
                    else:
                        items.append(phns)
                except:
                    pdb.set_trace()

                fw.write("|".join(items))




