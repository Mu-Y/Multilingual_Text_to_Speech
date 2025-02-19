import os
import time
import datetime
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.dataset import TextToSpeechDatasetCollection, TextToSpeechCollate
from params.params import Params as hp
from utils import audio, text
from modules.tacotron2 import Tacotron, TacotronLoss
from utils.logging import Logger
from utils.samplers import RandomImbalancedSampler, PerfectBatchSampler
from utils import lengths_to_mask, to_gpu
import warnings
from ewc import EWC
import pdb

warnings.filterwarnings("ignore", category=DeprecationWarning)  # do not print Deprecation Warnings


def cos_decay(global_step, decay_steps):
    """Cosine decay function

    Arguments:
        global_step -- current training step
        decay_steps -- number of decay steps
    """
    global_step = min(global_step, decay_steps)
    return 0.5 * (1 + math.cos(math.pi * global_step / decay_steps))


def train(logging_start_epoch, epoch, data, model, criterion, optimizer, ewc=None):
    """Main training procedure.

    Arguments:
        logging_start_epoch -- number of the first epoch to be logged
        epoch -- current epoch
        data -- DataLoader which can provide batches for an epoch
        model -- model to be trained
        criterion -- instance of loss function to be optimized
        optimizer -- instance of optimizer which will be used for parameter updates
    """

    model.train()

    # initialize counters, etc.
    learning_rate = optimizer.param_groups[0]['lr']
    cla = 0
    done, start_time = 0, time.time()

    # loop through epoch batches
    for i, batch in enumerate(data):

        global_step = done + epoch * len(data)
        optimizer.zero_grad()

        # parse batch
        batch = list(map(to_gpu, batch))
        src, src_len, trg_mel, trg_lin, trg_len, stop_trg, spkrs, langs = batch

        # get teacher forcing ratio
        if hp.constant_teacher_forcing: tf = hp.teacher_forcing
        else: tf = cos_decay(max(global_step - hp.teacher_forcing_start_steps, 0), hp.teacher_forcing_steps)

        # run the model
        post_pred, pre_pred, stop_pred, alignment, spkrs_pred, enc_output = model(src, src_len, trg_mel, trg_len, spkrs, langs, tf)

        # evaluate loss function
        post_trg = trg_lin if hp.predict_linear else trg_mel
        classifier = model._reversal_classifier if hp.reversal_classifier else None
        loss, batch_losses = criterion(src_len, trg_len, pre_pred, trg_mel, post_pred, post_trg, stop_pred, stop_trg, alignment,
                                       spkrs, spkrs_pred, enc_output, classifier)

        # evaluate adversarial classifier accuracy, if present
        if hp.reversal_classifier:
            input_mask = lengths_to_mask(src_len)
            trg_spkrs = torch.zeros_like(input_mask, dtype=torch.int64)
            for s in range(hp.speaker_number):
                speaker_mask = (spkrs == s)
                trg_spkrs[speaker_mask] = s
            matches = (trg_spkrs == torch.argmax(torch.nn.functional.softmax(spkrs_pred, dim=-1), dim=-1))
            matches[~input_mask] = False
            cla = torch.sum(matches).item() / torch.sum(input_mask).item()

        # comptute gradients and make a step
        if ewc is not None:
            loss += hp.ewc_importance * ewc.penalty(model)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.gradient_clipping)
        optimizer.step()

        # log training progress
        if epoch >= logging_start_epoch:
            Logger.training(global_step, batch_losses, gradient, learning_rate, time.time() - start_time, cla)

        # update criterion states (params and decay of the loss and so on ...)
        criterion.update_states()

        start_time = time.time()
        done += 1


def evaluate(epoch, data, model, criterion, data_old_tasks=None):
    """Main evaluation procedure.

    Arguments:
        epoch -- current epoch
        data -- DataLoader which can provide validation batches
        model -- model to be evaluated
        criterion -- instance of loss function to measure performance
    """

    model.eval()

    # initialize counters, etc.
    mcd, mcd_count = 0, 0
    cla, cla_count = 0, 0
    eval_losses = {}

    # loop through epoch batches
    with torch.no_grad():
        for i, batch in enumerate(data):

            # parse batch
            batch = list(map(to_gpu, batch))
            src, src_len, trg_mel, trg_lin, trg_len, stop_trg, spkrs, langs = batch

            # run the model (twice, with and without teacher forcing)
            post_pred, pre_pred, stop_pred, alignment, spkrs_pred, enc_output = model(src, src_len, trg_mel, trg_len, spkrs, langs, 1.0)
            post_pred_0, _, stop_pred_0, alignment_0, _, _ = model(src, src_len, trg_mel, trg_len, spkrs, langs, 0.0)
            stop_pred_probs = torch.sigmoid(stop_pred_0)

            # evaluate loss function
            post_trg = trg_lin if hp.predict_linear else trg_mel
            classifier = model._reversal_classifier if hp.reversal_classifier else None
            loss, batch_losses = criterion(src_len, trg_len, pre_pred, trg_mel, post_pred, post_trg, stop_pred, stop_trg, alignment,
                                           spkrs, spkrs_pred, enc_output, classifier)

            # compute mel cepstral distorsion
            for j, (gen, ref, stop) in enumerate(zip(post_pred_0, trg_mel, stop_pred_probs)):
                stop_idxes = np.where(stop.cpu().numpy() > 0.5)[0]
                stop_idx = min(np.min(stop_idxes) + hp.stop_frames, gen.size()[1]) if len(stop_idxes) > 0 else gen.size()[1]
                gen = gen[:, :stop_idx].data.cpu().numpy()
                ref = ref[:, :trg_len[j]].data.cpu().numpy()
                if hp.normalize_spectrogram:
                    gen = audio.denormalize_spectrogram(gen, not hp.predict_linear)
                    ref = audio.denormalize_spectrogram(ref, True)
                if hp.predict_linear: gen = audio.linear_to_mel(gen)
                mcd = (mcd_count * mcd + audio.mel_cepstral_distorision(gen, ref, 'dtw')) / (mcd_count+1)
                mcd_count += 1

            # compute adversarial classifier accuracy
            if hp.reversal_classifier:
                input_mask = lengths_to_mask(src_len)
                trg_spkrs = torch.zeros_like(input_mask, dtype=torch.int64)
                for s in range(hp.speaker_number):
                    speaker_mask = (spkrs == s)
                    trg_spkrs[speaker_mask] = s
                matches = (trg_spkrs == torch.argmax(torch.nn.functional.softmax(spkrs_pred, dim=-1), dim=-1))
                matches[~input_mask] = False
                cla = (cla_count * cla + torch.sum(matches).item() / torch.sum(input_mask).item()) / (cla_count+1)
                cla_count += 1

            # add batch losses to epoch losses
            for k, v in batch_losses.items():
                eval_losses[k] = v + eval_losses[k] if k in eval_losses else v

    # normalize loss per batch
    for k in eval_losses.keys():
        eval_losses[k] /= len(data)


    if data_old_tasks is not None:
        mcd_old_tasks, mcd_count_old_tasks = 0., 0.
        # loop through epoch batches
        with torch.no_grad():
            for i, batch in enumerate(data_old_tasks):

                # parse batch
                batch = list(map(to_gpu, batch))
                src_old, src_len_old, trg_mel_old, trg_lin_old, trg_len_old, stop_trg_old, spkrs_old, langs_old = batch

                # run the model (without teacher forcing, computing mcd only)
                post_pred_0_old, _, stop_pred_0_old, _, _, _ = model(src_old, src_len_old, trg_mel_old, trg_len_old, spkrs_old, langs_old, 0.0)
                stop_pred_probs = torch.sigmoid(stop_pred_0_old)


                # compute mel cepstral distorsion
                for j, (gen, ref, stop) in enumerate(zip(post_pred_0_old, trg_mel_old, stop_pred_probs)):
                    stop_idxes = np.where(stop.cpu().numpy() > 0.5)[0]
                    stop_idx = min(np.min(stop_idxes) + hp.stop_frames, gen.size()[1]) if len(stop_idxes) > 0 else gen.size()[1]
                    gen = gen[:, :stop_idx].data.cpu().numpy()
                    ref = ref[:, :trg_len_old[j]].data.cpu().numpy()
                    if hp.normalize_spectrogram:
                        gen = audio.denormalize_spectrogram(gen, not hp.predict_linear)
                        ref = audio.denormalize_spectrogram(ref, True)
                    if hp.predict_linear: gen = audio.linear_to_mel(gen)
                    mcd_old_tasks = (mcd_count_old_tasks * mcd_old_tasks + audio.mel_cepstral_distorision(gen, ref, 'dtw')) / (mcd_count_old_tasks+1)
                    mcd_count_old_tasks += 1

    # log evaluation
    Logger.evaluation(epoch+1, eval_losses, mcd, mcd_old_tasks, src_len, trg_len, src, post_trg, post_pred, post_pred_0, stop_pred_probs, stop_trg, alignment_0, cla)

    return sum(eval_losses.values())


class DataParallelPassthrough(torch.nn.DataParallel):
    """Simple wrapper around DataParallel."""
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


if __name__ == '__main__':
    import argparse
    import os
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_directory", type=str, default=".", help="Base directory of the project.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Name of the initial checkpoint.")
    parser.add_argument("--checkpoint_root", type=str, default="checkpoints", help="Base directory of checkpoints.")
    parser.add_argument("--warm_start", action="store_true", help="if set, start training using the optimizer state in checkpoint; if not set, start training with newly optimizer state but the pretrained model weights")
    parser.add_argument("--data_root", type=str, default="data", help="Base directory of datasets.")
    parser.add_argument("--flush_seconds", type=int, default=60, help="How often to flush pending summaries to tensorboard.")
    parser.add_argument('--hyper_parameters', type=str, default=None, help="Name of the hyperparameters file.")
    parser.add_argument('--logging_start', type=int, default=1, help="First epoch to be logged")
    parser.add_argument('--max_gpus', type=int, default=2, help="Maximal number of GPUs of the local machine to use.")
    parser.add_argument('--loader_workers', type=int, default=16, help="Number of subprocesses to use for data loading.")
    args = parser.parse_args()

    # set up seeds and the target torch device
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # prepare directory for checkpoints
    checkpoint_dir = os.path.join(args.base_directory, args.checkpoint_root)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # load checkpoint (dict) with saved hyper-parameters (let some of them be overwritten because of fine-tuning)
    if args.checkpoint:
        checkpoint = os.path.join(args.checkpoint)
        checkpoint_state = torch.load(checkpoint, map_location='cpu')
        hp.load_state_dict(checkpoint_state['parameters'])

    # load hyperparameters
    if args.hyper_parameters is not None:
        hp_path = os.path.join('params', f'{args.hyper_parameters}.json')
        hp.load(hp_path)


    # ensure later we don't use dataset sampler b/c there is only one language
    assert hp.balanced_sampling is False
    assert hp.perfect_sampling is False, print("in continue training after initial training, sampler needs to be disabled b/c there is only one language")
    assert args.checkpoint is not None, print("incontinue training after initial training, a checkpoint is required")


    ### old tasks dataloader, TODO
    dataset_old = TextToSpeechDatasetCollection("./data/css10",
                                            "train_initial5.txt", "val_initial5.txt")
    dp_devices = args.max_gpus if hp.parallelization and torch.cuda.device_count() > 1 else 1
    # train_sampler_old = PerfectBatchSampler(dataset_old.train, hp.languages[:5], hp.batch_size, data_parallel_devices=dp_devices, shuffle=True, drop_last=True)
    # train_data_old = DataLoader(dataset_old.train, batch_sampler=train_sampler_old, collate_fn=TextToSpeechCollate(hp.sort_by_text_len), num_workers=args.loader_workers) # was false
    # eval_sampler_old = PerfectBatchSampler(dataset_old.dev, hp.languages[:5], hp.batch_size, data_parallel_devices=dp_devices, shuffle=False)
    # eval_data_old = DataLoader(dataset_old.dev, batch_sampler=eval_sampler_old, collate_fn=TextToSpeechCollate(hp.sort_by_text_len), num_workers=args.loader_workers) # was false
    eval_data_old = DataLoader(dataset_old.dev, batch_size=hp.batch_size, drop_last=False, shuffle=False, collate_fn=TextToSpeechCollate(True), num_workers=args.loader_workers)
    #############



    # initialize logger
    log_dir = os.path.join(args.base_directory, "logs", f'{hp.version}-{datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")}')
    Logger.initialize(log_dir, args.flush_seconds)

    initial_epoch=0
    for train_lang in hp.continue_languages:

        # check directory
        if not os.path.exists(os.path.join(checkpoint_dir, train_lang)):
            os.makedirs(os.path.join(checkpoint_dir, train_lang))
        if not os.path.exists(os.path.join(args.base_directory, "logs")):
            os.makedirs(os.path.join(args.base_directory, "logs"))


        # load dataset
        dataset = TextToSpeechDatasetCollection(os.path.join(args.data_root, hp.dataset),
                                                "train_{}.txt".format(train_lang),
                                                "val_{}.txt".format(train_lang))




        sampler = None  # for single-lang training, do not use any sampler
        train_data = DataLoader(dataset.train, batch_size=hp.batch_size, drop_last=True, shuffle=(not hp.multi_language or not hp.balanced_sampling), sampler=sampler, collate_fn=TextToSpeechCollate(True), num_workers=args.loader_workers)
        eval_data = DataLoader(dataset.dev, batch_size=hp.batch_size, drop_last=False, shuffle=False,
                               collate_fn=TextToSpeechCollate(True), num_workers=args.loader_workers)


        # find out number of unique speakers and languages
        hp.speaker_number = 0 if not hp.multi_speaker else dataset.train.get_num_speakers()
        hp.language_number = 0 if not hp.multi_language else len(hp.languages)
        # save all found speakers to hyper parameters
        if hp.multi_speaker and not args.checkpoint:
            hp.unique_speakers = dataset.train.unique_speakers

        # acquire dataset-dependent constants, these should probably be the same while going from checkpoint
        if not args.checkpoint:
            # TODO: why we need this?
            # compute per-channel constants for spectrogram normalization
            hp.mel_normalize_mean, hp.mel_normalize_variance = dataset.train.get_normalization_constants(True)
            if hp.predict_linear:
                hp.lin_normalize_mean, hp.lin_normalize_variance = dataset.train.get_normalization_constants(False)

        # instantiate model
        if torch.cuda.is_available():
            model = Tacotron().cuda()
            if hp.parallelization and args.max_gpus > 1 and torch.cuda.device_count() > 1:
                model = DataParallelPassthrough(model, device_ids=list(range(args.max_gpus)))
        else: model = Tacotron()

        # instantiate optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
        if hp.encoder_optimizer:
            encoder_params = list(model._encoder.parameters())
            other_params = list(model._decoder.parameters()) + list(model._postnet.parameters()) + list(model._prenet.parameters()) + \
                           list(model._embedding.parameters()) + list(model._attention.parameters())
            if hp.reversal_classifier:
                other_params += list(model._reversal_classifier.parameters())
            optimizer = torch.optim.Adam([
                {'params': other_params},
                {'params': encoder_params, 'lr': hp.learning_rate_encoder}
            ], lr=hp.learning_rate, weight_decay=hp.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, hp.learning_rate_decay_each // len(train_data), gamma=hp.learning_rate_decay)
        criterion = TacotronLoss(hp.guided_attention_steps, hp.guided_attention_toleration, hp.guided_attention_gain)

        # load model weights, fisher, and optimizer, scheduler states from checkpoint state dictionary
        # initial_epoch = 0
        if args.checkpoint:
            checkpoint_state = torch.load(args.checkpoint, map_location='cpu')
            # hp.load_state_dict(checkpoint_state['parameters'])
            # load model state dict (can be imcomplete if pretraining part of the model)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint_state['model'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            criterion.load_state_dict(checkpoint_state['criterion'])
            if args.warm_start:
                # other states from checkpoint -- optimizer, scheduler, loss, epoch
                initial_epoch = checkpoint_state['epoch'] + 1
                optimizer.load_state_dict(checkpoint_state['optimizer'])
                scheduler.load_state_dict(checkpoint_state['scheduler'])
            print("model loaded from {}".format(args.checkpoint))
            if hp.use_ewc:
                ewc = EWC(model, criterion)
                ewc.load_fisher(checkpoint_state['fisher'])
            else:
                ewc = None


        # training loop
        best_eval = float('inf')
        for epoch in range(initial_epoch, initial_epoch + hp.epochs):
            train(args.logging_start, epoch, train_data, model, criterion, optimizer, ewc)
            if hp.learning_rate_decay_start - hp.learning_rate_decay_each < epoch * len(train_data):
                scheduler.step()
            eval_loss = evaluate(epoch, eval_data, model, criterion, eval_data_old)
            print("Epoch: {}, Eval_loss: {}".format(epoch, eval_loss))
            if (epoch + 1) % hp.checkpoint_each_epochs == 0:
                # save checkpoint together with hyper-parameters, optimizer and scheduler states
                checkpoint_file = f'{checkpoint_dir}/{train_lang}/{hp.version}_loss-{epoch}-{eval_loss:2.3f}'
                state_dict = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'parameters': hp.state_dict(),
                    'criterion': criterion.state_dict()
                }
                torch.save(state_dict, checkpoint_file)
                print("Saved model to {}".format(checkpoint_file))

        # after training on the current task, update the ewc fisher
        # ewc = EWC(model, criterion, train_data, hp.ewc_sample_size)
        if hp.use_ewc:
            ewc.update_fisher(train_data, hp.ewc_sample_size)
            state_dict = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'parameters': hp.state_dict(),
                'criterion': criterion.state_dict(),
                'fisher': ewc.get_fisher()
            }
            torch.save(state_dict, "{}-fisher".format(checkpoint_file))

        # update the checkpoint to be used in the next task
        if hp.use_ewc:
            args.checkpoint = "{}-fisher".format(checkpoint_file)
        else:
            args.checkpoint = checkpoint_file


        # make eval_data_old to be eval data for all tasks seen so far
        eval_data_old = list(eval_data_old) + list(eval_data)
        # re-initialize epoch num
        initial_epoch += hp.epochs
