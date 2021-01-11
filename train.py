import torch
from torch.utils.data import DataLoader
from torchtext import data, datasets
from linguistic_style_transfer_pytorch.config import GeneralConfig, ModelConfig
from linguistic_style_transfer_pytorch.data_loader import TextDataset
from linguistic_style_transfer_pytorch.model import AdversarialVAE
from tqdm import tqdm, trange
import os
import numpy as np
import pickle
import spacy
from sklearn.metrics import f1_score
from ray import tune
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('runs/disentangled_privacy')

spacy.load("en_core_web_sm")

use_cuda = True if torch.cuda.is_available() else False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    mconfig = ModelConfig()
    gconfig = GeneralConfig()
    weights = torch.FloatTensor(np.load(gconfig.word_embedding_path))

   #Dataloader Privacy
    TEXT = data.Field('spacy', include_lengths=True)
    RATING_LABEL = data.LabelField()
    GENDER_LABEL = data.LabelField()
    AGE_LABEL = data.LabelField()
    LOCATION_LABEL = data.LabelField()

    train_data, valid_data, test_data = data.TabularDataset.splits(path=gconfig.privacy_text_path,
                                                                   train="train.csv",
                                                                   validation="valid.csv",
                                                                   test="test.csv",
                                                                   fields=[('text', TEXT), ('rating', RATING_LABEL),
                                                                           ('gender', GENDER_LABEL),
                                                                           ('age', AGE_LABEL),
                                                                           ('location', LOCATION_LABEL)],
                                                                   format="csv")

    print("Number of train_data = {}".format(len(train_data)))
    print("Number of valid_data = {}".format(len(valid_data)))
    print("Number of test_data = {}".format(len(test_data)))
    print("vars(train_data[0]) = {}\n".format(vars(train_data[0])))

    # 3. data.BucketIterator
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                   batch_size=mconfig.batch_size,
                                                                   device=device,
                                                                   sort_key=lambda x: len(x.text))

    TEXT.build_vocab(train_data, vectors="glove.6B.100d", max_size = 9201)
    RATING_LABEL.build_vocab(train_data)
    GENDER_LABEL.build_vocab(train_data)
    AGE_LABEL.build_vocab(train_data)
    LOCATION_LABEL.build_vocab(train_data)

    INPUT_DIM = len(TEXT.vocab)

    model = AdversarialVAE(INPUT_DIM, mconfig.embedding_size, weights)



    if use_cuda:
        model = model.cuda()

    private_discriminator_params, style_disc_params, vae_and_classifier_params = model.get_params()




    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print("The model has {} trainable parameters".format(count_parameters(model)))

    #============== Define optimizers ================#
    # content discriminator/adversary optimizer
  #  content_disc_opt = torch.optim.RMSprop(
   #     content_discriminator_params, lr=mconfig.content_adversary_lr)
    # style discriminaot/adversary optimizer


    private_disc_opt = torch.optim.RMSprop(
        private_discriminator_params, lr=mconfig.style_adversary_lr)

    style_disc_opt = torch.optim.RMSprop(
        style_disc_params, lr=mconfig.style_adversary_lr)
    # autoencoder and classifiers optimizer
    vae_and_cls_opt = torch.optim.Adam(
        vae_and_classifier_params, lr=mconfig.autoencoder_lr)
    print("Training started!")

    total_step = len(train_iter)
    for epoch in trange(mconfig.epochs, desc="Epoch"):

        model.train()

        total_loss = []
        train_total_correct = 0
        train_age_correct_disc = 0
        train_rating_correct_disc = 0
        train_age_correct_classifier = 0
        train_rating_correct_classifier = 0
        train_gender_correct = 0


        for iteration, batch in enumerate(train_iter):

            # unpacking
           # sequences, seq_lens, labels, bow_rep = batch
            #print(iteration)
            text, seq_lens = batch.text
            text.resize_(text.shape[1], text.shape[0])
            y = torch.cuda.LongTensor(batch.rating)
            y_gender = batch.gender
            y_age = torch.cuda.LongTensor(batch.age)
            age_onehot = torch.cuda.FloatTensor(y_age.shape[0], 2)
            # In your for loop
            age_onehot.zero_()
            age_onehot[torch.arange(y_age.shape[0]), y_age] = 1

            rating_onehot = torch.cuda.FloatTensor(y.shape[0], 5)
            # In your for loop
            rating_onehot.zero_()
            rating_onehot[torch.arange(y.shape[0]), y] = 1


            y_location = batch.location
            age_pred_disc, rating_pred_disc, age_pred_classifier, rating_pred_classifier, private_disc_loss, \
            style_disc_loss, private_mul_loss, style_mul_loss, vae_and_cls_loss = model(
                text, seq_lens, y_age, y,  iteration+1, epoch == mconfig.epochs-1)

            train_age_correct_disc += (age_pred_disc == y_age).sum().item()
            train_rating_correct_disc += (rating_pred_disc == y).sum().item()

            train_age_correct_classifier += (age_pred_classifier == y_age).sum().item()
            train_rating_correct_classifier += (rating_pred_classifier == y).sum().item()

            #tune.report(vae_and_cls_loss = vae_and_cls_loss)

            #============== Update Adversary/Discriminator parameters ===========#
            # update content discriminator parameters
            # we need to retain the computation graph so that discriminator predictions are
            # not freed as we need them to calculate entropy.
            # Note that even even detaching the discriminator branch won't help us since it
            # will be freed and delete all the intermediary values(predictions, in our case).
            # Hence, with no access to this branch we can't backprop the entropy loss

            #content_disc_loss.backward(retain_graph=True)
            #content_disc_opt.step()
            #content_disc_opt.zero_grad()

            # update style discriminator parameters

            private_disc_loss.backward(retain_graph=True)
            private_disc_opt.step()
            private_disc_opt.zero_grad()

            style_disc_loss.backward(retain_graph=True)
            style_disc_opt.step()
            style_disc_opt.zero_grad()

            #=============== Update VAE and classifier parameters ===============#
            vae_and_cls_loss.backward()
            vae_and_cls_opt.step()
            vae_and_cls_opt.zero_grad()

          #  writer.add_graph(model, text)
          #  writer.close()







            total_loss.append(vae_and_cls_loss.item())

            if (iteration + 1) % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Vae Loss: {:.4f}, Priv disc Loss: {:.4f}, Style disc Loss: {:.4f}, Age classifier loss: {:.4f}, \
                      Rating classifier: {:.4f}'\
                      .format(epoch + 1, mconfig.epochs, iteration + 1, total_step, vae_and_cls_loss.item(), private_disc_loss.item(), style_disc_loss.item(), \
                              private_mul_loss.item(), style_mul_loss.item() ))

        print("Training total_loss = {}".format(total_loss))
       # print("Training total_rating_accuracy = {:.4f}%".format(100 * train_total_correct / len(train_data)))
        print("Training total_age_accuracy discriminator = {:.4f}%".format(100 * train_age_correct_disc / len(train_data)))
        print("Training total_rating_accuracy discriminator= {:.4f}%".format(100 * train_rating_correct_disc / len(train_data)))
        print("Training total_age_accuracy classifier = {:.4f}%".format(100 * train_age_correct_classifier / len(train_data)))
        print("Training total_rating_accuracy classifier= {:.4f}%".format(100 * train_rating_correct_classifier / len(train_data)))
        print("Training total_age_F1 discriminator = {:.4f}%".format(
            f1_score(y_age.cpu(), age_pred_disc.cpu())))
        print("Training total_rating_F1 discriminator= {:.4f}%".format(
            f1_score(y.cpu(), rating_pred_disc.cpu(), average = 'weighted')))
        print("Training total_age_F1 classifier = {:.4f}%".format(
            f1_score(y_age.cpu(), age_pred_classifier.cpu())))
        print("Training total_rating_F1 classifier= {:.4f}%".format(
            f1_score(y.cpu(), rating_pred_classifier.cpu(), average = 'weighted')))
       # print("Training total_gender_accuracy = {:.4f}%".format(100 * train_gender_correct / len(train_data)))
       # print("Training total_location_accuracy = {:.4f}%".format(100 * train_location_correct / len(train_data)))

        model.eval()

        #total_loss = []
        valid_total_correct = 0
        valid_age_correct_disc = 0
        valid_rating_correct_disc = 0
        valid_age_correct_classifier = 0
        valid_rating_correct_classifier = 0
        valid_gender_correct = 0

        for iteration, batch in enumerate(valid_iter):
            text, seq_lens = batch.text
            text.resize_(text.shape[1], text.shape[0])
            y = torch.cuda.LongTensor(batch.rating)
            y_gender = batch.gender
            y_age = torch.cuda.LongTensor(batch.age)
            age_onehot = torch.cuda.FloatTensor(y_age.shape[0], 2)
            # In your for loop
            age_onehot.zero_()
            age_onehot[torch.arange(y_age.shape[0]), y_age] = 1

            rating_onehot = torch.cuda.FloatTensor(y.shape[0], 5)
            # In your for loop
            rating_onehot.zero_()
            rating_onehot[torch.arange(y.shape[0]), y] = 1

            y_location = batch.location
            age_pred_disc, rating_pred_disc, age_pred_classifier, rating_pred_classifier, private_disc_loss, \
            style_disc_loss, private_mul_loss, style_mul_loss, vae_and_cls_loss = model(
                text, seq_lens, y_age, y, iteration + 1, epoch == mconfig.epochs - 1)

            valid_age_correct_disc += (age_pred_disc == y_age).sum().item()
            valid_rating_correct_disc += (rating_pred_disc == y).sum().item()

            valid_age_correct_classifier += (age_pred_classifier == y_age).sum().item()
            valid_rating_correct_classifier += (rating_pred_classifier == y).sum().item()



        #print("Training total_loss = {}".format(total_loss))
        # print("Training total_rating_accuracy = {:.4f}%".format(100 * train_total_correct / len(train_data)))

        print("Valid total_age_accuracy discriminator = {:.4f}%".format(
            100 * valid_age_correct_disc / len(valid_data)))
        print("Valid total_rating_accuracy discriminator= {:.4f}%".format(
            100 * valid_rating_correct_disc / len(valid_data)))
        print("Valid total_age_accuracy classifier = {:.4f}%".format(
            100 * valid_age_correct_classifier / len(valid_data)))
        print("Valid total_rating_accuracy classifier= {:.4f}%".format(
            100 * valid_rating_correct_classifier / len(valid_data)))
        print("Valid total_age_F1 discriminator = {:.4f}%".format(
            f1_score(y_age.cpu(), age_pred_disc.cpu())))
        print("Valid total_rating_F1 discriminator= {:.4f}%".format(
            f1_score(y.cpu(), rating_pred_disc.cpu(), average='weighted')))
        print("Valid total_age_F1 classifier = {:.4f}%".format(
            f1_score(y_age.cpu(), age_pred_classifier.cpu())))
        print("Valid total_rating_F1 classifier= {:.4f}%".format(
            f1_score(y.cpu(), rating_pred_classifier.cpu(), average='weighted')))





        print("Saving states")
        #================ Saving states ==========================#
        if not os.path.exists(gconfig.model_save_path):
            os.mkdir(gconfig.model_save_path)
        # save model state
        torch.save(model.state_dict(), gconfig.model_save_path +
                   f'/model_epoch_{epoch+1}.pt')
        # save optimizers states
        torch.save({'private_disc': private_disc_opt.state_dict(), 'vae_and_cls': vae_and_cls_opt.state_dict()}, gconfig.model_save_path+'/opt_epoch_{epoch+1}.pt')
    # Save approximate estimate of different style embeddings after the last epoch
   # with open(gconfig.avg_style_emb_path) as f:
   #     pickle.dump(model.avg_style_emb, f)
    print("Training completed!!!")


    model.eval()
    total_correct = 0
    total_loss = 0

    test_age_correct_disc = 0
    test_rating_correct_disc = 0
    test_age_correct_classifier = 0
    test_rating_correct_classifier = 0

    print('Testing started!')

    for i,batch in enumerate(test_iter):
        text, seq_lens = batch.text

        text.resize_(text.shape[1], text.shape[0])
        y = torch.cuda.LongTensor(batch.rating)

        y_age = torch.cuda.LongTensor(batch.age)
        age_onehot = torch.cuda.FloatTensor(y_age.shape[0], 2)
        # In your for loop
        age_onehot.zero_()
        age_onehot[torch.arange(y_age.shape[0]), y_age] = 1

        rating_onehot = torch.cuda.FloatTensor(y.shape[0], 5)
        # In your for loop
        rating_onehot.zero_()
        rating_onehot[torch.arange(y.shape[0]), y] = 1

        age_pred_disc, rating_pred_disc, age_pred_classifier, rating_pred_classifier, private_disc_loss, \
        style_disc_loss, private_mul_loss, style_mul_loss, vae_and_cls_loss = model(
            text, seq_lens, y_age, y, iteration + 1, epoch == mconfig.epochs - 1)

        total_loss += vae_and_cls_loss.item()

        test_age_correct_disc += (age_pred_disc == y_age).sum().item()
        test_rating_correct_disc += (rating_pred_disc == y).sum().item()

        test_age_correct_classifier += (age_pred_classifier == y_age).sum().item()
        test_rating_correct_classifier += (rating_pred_classifier == y).sum().item()

    avg_loss =  total_loss / len(test_data)

    print("Test Avg. Loss: {:.4f}\n".format(avg_loss))

    print("Test total_age_accuracy discriminator = {:.4f}%".format(
        100 * test_age_correct_disc / len(test_data)))
    print("Test total_rating_accuracy discriminator= {:.4f}%".format(
        100 * test_rating_correct_disc / len(test_data)))
    print("Test total_age_accuracy classifier = {:.4f}%".format(
        100 * test_age_correct_classifier / len(test_data)))
    print("Test total_rating_accuracy classifier= {:.4f}%".format(
        100 * test_rating_correct_classifier / len(test_data)))
    print("Test total_age_F1 discriminator = {:.4f}%".format(
        f1_score(y_age.cpu(), age_pred_disc.cpu())))
    print("Test total_rating_F1 discriminator= {:.4f}%".format(
        f1_score(y.cpu(), rating_pred_disc.cpu(), average='weighted')))
    print("Test total_age_F1 classifier = {:.4f}%".format(
        f1_score(y_age.cpu(), age_pred_classifier.cpu())))
    print("Test total_rating_F1 classifier= {:.4f}%".format(
        f1_score(y.cpu(), rating_pred_classifier.cpu(), average='weighted')))

    print("Testing completed!!!")






if __name__ == "__main__":
    train()
   # analysis = tune.run(
   #     train, config={"style_adversary_loss_weight": tune.grid_search([0.001, 0.01, 0.1]),
   #                    "style_multitask_loss_weight": tune.grid_search([0.001, 0.01, 0.1])})

   # print("Best config: ", analysis.get_best_config(metric="vae_and_cls_loss"))