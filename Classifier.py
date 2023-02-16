import torch
import clip
import matplotlib.pyplot as plt
import numpy as np
import heapq
from torch.nn.functional import log_softmax, softmax


class Classifier:
    def __init__(self, labels):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.labels = labels   #categories
        self.descriptors = {label: [] for label in labels}
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.gpt = 0  # todo
        self.tokenizer = 0  # todo
        self.threshold = 0.2
        self.image = None   #image to analyze
        self.similarity_desc = None #similarity scores for descriptors
        self.similarity = None  #similarity scores for categories
        self.softmax_similarity_desc = None   #probabilities for descriptors

    def descriptors_fn(self):
        #for label in self.labels
            # Method code here with GPT
            #descriptors = [label + ' which is/has ' + descriptor for descriptor in descriptors]
            #self.descriptors[label] = descriptors
        raise NotImplementedError("This function is not implemented yet")
    def compute_threshold(self):
        s=0
        for label in self.labels:
            s+= len(self.descriptors[label])
        s/= len(self.labels)
        self.threshold = 0.5/s
        print(self.threshold)
    def similarity_fn(self, images_, label):
        # Method code here
        single_image = False
        images = self.preprocess(images_).to(self.device)
        if len(images.shape) == 3:
            single_image = True
            images = images.unsqueeze(0)
        token_desc = clip.tokenize([label+desc for desc in self.descriptors[label]]).to(self.device)
        with torch.no_grad():
            logits_per_image, logits_per_descriptor = self.clip_model(images, token_desc)
            similarity = np.mean(logits_per_image.cpu().numpy(), axis=1)
            if single_image:
                self.similarity.append(similarity.item())
        return similarity, logits_per_image.numpy()

    def set_image(self, image):
        self.image = image
        self.similarity_desc = None
        self.similarity = None

    def classify(self):
        if self.image is None:
            print('Set Image first!')
            return
        self.similarity = []
        self.similarity_desc = {label: self.similarity_fn(self.image, label)[1][0] for label in self.labels}
        probs = softmax(torch.tensor(self.similarity), dim=-1)
        probs = [(label, probs[i].item()) for i, label in enumerate(self.labels)]
        top_labels = heapq.nlargest(5, probs)
        plt.imshow(self.image)
        print('This image may show: ')
        for i, pair in enumerate(top_labels):
            print('{}) {}: {}%'.format(i + 1, pair[0], pair[1]*100))

    def explain(self, label):
        if self.similarity_fn is None:
            print('Classify first!')
            return
        descriptors = self.descriptors[label]
        if self.softmax_similarity_desc is None:
            exp_similarity_desc = {key: np.exp(val) for (key, val) in self.similarity_desc.items()}
            sum_similarity_desc = np.sum([np.sum(val) for (key, val) in exp_similarity_desc.items()])
            #self.softmax_similarity_desc = {key: val / sum_similarity_desc for (key, val) in exp_similarity_desc.items()}  #softmax on all descriptors (labels combined)
            self.softmax_similarity_desc = {key: val / np.sum(val) for (key, val) in exp_similarity_desc.items()}  #for each label, softmax of descriptors
        plt.imshow(self.image)
        print(label + ' is characterized by the following features: ')
        for i in range(len(descriptors)):
            if self.softmax_similarity_desc[label][i] > self.threshold:
                print('\U00002705 {}: {}%'.format(descriptors[i], 100*self.softmax_similarity_desc[label][i]))  # green check emoji
            else:
                print('\U0000274C {}: {}%'.format(descriptors[i], 100*self.softmax_similarity_desc[label][i]))  # red mark emoji

    def reset(self):
        self.image = None
        self.similarity = None
        self.similarity_desc = None
        self.softmax_similarity_desc = None

    def reset_label(self, labels):
        self.labels = labels
        self.descriptors = {label: [] for label in labels}
        self.reset()
