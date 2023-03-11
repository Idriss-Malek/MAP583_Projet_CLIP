from Classifier import Classifier
import sys
import configparser
import openai
import time
from github import Github

if __name__ == "__main__":
    dataset = ImageFolder(root=sys.argv[1])
    labels = dataset.classes
    classifier = Classifier(labels)
    if len(sys.argv) > 2:
        classifier.load_classifier(str(sys.argv[2]))
    else:
        config = configparser.ConfigParser()
        config.read('config.ini')
        api_key = config['openai']['api_key']
        openai.api_key = api_key
        classifier.descriptors_fn(engine='text-davinci-002')
        classifier.save_classifier('descriptors/'+str(time.time()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    tokenized_text = torch.cat([clip.tokenize(label) for label in labels]).to(device)
    from time import time

    sum1 = 0
    sum2 = 0
    for i in range(0, len(dataset), 100):
        if i == 8900:
            subset = Subset(dataset, [j for j in range(i, len(dataset))])
            images = torch.stack([caltech.preprocess(image) for (image, label) in subset])
            lab = torch.Tensor([label for (image, label) in subset])
        else:
            subset = Subset(dataset, [j for j in range(i, i + 100)])
            images = torch.stack([caltech.preprocess(image) for (image, label) in subset])
            lab = torch.Tensor([label for (image, label) in subset])
        print(f'Starting predictions: {i}')
        t = time()
        pred1 = caltech.multi_classify(images, preprocessed=True)
        sum1 += torch.sum((pred1 == lab) + 0.)

        logits = model(images, tokenized_text)[0]
        pred2 = torch.argmax(logits, dim=1)
        sum2 += torch.sum((pred2 == lab) + 0.)
        print(time() - t)
    avg1 = sum1 / len(dataset)
    avg2 = sum2 / len(dataset)

    print(f'Accuracy of the custom algorithm: {avg1}')
    print(f'Accuracy of the traditional zero shot prediction: {avg2}')

    with open(sys.argv[1]+'.txt', 'w') as f:
        f.write(f'Accuracy of the custom algorithm: {avg1}')
        f.write(f'Accuracy of the traditional zero shot prediction: {avg2}')
    with open(sys.argv[1]+'.txt', 'r') as file:
        content = file.read()

    # authenticate with GitHub using a personal access token
    access_token = '' #tofill
    g = Github(access_token)

    # get the repository where you want to push the file
    repo = g.get_repo('Idriss-Malek/MAP583_Projet_CLIP')

    # create a new commit with the updated txt file
    commit_message = 'results'
    repo.create_file(sys.argv[1]+'.txt', commit_message, content)

    # push the changes to the repository
    branch = repo.get_branch('main')
    branch.edit(branch.commit.sha, None, 'results')




