import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD, Adam
import torch.nn.functional as F
import os
import nltk
import string
from nltk.probability import FreqDist
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import matplotlib.pyplot as plt
import numpy as np

# check for the availability of GPU with CUDA support
cuda_available = torch.cuda.is_available()
print(cuda_available)

RESUME_TRAINING = True # determines if training should be continued if a checkpoint is available
USE_ADAM = False # Switch between optimizer adam and sgd
PLOT_DATA = False   # Switch for plotting word frequency distributions
DEBUG_PRINT = False # Switch for printing debug info
DETAILED_DEBUG_PRINT = False # Swith for printing special checkpoint information at each epoch
REMOVE_RARE_WORDS = True # Switch for enabling rare words removal 

CONTEXT_SIZE = 5
EMBEDDING_DIM = 20
CONTEXT_SIZE_5 = 5

EPOCH = 50
VERVOSE = 5
corpus_attributes = {'No_Of_Words': 0, 
                     'Stop_Words_Count': 0, 
                     'Punctuation_Count': 0,
                     'Number_Count': 0,
                     'NoW_After_CleanUp' :0,
                     "Non_english_word_Count":0}


"""
Saves a checkpoint of the training to a file.
This can be used to resume training or to reload a specific model state.

IMPORTANT: 
If you change context_size, dimensions, optimizers, ... you need to delete or rename the checkpoint, as the dimensions of the tensors wont match with the previous settings!
"""
def save_model(model, optimizer, current_epoch, checkpoint_file):
  if DEBUG_PRINT and DETAILED_DEBUG_PRINT:
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
      print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
      print(var_name, "\t", optimizer.state_dict()[var_name])
  
  checkpoint = {
          #'model': CBOW(model.vocab_size, model.embedding_size, model.context_size),
          'state_dict': model.state_dict(),
          'optimizer' : optimizer.state_dict(),
          'epoch': current_epoch + 1}
  
  torch.save(checkpoint, checkpoint_file)

# Loads a checkpoint file
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)    
    return checkpoint

# Resets a model to a given checkpoint
def reset_model_to_checkpoint(model, optimizer, checkpoint_file):
  print('loading checkpoint...\n')
  checkpoint = load_checkpoint(checkpoint_file)
  #model = checkpoint['model']
  model.load_state_dict(checkpoint['state_dict'])  
  optimizer.load_state_dict(checkpoint['optimizer'])
  model = model_to_cuda(model)

  if RESUME_TRAINING:
    # reset epoch if training is to be resumed
    current_epoch =checkpoint['epoch']
    print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))
  else:
    # freez params
    for parameter in model.parameters():
      parameter.requires_grad = False
  # evaluate model with the checkpoint data
  model.eval()
  return model, optimizer, current_epoch

# function to send tensors to gpu if cuda is enabled
def tensor_to_cuda(tensor):
    if cuda_available:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = tensor.to(device)
    return tensor

# enables cuda on the given model
def model_to_cuda(model):
    if cuda_available:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.cuda()
    return model



class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_size, context_size):
        super(CBOW, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        # return vector size will be context_size*2*embedding_size
        self.lin1 = nn.Linear(self.context_size * 2 * self.embedding_size, 50)
        self.lin2 = nn.Linear(50, self.vocab_size)
    
    def forward(self, inp):
        """
        CBOW Structure:
        2 Layers
        1 Hidden Layer using ReLu
        """
        out = self.embeddings(inp).view(1, -1)
        out = out.view(1, -1)
        out = self.lin1(out)
        out = F.relu(out)
        out = self.lin2(out)
        out = F.log_softmax(out, dim=1)
        return out
    
    def get_word_vector(self, word_idx):
        
        '''
        Returns the vector corresponding to a word in the embedding
        '''
        
        word = Variable(torch.LongTensor([word_idx]))
        
        # use cuda support if available and send the tensors to the correct device
        if cuda_available:
          self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          word = word.to(self.device)

        return self.embeddings(word).view(1, -1)
    
    def write_embedding_to_file(self,filename):
        for i in self.embeddings.parameters():
            weights = i.data.numpy()
        np.save(filename,weights)
    
class DATA_PREPROCESSOR():
    def __init__(self, file_path):
        self.file_path = file_path

    def readDataSet(self):
        '''
        Read input corpus file 
        '''
        filename = self.file_path
        print(filename)    
        
        with open(filename, 'r',encoding="utf8") as file:
            corpus = file.read().replace('\t', '')
        
        return corpus

    def removeStopWords(self, corpus):
        '''
        Remove Stop Words, Punctuations, Special Characters, Rare Words
        Fill data structures for plotting the data distribution
        Stop words list is taken from NLTK
        '''
        from nltk.corpus import stopwords 
        from nltk.tokenize import word_tokenize 
        from nltk.tokenize.treebank import TreebankWordDetokenizer
        nltk.download('words')
        nltk.download('stopwords')
        nltk.download('punkt')
        stop_words = set(stopwords.words('english')) 
        word_tokens = word_tokenize(corpus) 
        filtered_sentence = [] 
        stop_word_count = 0
        punctuation_cnt = 0
        digit_cnt = 0
        punc = set(string.punctuation)
        words = set(nltk.corpus.words.words())
        nonEnglish=0
        
        for w in word_tokens: 
            # Calculate non english words
            if w.lower() not in words:
                nonEnglish+=1
            # Calculate Punctuation Count    
            if w in punc:
                punctuation_cnt += 1  
            # Calculate Nuymeric Digits Count     
            elif w.isnumeric():
                digit_cnt += 1
                
            elif w not in stop_words: 
                filtered_sentence.append(w)
                
            else:
                stop_word_count+= 1  
        # Remove rare words
        if REMOVE_RARE_WORDS:
            fdist = FreqDist(filtered_sentence)
            # Get list of words with frequency less than 5
            rare_words = list(filter(lambda x: x[1]<=5,fdist.items()))
            if DEBUG_PRINT:
                print(rare_words)
            for word in filtered_sentence:  
                for entry in rare_words:
                    if word in entry:
                        filtered_sentence.remove(word)
        
        # Update internal data structure used in plotting         
        corpus_attributes.update({"Stop_Words_Count":(stop_word_count)})
        corpus_attributes.update({"Punctuation_Count":(punctuation_cnt)})
        corpus_attributes.update({"No_Of_Words":(len(word_tokens))})   
        corpus_attributes.update({"Number_Count": (digit_cnt)})
        corpus_attributes.update({"Non_english_word_Count": (nonEnglish)})
        if DEBUG_PRINT:
            print('stop_word_count', stop_word_count)
            print('punctuation_cnt', punctuation_cnt)
            print('No_Of_Words', (len(word_tokens)))
            print('Number_Count', digit_cnt)
            print('Non_english_word_Count', nonEnglish)
        return TreebankWordDetokenizer().detokenize(filtered_sentence)
    # add abbreviations won't don't etc..

    def preprocess_data(self):
        '''
        Preprocess the input corpus. Following steps are involed in data cleaning
        a) Make all words lowercase
        b) Remove the stop words
        c) Remove numeric digits
        d) Remove puntuation and special characters
        e) Remove rare words. 
        '''
        corpus = self.readDataSet()
    
        # remove additional spaces
        corpus = corpus.strip()
    
        # make everything lowercase
        corpus = corpus.lower()
      
        #remove stop words
        corpus = self.removeStopWords(corpus)
     
        # remove puntuations
        corpus = corpus.translate(str.maketrans('', '', string.punctuation))
        #clean_data = corpus
        corpus = corpus.split()
    
        for i,x in enumerate(corpus):
            if '’' in x:
                del corpus[i]
                
        if DEBUG_PRINT:
            print('Corpus Size After CleanUp',len(corpus))    
        corpus_attributes.update({"NoW_After_CleanUp":(len(corpus))})
   
        return corpus
    
        print(len(data))

class MODEL_EXECUTOR():
    def __init__(self, model):
        self.model = model
    
    def train(self, optimizer, data, unique_vocab, word_to_idx, current_epoch, checkpoint_file):
        '''
        Train the model. Training parameters will come from self class
        Epoch is defined as a macro
        '''
        nll_loss = nn.NLLLoss()  # loss function
        print(len(data))
        print('Starting training...\n')
        for epoch in range(current_epoch, EPOCH):
            total_loss = 0
            for context, target in data:           
                # use cuda support if available and send the model and tensors to the correct device
                self.model = model_to_cuda(self.model)
                inp_var = tensor_to_cuda(Variable(torch.LongTensor([word_to_idx[word] for word in context])))
                target_var = tensor_to_cuda(Variable(torch.LongTensor([word_to_idx[target]])))

                # set grad to zero for each context
                self.model.zero_grad()
                # calculate loss
                log_prob = self.model(inp_var)
                loss = nll_loss(log_prob, target_var)
                # execute backward pass
                loss.backward()
                optimizer.step()
                total_loss += loss.data
            
            if epoch % VERVOSE == 0:
                loss_avg = float(total_loss / len(data))
                print("{}/{} loss {:.2f}".format(epoch, EPOCH, loss_avg))
            # save current checkpoint
            save_model(self.model, optimizer,epoch, checkpoint_file)
        return self.model

    def test(self, unique_vocab, word_to_idx):
        '''
        Print the cosine similarity between random words from the vocab
         
        '''
        # test word similarity
        word_1 = unique_vocab[2]
        word_2 = unique_vocab[3]
        
        word_1_vec = self.model.get_word_vector(word_to_idx[word_1])
        word_2_vec = self.model.get_word_vector(word_to_idx[word_2])
        
        word_similarity = (word_1_vec, word_2_vec)
        word_similarity = (((word_1_vec.squeeze()).dot(word_2_vec.squeeze())) / (torch.norm(word_1_vec) * torch.norm(word_2_vec))).data.item()
        print("Similarity between '{}' & '{}' : {:0.4f}".format(word_1, word_2, word_similarity))

def plot (corpus):
    '''
    Plot the frequency distribution of 50 most common and least common words
    Plot the wods distribution of input corpus
    '''
    names = list(corpus_attributes.keys())
    values = list(corpus_attributes.values())
    plt.figure(figsize=(10, 5))
#   tick_label does the some work as plt.xticks()
    plt.bar(range(len(corpus_attributes)),values,tick_label=names)
    plt.show()

    fdist = FreqDist(corpus)
    plt.figure(figsize=(10, 5))

    fdist.plot(50,cumulative=False,title='Most frequent Words', linewidth=2)

def create_context(corpus_text):
    # consider 2*CONTEXT_SIZE as context window where middle word as target
    '''
    Create Context from the input corpus. 
    Context length will be decided based on context size
    '''
    data = []
    for i in range(CONTEXT_SIZE, len(corpus_text) - CONTEXT_SIZE):
        data_context = []
        # Append left side of context
        for j in range(CONTEXT_SIZE):
            data_context.append(corpus_text[i - CONTEXT_SIZE + j])
        #Append right side of context
        for j in range(1, CONTEXT_SIZE + 1):
            data_context.append(corpus_text[i + j])
        #Get center word    
        data_target = corpus_text[i]
        # Create the context and target word( Need to use append not extend)
        data.append((data_context, data_target))
 
    print("Some data: ",data[:4])

    unique_vocab = list(set(corpus_text))
      # mapping to index
    word_to_idx = {w: i for i, w in enumerate(unique_vocab)}
    return data,  unique_vocab, word_to_idx 

def print_closest_word(cbow, word, word_to_idx,unique_vocab):
    closest_word = get_closest_word(cbow, word, word_to_idx, unique_vocab)
    print("Closest words to %s are %s" % (word, closest_word))


def get_closest_word(cbow, word, word_to_idx,unique_vocab):
    '''
    Returns 5 closest neighbours( determined by cosine similarity)
    for the input word.
    Returns a list of 5 closest neighbours and their distansce from the word
    '''
    word_distance = []
    topn=5
    emb = cbow.embeddings
    pdist = nn.PairwiseDistance()
    pdist = nn.CosineSimilarity(dim=1, eps=1e-6)
    i = word_to_idx[word]
    # use cuda support if available and send the tensors to the correct device
    lookup_tensor_i = tensor_to_cuda(torch.tensor([i], dtype=torch.long))

    v_i = emb(lookup_tensor_i).view(1, -1)
    index_to_word = {idx: w for (idx, w) in enumerate(unique_vocab)}
    for j in range(len(unique_vocab)):
        if j != i:
            # use cuda support if available and send the tensors to the correct device
            lookup_tensor_j = tensor_to_cuda(torch.tensor([j], dtype=torch.long))
            v_j = emb(lookup_tensor_j)
            word_distance.append((index_to_word[j], float(abs(pdist(v_i, v_j)))))
    word_distance.sort(key=lambda x: x[1])
    return word_distance[:topn]


def show_closest_words(cbow, word_to_idx, unique_vocab):
    # Todo get run frequency calc on whole corpus and get one rare, one normal and one frequent each
    print('\n verbs:\n')
    print_closest_word(cbow, 'come',word_to_idx,unique_vocab)
    print_closest_word(cbow, 'see',word_to_idx,unique_vocab)
    print_closest_word(cbow, 'take',word_to_idx,unique_vocab)
    
    print('\n noun:\n')
    print_closest_word(cbow, 'king',word_to_idx,unique_vocab)
    print_closest_word(cbow, 'father',word_to_idx,unique_vocab)
    print_closest_word(cbow, 'lady',word_to_idx,unique_vocab)
    
    print('\n adjectives:\n')
    print_closest_word(cbow, 'noble',word_to_idx,unique_vocab)
    print_closest_word(cbow, 'great',word_to_idx,unique_vocab)
    print_closest_word(cbow, 'fair',word_to_idx,unique_vocab)
    
def display_word_similarity(model):
    word_1 = 'king'
    word_2 = 'queen'
    word_1_vec = model.get_word_vector(word_to_idx[word_1])
    word_2_vec = model.get_word_vector(word_to_idx[word_2])
    word_similarity = (word_1_vec, word_2_vec)
    word_similarity = (((word_1_vec.squeeze()).dot(word_2_vec.squeeze())) / (torch.norm(word_1_vec) * torch.norm(word_2_vec))).data.item()
    print("Similarity between '{}' & '{}' : {:0.4f}".format(word_1, word_2, word_similarity))

def main():
    """
    main function
    In order to run CBOW 2 or 5 change CONTEXT_SIZE to 2 or 5 respectively
    """
    preprocessor = DATA_PREPROCESSOR('shakespeare-corpus.txt')
    corpus = preprocessor.preprocess_data()
    plot(corpus)
    data, unique_vocab, word_to_idx = create_context(corpus)

    #train model- changed global variable if needed
    model=CBOW(len(unique_vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    if USE_ADAM:
        print('Using adam as optimizer')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        print('Using SGD as optimizer')
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    checkpoint_file ='checkpoint.pth'
    checkpoint_available= os.path.exists(checkpoint_file)
    if checkpoint_available:
      model, optimizer, current_epoch = reset_model_to_checkpoint(model, optimizer, checkpoint_file)
    else:
      print('no checkpoint found. initializing new model..\n')
      current_epoch=0  

    executor = MODEL_EXECUTOR(model)
    if RESUME_TRAINING or not checkpoint_available:
      print('resuming training...\n')
      import time
      start_time = time.time()
      cbow = executor.train(optimizer, data, unique_vocab, word_to_idx, current_epoch, checkpoint_file)
      print("--- %s seconds ---" % (time.time() - start_time))
    else:
      print('pre-trained model loaded. no further training...\n')

    # get two words similarity
    executor.test(unique_vocab,word_to_idx)

    show_closest_words(cbow, word_to_idx,unique_vocab)
if __name__ == "__main__": main()
