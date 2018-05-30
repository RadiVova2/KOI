
# coding: utf-8

# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import zipfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify' + filename + '. Can you get to it with a browser?')
    return filename

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        name = f.namelist()
        contents = f.read(names[0])
        text = tf.compat.as_str(contents)
        return text.split()
    
filename = maybe_download('text8.zip',31344016)
vocabulary = read_data(filename)
print('Data size', len(vocabulary))

vocabulary_size = 50000

def build_dataset(words, n_words):
    unique = collections.Counter(words)
    orders = unique.most_common(n_words - 1)
    count = [['UNK', -1]]
    count.extend(orders)
    
    dictionary = {word: i for i, (word, _) in enumerate(count)}
    
    data = []
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            count[0][1] += 1
        data.append(index)
        
    return data, count, list(dictionary.keys())

data, count, ordered_words = build_dataset(vocabulary, vocabulary_size)
del vocabulary, count

def generate_batch(data, batch_size, num_skips, skip_window, data_index):
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    
    span = 2 * skip_window + 1
    assert span > num_skips
    
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
        
    for i in range(batch_size // num_skips):
        
        targets = list(range(span))
        targets.pop(skip_window)
        np.random.shuffle(targets)
        
        start = i * num_skips
        
        batch[start:start+num_skips] = buffer[skip_window]
        
        for j in range(num_skips):
            labels[start+j, 0] = buffer[targets[j]]
            
        buffer.append(data[data_indes])
        data_index = (data_index + 1) % len(data)
        
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels, data_index


np.random.seed(1)
tf.set_random_seed(1)

batch_size = 128
embedding_size = 128 
skip_window = 1
num_skips = 2

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

truncated = tf.truncated_normal([vocabulary_size, embedding_size], 
                                stddev=1.0 / math.sqrt(embedding_size))
nce_weights = tf.Variable(truncated)
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

embedding = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

nce_loss = tf.nn.nce_loss(weights=nce_weights,
                          biases=nce_biases,
                          labels=train_labels,
                          inputs=embed,
                          num_sampled=num_sampled,
                          num_classes=vocabulary_size)
loss = tf.reduce_mean(nce_loss)

optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)


num_steps = 100001

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    
    average_loss, data_index = 0, 0
    for step in range(num_steps):
        batch_inputs, batch_labels, data_index = generate_batch(data, batch_size, num_skips, skip_window, data_index)
        
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
        
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            printf('Average loss at step {} : {}'.format(step, average_loss))
            average_loss = 0
            
        if step % 10000 == 0:
            sim = similarity.eval()
            
            for i in range(valid_size):
                valid_word = ordered_words[valid_examples[i]]
                
                top_k = 8
                nearest = sim[i].argsort()[-top_k - 1:-1][::-1]
                log_str = ', '.join([ordered_words[k] for k in nearest])
                print('Nearest to {}: {}'.format(valid_word, log_str))
                
    final_embeddings = normalized_embeddings.eval()
    
    
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    
    plt.figure(figsize=(18, 18))
    
    for (x,y), label in zip(low_dim_embs, labels):
        plt.scatter(x,y)
        plt.annotate(label,
                     xy=(x, y),
                    
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha = 'right',
                     va = 'bottom')
        
    plt.savefig(filename)
     
    
try:
    from sklearn.mainfold import TSNE
    import matplotlib.pyplot as plt
        
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only])
    labels = ordered_words[:plot_only]
        
    plot_with_labels(low_dim_embs,labels)
    
except ImportError:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')

