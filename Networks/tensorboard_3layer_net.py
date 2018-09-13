'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by Kohulan Rajan
'''
#Implemenation of tensorboard on a 3 layer network
import tensorflow as tf
import os
import sys
import numpy as np
import matplotlib as mpl
import csv
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import array

ts = datetime.now().strftime('%Y/%m/%d %H:%M:%S')

f = open('logfile_20180828test.txt', 'w',0)
sys.stdout = f
print(datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"packages loaded\n")
#f.write("packages loaded\n")
#Data input from image data

def label_data(is_test=False):
    data_path = "train"
    if is_test:
        data_path = "test"
    myFile = open('/home/ce85vof/Project/Profile_test_data/Labels_'+data_path+'.csv',"r")
    labels = []
    for row in myFile:
        x = int(row.strip().split(",")[1])
        labels.append(x)
    myFile.close()
    return np.asarray(labels)

y_train = label_data()
y_test = label_data(is_test=True)

print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Labels loaded !!")
#f.write("Labels loaded !!\n")
def one_hot(y, n_labels):
    mat = np.zeros((len(y), n_labels))
    for i, val in enumerate(y):
        mat[i, val] = 1
    return mat

#print(y_test)
#print(y_train)
#X_train= 0.95 - 0.9 / 255 * 255 - X_train
#X_test= 0.95 - 0.9 / 255 * 255 - X_test

# Parameters
learning_rate = 0.01
training_epochs = 35
batch_size = 100
dibeginay_step = 1
testbatch_size = 125
logs_path = '/home/ce85vof/Project/Profile_test_data/tensorflow_logs/test/'

# Network Parameters
n_hidden_1 = 512# 1st layer number of neurons
#n_hidden_2 = 512# 2nd layer number of neurons
n_input = 1024*1024 # Data input (img shape: 128*128)
n_classes = 90 # Bond_Count

# encoding labels to one_hot vectors
y_data_enc = one_hot(y_train, n_classes)
y_test_enc = one_hot(y_test, n_classes)

# tf Graph input
x = tf.placeholder("float", [None, n_input], name = 'Input_Data')
y = tf.placeholder("float", [None, n_classes], name = 'Label_Data' )

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name = 'W1'),
    #'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name = 'W2'),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]), name = 'W3')
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name = 'b1'),
    #'b2': tf.Variable(tf.random_normal([n_hidden_2]), name = 'b2'),
    'out': tf.Variable(tf.random_normal([n_classes]), name = 'b3')
}


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden fully connected layer
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    #Create a summary to visualize first layer
    tf.summary.histogram("relu1", layer_1)
    
    # Hidden fully connected layer
    #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    #layer_2 = tf.nn.relu(layer_2)
    #Create a summary to visualize second layer
    #tf.summary.histogram("relu2", layer_2)
    
    # Output fully connected layer with a neuron for each class
    out_layer = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])
    return out_layer

# Construct model
with tf.name_scope('Model'):
    logits = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
with tf.name_scope('Loss'):
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))

with tf.name_scope('SGD'):
    #Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    #Optimizer to calculate every variable gradient
    grads = tf.gradients(loss_op, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    
    #Update all variables according to their gradient
    apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

with tf.name_scope('Accuracy'):
    # Accuracy
    acc = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

#Initiating data for plots
loss_history = []
acc_history = []
acc_valid_history = []

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss_op)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", acc)
# Create summaries to visualize weights
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
# Summarize all gradients
for grad, var in grads:
    tf.summary.histogram(var.name + '/gradient', grad)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Training Started")

totaltest_batch = int(len(y_test)/testbatch_size)
with tf.Session() as sess:
    sess.run(init)

    #Write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(y_train)/batch_size)
        count=0
        counttest=0
        total_correct_preds = 0
             
        # Loop over all batches
        for l in range(total_batch):
            print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),count,"batch over")
            with open('/home/ce85vof/Project/Profile_test_data/train.txt', 'r') as fp:
                strarray = []
                
                j=0
                for i, line in enumerate(fp):
                    if i == count+j:
                        strarray.append(line.strip().split(".0,[")[1])
                        j+=1
                        if j >= batch_size:
                            fin = []
                            for k in strarray:
                                str = k[0:-1]
                                z = []
                                z = list(map(float,str.split(',')))
                                fin.append(z)
                                
                            loadedImages = np.array(fin).astype(np.float32)
                            #print(loadedImages[0])
                            batch_x=(1.0-(loadedImages/255.0))
                            batch_y=y_data_enc[count:(count+j)]
                            _, c, summary = sess.run([apply_grads, loss_op, merged_summary_op], feed_dict={x: batch_x,y: batch_y})
                            avg_cost += c / total_batch
                            loss_history.append(c)
                            # Write logs at every iteration
                            summary_writer.add_summary(summary, epoch * total_batch + l)
                            count = j+count
                            break

         # Dibeginay logs per epoch step
        if epoch % dibeginay_step == 0:
            print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))

    print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Optimization Finished!")

    # Test model
    totaltest_batch = int(len(y_test)/testbatch_size)
    for q in range(totaltest_batch):
        print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),counttest,"batch over")
        with open('/home/ce85vof/Project/Profile_test_data/test.txt', 'r') as fpt:
            strarrayt = []
            e=0
            for w, linet in enumerate(fpt):
                if w == counttest+e:
                    strarrayt.append(linet.strip().split(".0,[")[1])
                    e+=1
                    if e >= testbatch_size:
                        fint = []
                        for m in strarrayt:
                            str = k[0:-1]
                            d = []
                            d = list(map(float,str.split(',')))
                            fint.append(d)
                            
                        loadedImagest = np.array(fint).astype(np.float32)
                        X_test=(1.0-(loadedImagest/255.0))
                        Y_test=y_test_enc[counttest:(counttest+e)]
                        print (len(X_test))
                        acc_history.append(acc.eval({x: X_test, y: Y_test}))
                        print (datetime.now().strftime('%Y/%m/%d %H:%M:%S'),"Accuracy",acc.eval({x: X_test, y: Y_test}))
                        counttest = e+counttest
                        break

#Matplot plot generation
plt.subplot(2,1,1)
plt.plot(loss_history, '-o', label='Loss value')
plt.title('Training Loss')
plt.xlabel('Epoch x Batches')
plt.ylabel('Loss Value')
plt.legend(ncol=2, loc='upper right')
plt.subplot(2,1,2)
plt.plot(acc_history, '-o', label='Test Accuracy value')
plt.title('Test Accuracy')
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.legend(ncol=2, loc='lower right')
plt.gcf().set_size_inches(15, 30)
plt.savefig('plot_20180828.png')


f.close()