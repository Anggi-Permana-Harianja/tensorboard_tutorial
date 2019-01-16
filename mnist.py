'''
CNN for MNIST by dendelion mane from tensorflow,
demonstrated how to take out the most of tensorboard. Learn this to have better insight to build a platform
'''

#import modules
import os
import os.path 
import shutil
import tensorflow as tf

#directories
logdir= 'graph_log_1/'
labels = 'labels_1024.tsv'
sprites = 'sprite_1024.png'

#mnist embeddings
mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir = logdir + "data", one_hot = True)

#conv_layer
def conv_layer(input, size_in, size_out, name = 'convolutional_layers'):
	#scoping graph
	with tf.name_scope(name):
		#variable definition
		weights = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev = 0.1), name = 'weights')
		bias = tf.Variable(tf.constant(0.1, shape = [size_out]), name = 'bias')
		
		#conv layer
		conv = tf.nn.conv2d(input, weights, strides = [1, 1, 1, 1], padding = 'SAME')

		#activation 
		activation = tf.nn.relu(conv + bias)

		#writing into tensorboard summary and naming graph entity
		tf.summary.histogram('weights', weights)
		tf.summary.histogram('biases', bias)
		tf.summary.histogram('activations', activation)

		return tf.nn.max_pool(activation, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


def fc_layer(input, size_in, size_out, name = 'fully_connected_layers'):
	#scope naming
	with tf.name_scope(name):
		#varible definition
		weights = tf.Variable(tf.truncated_normal([size_in, size_out], stddev = 0.1), name = 'weights')
		bias = tf.Variable(tf.constant(0.1, shape = [size_out]), name = 'bias')

		#y
		y = tf.matmul(input, weights)

		#activation
		activation = y + bias

		#writing into tensorboard summary and naming graph entity
		tf.summary.histogram('weights',weights)
		tf.summary.histogram('biases', bias)
		tf.summary.histogram('actiovations', activation)

		return activation


def mnist_model(learning_rate, use_two_fc, use_two_conv, hparam):
	#reset graph
	tf.reset_default_graph()

	sess = tf.Session()

	#placeholders
	x = tf.placeholder(tf.float32, shape = [None, 784], name = 'x')
	x_image = tf.reshape(x, [-1, 28, 28, 1])

	#write x_image into summary graph 
	tf.summary.image('input', x_image, 3)

	y = tf.placeholder(tf.float32, shape = [None, 10], name = 'labels')


	#ensamble various architecture
	if use_two_conv:
		conv1 = conv_layer(x_image, 1, 32, 'conv1')
		conv_out = conv_layer(conv1, 32, 64, 'conv2')
	else:
		conv_out = conv_layer(x_image, 1, 16, 'conv')

	#flattened the result of CNN to fed to FCNN
	flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])

	if use_two_fc:
		fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, 'fc1')
		#relu activation
		relu = tf.nn.relu(fc1)
		embedding_input = relu

		#writing to embedding tensorboard
		embedding_size = 1024
		tf.summary.histogram('fc1/relu', relu)

		#logits
		logits = fc_layer(relu, 1024, 10, 'fc2')
	else:
		embedding_input = flattened
		embedding_size = 7 * 7 * 64

		#logits 
		logits = fc_layer(flattened, 7 * 7 * 64, 10, 'fc')

	with tf.name_scope('cross_entropy'):
		xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y), name = 'xent')
		#write to tensorboard scalar
		tf.summary.scalar('xent', xent)

	#training scope
	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

	#accuracy scope
	with tf.name_scope('accuracy'):
		correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		#writing accuracy to tensorboard summary
		tf.summary.scalar('accuracy', accuracy)

	#merge all summary
	merged_summary = tf.summary.merge_all()

	#initialize embedding layer
	embedding = tf.Variable(tf.zeros([1024, embedding_size]), name = 'test_embedding')
	assignment = embedding.assign(embedding_input)

	saver = tf.train.Saver()

	#run global variable initializer
	sess.run(tf.global_variables_initializer())

	#writing all summary into tensorboards
	writer = tf.summary.FileWriter(logdir + hparam)
	writer.add_graph(sess.graph)

	#tensorboard configuration
	config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
	embedding_config = config.embeddings.add()
	embedding_config.tensor_name = embedding.name
	embedding_config.sprite.image_path = sprites
	embedding_config.metadata_path = labels
	#specify the width and height of a single thumbnail
	embedding_config.sprite.single_image_dim.extend([28, 28])
	#writing embedding into tensorboard
	tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

	for i in range(2001):
		batch = mnist.train.next_batch(100)

		if i % 5 == 0:
			#run accuracy and merge summary
			[train_accuracy, merged_summaries] = sess.run([accuracy, merged_summary], 
													feed_dict = {x : batch[0], y : batch[1]})
			#write into summary
			writer.add_summary(merged_summaries, i)

		if i % 500 == 0:
			sess.run(assignment, feed_dict = {x : mnist.test.images[ : 1024], y : mnist.test.labels[ : 1024]})
			saver.save(sess, os.path.join(logdir, "model.ckpt"), i)

		#train_step is actually optimizers
		sess.run(train_step, feed_dict = {x : batch[0], y : batch[1]})

def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
	conv_param = 'conv = 2' if use_two_conv else 'conv = 1'
	fc_param = 'fc = 2' if use_two_fc else 'fc = 2'

	return 'lr_%.0E, %s, %s' % (learning_rate, conv_param, fc_param)

def main():
	#various learning rate used and writen on tensorboard
	for learning_rate in [1E-3, 1E-4]:
		#various architectures
		for use_two_fc in [True, False]:
			for use_two_conv in [True, False]:
				
				#construct hyperparameters
				hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
				
				print('starting run for %s' % hparam)

				#run each hparam settings
				mnist_model(learning_rate, use_two_fc, use_two_conv, hparam)


	print('done training')
	print('run tensorboard --logdir=%s to see the results' % logdir)

if __name__ == '__main__':
	main()

















