import tensorflow as tf
import numpy as np
import sys
from sklearn.metrics import precision_recall_fscore_support
import time


def softplus(x):
    return np.maximum(0, x) + np.log1p(np.exp(-np.abs(x)))


def main():

	neural_network_model_file = "./mlp.ckpt"


	img_size = 32
	num_channels = 3
	output_size = 10
	neuron_size = 4096
	init_val = 1E-2
	lr = 0.00001

	size = int(sys.argv[1])

	fold = int(sys.argv[2])


	print ("Setting network up")
	with tf.device("/device:GPU:0"):
		is_training = tf.placeholder(tf.bool)
		x = tf.placeholder("float32", [None, img_size, img_size, num_channels] , name="x")
		y = tf.placeholder("float32", [None, output_size]  )
		learning_rate = tf.placeholder("float32", None)

		conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=init_val))
		conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], mean=0, stddev=init_val))

		conv3_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=init_val))
		conv4_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], mean=0, stddev=init_val))

		conv5_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], mean=0, stddev=init_val))
		conv6_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], mean=0, stddev=init_val))
		conv7_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], mean=0, stddev=init_val))

		conv8_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 512], mean=0, stddev=init_val))
		conv9_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], mean=0, stddev=init_val))
		conv10_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], mean=0, stddev=init_val))


		conv11_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], mean=0, stddev=init_val))
		conv12_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], mean=0, stddev=init_val))
		conv13_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], mean=0, stddev=init_val))


		cnnbias = {
		'b1': tf.Variable(tf.random_normal([ 64 ], stddev=init_val, dtype=tf.float32), dtype=tf.float32),
		'b2': tf.Variable(tf.random_normal([ 64 ], stddev=init_val, dtype=tf.float32), dtype=tf.float32),

		'b3': tf.Variable(tf.random_normal([ 128 ], stddev=init_val, dtype=tf.float32), dtype=tf.float32),
		'b4': tf.Variable(tf.random_normal([ 128 ], stddev=init_val, dtype=tf.float32), dtype=tf.float32),

		'b5': tf.Variable(tf.random_normal([ 256 ], stddev=init_val, dtype=tf.float32), dtype=tf.float32),
		'b6': tf.Variable(tf.random_normal([ 256 ], stddev=init_val, dtype=tf.float32), dtype=tf.float32),
		'b7': tf.Variable(tf.random_normal([ 256 ], stddev=init_val, dtype=tf.float32), dtype=tf.float32),


		'b8': tf.Variable(tf.random_normal([ 512 ], stddev=init_val, dtype=tf.float32), dtype=tf.float32),
		'b9': tf.Variable(tf.random_normal([ 512 ], stddev=init_val, dtype=tf.float32), dtype=tf.float32),
		'b10': tf.Variable(tf.random_normal([ 512 ], stddev=init_val, dtype=tf.float32), dtype=tf.float32),


		'b11': tf.Variable(tf.random_normal([ 512 ], stddev=init_val, dtype=tf.float32), dtype=tf.float32),
		'b12': tf.Variable(tf.random_normal([ 512 ], stddev=init_val, dtype=tf.float32), dtype=tf.float32),
		'b13': tf.Variable(tf.random_normal([ 512 ], stddev=init_val, dtype=tf.float32), dtype=tf.float32)

		}



		A = tf.Variable(tf.random_normal([1], mean=1.1,  stddev=0.0, dtype=tf.float32), dtype=tf.float32)
		B = tf.Variable(tf.random_normal([1], mean=-0.01, stddev=0.0, dtype=tf.float32), dtype=tf.float32)
		C = tf.constant(0.0, dtype=tf.float32)
		D = tf.Variable(tf.random_normal([1], mean=-0.9, stddev=0.0, dtype=tf.float32), dtype=tf.float32)
		E = tf.Variable(tf.random_normal([1], mean=0.0, stddev=0.0, dtype=tf.float32), dtype=tf.float32)

		three = tf.constant(3.0, dtype=tf.float32)
		two = tf.constant(2.0, dtype=tf.float32)
		one = tf.constant(1.0, dtype=tf.float32)
		eps = tf.constant(1E-8, dtype=tf.float32)

		def UAF(vv,UAF_A,UAF_B,UAF_C,UAF_D,UAF_E):
			P1 = tf.multiply(UAF_A,(vv+UAF_B)) +tf.multiply(UAF_C,tf.pow(vv,two))
			P2 = tf.multiply(UAF_D,(vv-UAF_B))

			P3 = tf.nn.relu(P1) + tf.math.log1p(tf.exp(-tf.abs(P1)))
			P4 = tf.nn.relu(P2) + tf.math.log1p(tf.exp(-tf.abs(P2)))
			return P3 - P4  + UAF_E


		conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME') + cnnbias['b1']
		conv1_bn = tf.layers.batch_normalization(conv1,training = is_training)
		conv1_act = UAF(conv1_bn,A,B,C,D,E)

		conv2 = tf.nn.conv2d(conv1_act, conv2_filter, strides=[1,1,1,1], padding='SAME') + cnnbias['b2']
		conv2_bn = tf.layers.batch_normalization(conv2,training = is_training)
		conv2_act = UAF(conv2_bn,A,B,C,D,E)
		conv2_pool = tf.nn.max_pool(conv2_act, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')




		conv3 = tf.nn.conv2d(conv2_pool, conv3_filter, strides=[1,1,1,1], padding='SAME') + cnnbias['b3']
		conv3_bn = tf.layers.batch_normalization(conv3,training = is_training)
		conv3_act = UAF(conv3_bn,A,B,C,D,E)

		conv4 = tf.nn.conv2d(conv3_act, conv4_filter, strides=[1,1,1,1], padding='SAME') + cnnbias['b4']
		conv4_bn = tf.layers.batch_normalization(conv4,training = is_training)
		conv4_act = UAF(conv4_bn,A,B,C,D,E)
		conv4_pool = tf.nn.max_pool(conv4_act, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')





		conv5 = tf.nn.conv2d(conv4_pool, conv5_filter, strides=[1,1,1,1], padding='SAME') + cnnbias['b5']
		conv5_bn = tf.layers.batch_normalization(conv5,training = is_training)
		conv5_act = UAF(conv5_bn,A,B,C,D,E)

		conv6 = tf.nn.conv2d(conv5_act, conv6_filter, strides=[1,1,1,1], padding='SAME') + cnnbias['b6']
		conv6_bn = tf.layers.batch_normalization(conv6,training = is_training)
		conv6_act = UAF(conv6_bn,A,B,C,D,E)

		conv7 = tf.nn.conv2d(conv6_act, conv7_filter, strides=[1,1,1,1], padding='SAME') + cnnbias['b7']
		conv7_bn = tf.layers.batch_normalization(conv7,training = is_training)
		conv7_act = UAF(conv7_bn,A,B,C,D,E)

		conv7_pool = tf.nn.max_pool(conv7_act, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')





		conv8 = tf.nn.conv2d(conv7_pool, conv8_filter, strides=[1,1,1,1], padding='SAME') + cnnbias['b8']
		conv8_bn = tf.layers.batch_normalization(conv8,training = is_training)
		conv8_act = UAF(conv8_bn,A,B,C,D,E)

		conv9 = tf.nn.conv2d(conv8_act, conv9_filter, strides=[1,1,1,1], padding='SAME') + cnnbias['b9']
		conv9_bn = tf.layers.batch_normalization(conv9,training = is_training)
		conv9_act = UAF(conv9_bn,A,B,C,D,E)

		conv10 = tf.nn.conv2d(conv9_act, conv10_filter, strides=[1,1,1,1], padding='SAME') + cnnbias['b10']
		conv10_bn = tf.layers.batch_normalization(conv10,training = is_training)
		conv10_act = UAF(conv10_bn,A,B,C,D,E)

		conv10_pool = tf.nn.max_pool(conv10_act, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')




		conv11 = tf.nn.conv2d(conv10_pool, conv11_filter, strides=[1,1,1,1], padding='SAME') + cnnbias['b11']
		conv11_bn = tf.layers.batch_normalization(conv11,training = is_training)
		conv11_act = UAF(conv11_bn,A,B,C,D,E)

		conv12 = tf.nn.conv2d(conv11_act, conv12_filter, strides=[1,1,1,1], padding='SAME') + cnnbias['b12']
		conv12_bn = tf.layers.batch_normalization(conv12,training = is_training)
		conv12_act = UAF(conv12_bn,A,B,C,D,E)

		conv13 = tf.nn.conv2d(conv12_act, conv13_filter, strides=[1,1,1,1], padding='SAME') + cnnbias['b13']
		conv13_bn = tf.layers.batch_normalization(conv13,training = is_training)
		conv13_act = UAF(conv13_bn,A,B,C,D,E)


		# 9
		flat = tf.contrib.layers.flatten(conv13_act)


		weights = {
		'w1': tf.Variable(tf.random_normal([ 2048, neuron_size ], stddev=init_val, dtype=tf.float32), dtype=tf.float32),
		'w2': tf.Variable(tf.random_normal([ neuron_size, neuron_size ], stddev=init_val, dtype=tf.float32), dtype=tf.float32),
		'w3': tf.Variable(tf.random_normal([ neuron_size, output_size ], stddev=init_val, dtype=tf.float32), dtype=tf.float32)
		}

		bias = {
		'b1': tf.Variable(tf.random_normal([ neuron_size ], stddev=init_val, dtype=tf.float32), dtype=tf.float32),
		'b2': tf.Variable(tf.random_normal([ neuron_size ], stddev=init_val, dtype=tf.float32), dtype=tf.float32),
		'b3': tf.Variable(tf.random_normal([ output_size ], stddev=init_val, dtype=tf.float32), dtype=tf.float32)
		}


		# 10
		Y1 = tf.layers.batch_normalization(tf.matmul( flat , weights['w1'] )+bias['b1'],training = is_training)
		Y1act = UAF(Y1,A,B,C,D,E)

		Y2 = tf.layers.batch_normalization(tf.matmul( Y1act , weights['w2'] )+bias['b2'],training = is_training)
		Y2act = UAF(Y2,A,B,C,D,E)

		# 12
		yhat = UAF(tf.matmul( Y2act , weights['w3'] )+bias['b3'],A,B,C,D,E)


		train_vars = [conv1_filter
		, conv2_filter
		, conv3_filter
		, conv4_filter
		, conv5_filter
		, conv6_filter
		, conv7_filter
		, conv8_filter
		, conv9_filter
		, conv10_filter
		, conv11_filter
		, conv12_filter
		, conv13_filter



		, cnnbias['b1']
		, cnnbias['b2']
		, cnnbias['b3']
		, cnnbias['b4']
		, cnnbias['b5']
		, cnnbias['b6']
		, cnnbias['b7']
		, cnnbias['b8']
		, cnnbias['b9']
		, cnnbias['b10']
		, cnnbias['b11']
		, cnnbias['b12']
		, cnnbias['b13']


		,weights['w1']
		,weights['w2']
		,weights['w3']

		,bias['b1']
		,bias['b2']
		,bias['b3']
		]

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		update_ops2 = tf.get_collection(tf.GraphKeys.UPDATE_OPS)



		#ta = three*tf.math.sigmoid(tf.Variable(tf.random_normal([1], mean=0.5, stddev=0.000000001, dtype=tf.float32), dtype=tf.float32))
		#vc = tf.Variable(tf.random_normal([1], mean=0.2, stddev=0.000000000001, dtype=tf.float32), dtype=tf.float32)
		#tc = tf.nn.relu(vc) + tf.math.log1p(tf.exp(-tf.abs(vc))) + eps

		#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = yhat, labels = y))
		loss = tf.math.sqrt(tf.reduce_mean(  tf.square(tf.subtract(y, yhat))  ) )




		train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
		train_op2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=train_vars)
		train_op = tf.group([train_op, update_ops])
		train_op2 = tf.group([train_op2, update_ops2])




		tf.set_random_seed(4123)

		with tf.device("/device:CPU:0"):
			saver = tf.train.Saver()

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			#tf.train.write_graph(sess.graph_def, '.', 'cnn_lstm.pbtxt')

			#saver.restore(sess, neural_network_model_file)
			ims = []
			testloss = []
			trainloss = []

			prev = 10000
			gl = 100
			thres = 1E-10
			epoch = 0
			c = 1000


			resetvar = 0
			test_img = np.load("test_img.npy").astype("float32")
			test_label = np.load("test_label.npy").astype("float32")

			#for epoch in range(4000):
			st = time.process_time()
			while(1==1):
				train_img = np.load("train_img"+str(epoch%size)+".npy").astype("float32")
				train_label = np.load("train_label"+str(epoch%size)+".npy").astype("float32")



				feed_dict = {y: train_label,
				learning_rate: lr,
				is_training: True,
				x: train_img}


				if (gl < 0.08):
					lr = 0.000001
				else:
					lr = 0.00001

				if (resetvar == 0):
					_, gl= sess.run([ train_op, loss],
					feed_dict=feed_dict)
					if (gl < 0.01):
						resetvar = 1
						UA,UB,UD,UE= sess.run([ A,B,D,E],
						feed_dict=feed_dict)
						sess.run(tf.global_variables_initializer())
						A.load(UA)
						B.load(UB)
						D.load(UD)
						E.load(UE)
				elif (resetvar == 1):
					if (gl < 0.01):
						resetvar = 2
					_, gl= sess.run([ train_op2, loss],
					feed_dict=feed_dict)
				elif (resetvar == 2):
					_, gl= sess.run([ train_op, loss],
					feed_dict=feed_dict)


				if (epoch % 100) == 0:
					feed_dict = {is_training: False,
					x: test_img}

					output_y = sess.run(yhat,
	    			feed_dict=feed_dict)

					y_pred = np.argmax(output_y,axis=1).astype("int32")
					y_true = np.argmax(test_label,axis=1).astype("int32")
					_,_,f1,_ = precision_recall_fscore_support(y_true, y_pred, average='macro')
					testloss.append(f1)
					print(precision_recall_fscore_support(y_true, y_pred, average='macro'))


					feed_dict = {
					is_training: False,
					x: train_img}

					output_y = sess.run(yhat,
	    			feed_dict=feed_dict)

					y_pred = np.argmax(output_y,axis=1).astype("int32")
					y_true = np.argmax(train_label,axis=1).astype("int32")
					_,_,f1,_ = precision_recall_fscore_support(y_true, y_pred, average='macro')
					trainloss.append(f1)

					#UA,UB,UC,UD,UE= sess.run([ A,B,C,D,E],
					#feed_dict=feed_dict)

					UA,UB,UD,UE= sess.run([ A,B,D,E],
					feed_dict=feed_dict)

					ims.append(np.array([UA,UB,0.0,UD,UE]))






                #loss_val.append([gl,dl])
				print(epoch)
				epoch = epoch + 1
				print(gl)
				prev = c
				c = gl


				if (epoch > 12000):
					elapsed_time = time.process_time() - st

					data_dir = "time_"+str(size)+ "_" + str(fold) + ".npy"
					np.save(data_dir, np.array([elapsed_time, epoch]))


					feed_dict = {
					is_training: False,
					x: test_img}

					output_y = sess.run(yhat,
	    			feed_dict=feed_dict)


					y_pred = np.argmax(output_y,axis=1).astype("int32")
					y_true = np.argmax(test_label,axis=1).astype("int32")

					data_dir = "test_pred_"+str(size) + "_" + str(fold) + ".npy"
					np.save(data_dir, y_pred)

					data_dir = "test_true_"+str(size) + "_" + str(fold) + ".npy"
					np.save(data_dir, y_true)





					feed_dict = {
					is_training: False,
					x: train_img}

					output_y = sess.run(yhat,
	    			feed_dict=feed_dict)


					y_pred = np.argmax(output_y,axis=1).astype("int32")
					y_true = np.argmax(train_label,axis=1).astype("int32")

					data_dir = "train_pred_"+str(size) + "_" + str(fold) + ".npy"
					np.save(data_dir, y_pred)

					data_dir = "train_true_"+str(size) + "_" + str(fold) + ".npy"
					np.save(data_dir, y_true)


					trainloss = np.array(trainloss)
					np.save("trainloss"+str(size) + "_" + str(fold) + ".npy", trainloss)

					testloss = np.array(testloss)
					np.save("testloss"+str(size) + "_" + str(fold) + ".npy", testloss)

					ims = np.asarray(ims)
					np.save("ims"+str(size) + "_" + str(fold) + ".npy", ims)

					break




			#ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=2000)
			#ani.save('animation.gif', writer='imagemagick', fps=10)

			saver.save(sess, neural_network_model_file)










if __name__ == '__main__':
    main()
