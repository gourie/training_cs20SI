import tensorflow as tf

a = tf.Variable(2, name="scalar")
b = tf.Variable([2,3], name="vector")
c = tf.Variable([[0,1],[2,3]], name="matrix")
W = tf.Variable(tf.zeros([784,10]))
W = tf.Variable(tf.truncated_normal([784,8]))
U = tf.Variable(2 * W.initialized_value())  # ensure W is initialized before using its value to create U

# init all vars = assign their value
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # writer = tf.summary.FileWriter('./graphs', sess.graph)
    sess.run(init)
    print(W)
    print(W.eval())