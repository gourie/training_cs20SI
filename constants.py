import tensorflow as tf
a = tf.constant([2,3], shape=[1,2], name="a")
b = tf.constant([[0,1],[2,3]], name="b")
x = tf.add(a,b, name="add")
y = tf.multiply(a,b, name="mul")
y2 = tf.matmul(a,b, name="matmul")
z = tf.range(3,18,3)
z = tf.linspace(10.0,15.0,5)
z = tf.random_normal([2,2], mean=0.0, stddev=2.0)

with tf.Session() as sess:
    # writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(x))
    print(sess.run(y2))
    print(sess.run(z))