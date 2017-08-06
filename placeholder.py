import tensorflow as tf

a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)
c = a + b

with tf.Session() as sess:
    # writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(c, {a: [1,2,3]} ))   # feed values as with key the tensor variable a
    # feed multiple values, one at at time
    # for a_value in values_list:
    #     print(sess.run(c, {a: a_value}))

# same principle of feeding a tensor y
z = tf.placeholder(tf.float32, shape=None)
x = tf.add(2,5)
y = tf.multiply(z,3)

with tf.Session() as sess2:
    replace_dict = {z: 5}
    print(sess2.run(y, feed_dict= replace_dict))


# lazy loading
a1 = tf.Variable(10)
b1 = tf.Variable(20)
c1 = tf.add(a1,b1)
with tf.Session() as sess3:
    sess3.run(tf.global_variables_initializer())
    for _ in range(10):
        sess3.run(c1)
    print(c1.eval())