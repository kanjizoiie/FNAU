import tensorflow as tf
import model

sess = tf.Session(config = tf.ConfigProto(log_device_placement=True))

m = model.model()

m.create()
m.compile()
m.train()
m.save("weights.h5")
