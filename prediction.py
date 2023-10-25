import tensorflow as tf
def predict(img_array):
  model = tf.keras.models.load_model('cifar10_model.h5')




  
    predictions = model.predict(img_array)
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'dog', 'deer', 'frog', 'horse', 'ship', 'truck']
