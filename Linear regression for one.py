from __future__ import absolute_import
from __future__ import division

from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import main_op
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils

from __future__ import print_function
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf

#from tensorflow.python.lib.io import file_io
#from tensorflow.python.saved_model import main_op

#FLAGS = None

rng = numpy.random

#Parametros
learning_rate = 0.01
training_epochs = 1000
display_step = 50


#Datos de entrenamiento
train_X = numpy.asarray([8, 8, 3, 3, 3, 8, 3, 5, 3, 0, 0, 0, 2, 2, 0, 3, 5, 5, 3, 5, 0, 0, 3, 0, 8, 3, 2, 2, 2, 3, 
                       3, 3, 0, 0, 2, 3, 2, 8, 3, 2, 2, 8, 0, 5, 3, 8, 8, 3, 3, 0, 3, 2, 0, 0, 0, 3, 8, 3, 3, 3,
                       3, 3, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 8, 3, 0, 0, 0, 3, 0, 3, 3, 8, 8, 8, 3,
                       2, 2, 3, 3, 0, 3, 3, 3, 3, 2, 3, 0, 0, 3, 8, 8, 8, 8, 0, 3, 2, 2, 0, 0, 5, 3, 3, 3, 3, 3,
                       3, 3, 3, 3, 0, 8, 8, 0, 0, 0, 3, 8, 3, 2, 0, 3, 3, 3, 0, 2, 3, 0, 3, 3, 3, 3, 3, 0, 3, 0,
                       0, 0, 0, 3, 0, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5,
                       0, 0, 0, 0, 0, 0, 0, 2, 8, 0, 0, 8, 0, 8, 0, 0, 0, 8, 2, 5, 0, 0, 0, 8, 0, 0, 2, 3, 2, 3,
                       3, 3, 8, 2, 3, 2, 3, 0, 2, 2, 2, 3, 3, 0, 3, 0, 3, 3, 3, 3, 3, 3, 0, 3, 3, 2, 0, 2, 3, 2,
                       2, 2, 3, 2, 0, 3, 3, 0, 0, 3, 0, 3, 3, 2, 5, 2])
train_Y = numpy.asarray([3, 3, 2, 8, 8, 3, 5, 3, 8, 3, 8, 8, 8, 8, 8, 2, 3, 3, 8, 3, 8, 8, 2, 3, 3, 5, 3, 3, 8, 8,
                        8, 8, 3, 3, 3, 8, 3, 3, 2, 3, 8, 3, 3, 8, 8, 3, 3, 8, 8, 3, 5, 5, 3, 3, 3, 8, 3, 8, 5, 8,
                         5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 8, 3, 3, 3, 3, 3, 5, 3, 3, 3, 8, 3, 5, 5, 3, 3, 3, 5,
                         3, 3, 2, 8, 8, 8, 8, 2, 5, 8, 8, 8, 8, 5, 3, 3, 3, 3, 8, 2, 8, 3, 8, 8, 3, 8, 5, 8, 8, 8,
                         8, 5, 2, 8, 3, 3, 3, 8, 3, 3, 8, 3, 8, 3, 3, 5, 3, 8, 8, 8, 8, 8, 8, 8, 2, 5, 8, 3, 8, 8,
                         3, 3, 3, 8, 8, 8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5,
                         5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                         8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8])


n_samples = train_X.shape[0]

# Entradas de la grafica
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Pesos del modelo
w = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Desarrollo del modelo lineal
pred = tf.add(tf.multiply(X, w), b) # y=mx + b

# Error medio cuadrado
product = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# Gradiente descendiente
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(product)


# Inicio de variables
init = tf.global_variables_initializer()

# Inicio de entrenamiento
with tf.Session() as sess:
    
    sess.run(init)

    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        if (epoch+1) % display_step == 0:
            c = sess.run(product, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "product=", "{:.9f}".format(c), \
                "w=", sess.run(w), "b=", sess.run(b))

    print("Optimizacion terminada")
    training_product = sess.run(product, feed_dict={X: train_X, Y: train_Y})
    print("Producto de entrenamiento =", training_product, "w=", sess.run(w), "b=", sess.run(b), '\n')

    # Grafica
    plt.plot(train_X, train_Y, 'ro', label='Dato Originales')
    plt.plot(train_X, sess.run(w) * train_X + sess.run(b), label='Recta ajustada')
    plt.legend()
    plt.show()

    # Datos de prueba
    test_X = numpy.asarray([0, 0, 0, 0, 0, 0, 3, 5, 5, 3, 3])
    test_Y = numpy.asarray([8, 8, 3, 3, 3, 3, 2, 3, 3, 8, 5])

    print("Probando")
    testing_product = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  
    
    print("Producto de Prueba =", testing_product)
    print("Diferencia de pérdida cuadrática media absoluta :", abs(
        training_product - testing_product))
    
    from tensorflow.python.util.all_util import remove_undocumented


_allowed_symbols = [
    "builder",
    "constants",
    "loader",
    "main_op",
    "signature_constants",
    "signature_def_utils",
    "tag_constants",
    "utils",
]
remove_undocumented(__name__, _allowed_symbols) 

    #plt.plot(test_X, test_Y, 'bo', label='Datos de prueba')
    #plt.plot(train_X, sess.run(w) * train_X + sess.run(b), label='Recta ajustada')
    #plt.legend()
    #plt.show()
   


##Save the model#

  #  export_path_base = sys.argv[-1]
   # export_path = os.path.join(
    #compat.as_bytes(export_path_base),
    #compat.as_bytes(str(FLAGS.model_version)))
    
   # print 'Exporting trained model to', export path
    
   # builder = saved_model_builder.SavedModelBuilder(export_path)
    
   # classification_inputs = utils.build_tensor_info(serialized_tf_example) ####
   # classification_outputs_classes = utils.build_tensor_info(prediction_classes)####
   # classification_outputs_scores = utils.build_tensor_info(values)####
    
   # classification_signature = signature_def_utils.build_signature_def(
    #    inputs={},
     #   outputs={},
   # methd_name=signature_constants.CLASSIFY_METHOD_NAME)
    
    #    tensor_info_x = utils.build_tensor_info(x)
     #   tensor_info_y = utils.build_tensor_info(y)
        
      #  prediction_signature = signature_def_utils.build_signature_def(
       # inputs={'images': tensor_info_x},
       # outputs={'scores': tensor_info_y},
       # method_name=signature_constants.PREDICT_METHOD_NAME
       # )
        
      #  legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        
       # builder.add_meta_graph_and_variables(
        #sess, [tag_constants.SERVIN], signature_def_map={
         #   'predict_images':
          #  prediction_signature,
           # signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            #classification_signature,
        #}
        
        #legacy_init_op=legacy_init_op)
        
   # builder.save()
    #print 'Done exporting'
