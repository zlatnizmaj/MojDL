# https://towardsdatascience.com/tensorflow-for-complete-begineers-getting-started-with-tensors-780f846f007
import tensorflow as tf

print(tf.version)

#The science (and art) of creating tensors
scalar_val = tf.Variable(123, tf.int16)
floating_val = tf.Variable(123.456, tf.float32)
string_val = tf.Variable("hello everyone. Nice to learn tensoflow!", tf.string)

#Let us display the values (print) these tensors
print(scalar_val)
print(floating_val)
print(string_val)

#The idea behind shape and rank of tensors
#Shape: Describes the dimension of the tensor (total elements contained along each dimension)
scalar_val_shap = tf.shape(scalar_val)
print(scalar_val_shap)

floating_val_shap = tf.shape(floating_val)
print(floating_val_shap)

#Now, if we use e.g. lists/nested lists instead of just a “single” scalar value
list_tensor1 = tf.Variable([1, 3, 5, 6], tf.int16)
print(list_tensor1)
print(tf.shape(list_tensor1))

list_tensor2 = tf.Variable([[1, 2, 3], [4, 5, 6]], tf.int16)
print(list_tensor2)
print(tf.shape(list_tensor2))

#how about the rank? It describes the level of nesting within the tensor in simple words.
print(tf.rank(list_tensor1))
print(tf.rank(list_tensor2))