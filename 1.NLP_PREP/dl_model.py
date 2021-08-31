import tensorflow as tf
from tensorflow.keras import layers
import text_preprocessing as tp

# Base setting
batch_size = 2
num_epochs = 100

vocab_size = len(tp.word_index) + 1
emb_size = 128
hidden_dimension = 256
output_dimension = 1


# Keras Sequential
def sequential_model():
    model = tf.keras.Sequential([
        layers.Embedding(vocab_size, emb_size, input_length=4),
        layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)),
        layers.Dense(hidden_dimension, activation='relu'),
        layers.Dense(output_dimension, activation='sigmoid')])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    model.fit(tp.input_sequences, tp.labels, epochs=num_epochs, batch_size=batch_size)


# Keras Functional API
def functional_model():
    inputs = layers.Input(shape=(4,))
    embed_output = layers.Embedding(vocab_size, emb_size)(inputs)
    pooled_output = tf.reduce_mean(embed_output, axis=1)
    hidden_layer = layers.Dense(hidden_dimension, activation='relu')(pooled_output)
    outputs = layers.Dense(output_dimension, activation='sigmoid')(hidden_layer)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    model.fit(tp.input_sequences, tp.labels, epochs=num_epochs, batch_size=batch_size)


# Keras Custom Model
class CustomModel(tf.keras.Model):
    def __init__(self, vocab_size, embed_dimension, hidden_dimension, output_dimension):
        super(CustomModel, self).__init__(name='my_model')
        self.embedding = layers.Embedding(vocab_size, embed_dimension)
        self.dense_layer = layers.Dense(hidden_dimension, activation='relu')
        self.output_layer = layers.Dense(output_dimension, activation='sigmoid')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = tf.reduce_mean(x, axis=1)
        x = self.dense_layer(x)
        x = self.output_layer(x)
        return x


def custom_model():
    model = CustomModel(vocab_size=vocab_size, embed_dimension=emb_size, hidden_dimension=hidden_dimension,
                        output_dimension=output_dimension)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(tp.input_sequences, tp.labels, epochs=num_epochs, batch_size=batch_size)


# Keras Custom Layer
class CustomLayer(layers.Layer):
    def __init__(self, hidden_dimension, output_dimension, **kwargs):
        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension
        super(CustomLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense_layer1 = layers.Dense(self.hidden_dimension, activation='relu')
        self.dense_layer2 = layers.Dense(self.output_dimension)

    def call(self, inputs):
        hidden_output = self.dense_layer1(inputs)
        return self.dense_layer2(hidden_output)

    # Optional
    def get_config(self):
        base_config = super(CustomLayer, self).get_config()
        base_config['hidden_dim'] = self.hidden_dimension
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def custom_layer():
    model = tf.keras.Sequential([
        layers.Embedding(vocab_size, emb_size, input_length=4),
        layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)),
        CustomLayer(hidden_dimension, output_dimension),
        layers.Activation('sigmoid')])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    model.fit(tp.input_sequences, tp.labels, epochs=num_epochs, batch_size=batch_size)


if __name__ == '__main__':
    # sequential
    sequential_model()

    # functional
    functional_model()

    # custom model
    custom_model()

    # custom layer
    custom_layer()
