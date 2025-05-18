import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

class SkinClassifier:
    def _init_(self, model_path, layer_name, x_train_mean, x_train_std, class_names):
        self.model = tf.keras.models.load_model(model_path)
        self.layer_name = layer_name
        self.x_train_mean = x_train_mean
        self.x_train_std = x_train_std
        self.class_names = class_names

    def preprocess(self, image):
        image = cv2.resize(image, (100, 75))  # Match input shape
        image = image.astype('float32')
        image = (image - self.x_train_mean) / self.x_train_std
        return image

    def predict(self, image):
        image_processed = self.preprocess(image)
        image_batch = np.expand_dims(image_processed, axis=0)
        preds = self.model.predict(image_batch)[0]
        class_idx = np.argmax(preds)
        confidence = preds[class_idx]
        return class_idx, confidence, preds

    def get_gradcam_heatmap(self, image, class_index):
        image_processed = self.preprocess(image)
        img_tensor = tf.expand_dims(image_processed, axis=0)

     

        grad_model = tf.keras.models.Model(
                inputs=self.model.input,
                outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
                )


        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
        return heatmap.numpy()

    def overlay_gradcam(self, image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
        return overlayed

    def classify(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        class_idx, confidence, _ = self.predict(image)
        heatmap = self.get_gradcam_heatmap(image, class_idx)
        image_uint8 = np.uint8((image - np.min(image)) / (np.max(image) - np.min(image)) * 255)
        overlay_img = self.overlay_gradcam(image_uint8, heatmap)

        return {
            "predicted_class": self.class_names[class_idx],
            "confidence": float(confidence),
            "gradcam_overlay": overlay_img,
            "heatmap": heatmap
        }







# Example init (values to be replaced with your actual ones)
x_train_mean = 159.88411714650246
x_train_std = 46.45448942251351
class_names = ['Melanocytic nevi (Healhty)','Melanoma','Benign keratosis-like lesions ','Basal cell carcinoma','Actinic keratoses','Vascular lesions','Dermatofibroma']


model_path = '4_GEN_classifier.h5'

# Initialize classifier
classifier = SkinClassifier(model_path, layer_name='conv2d_4',
                            x_train_mean=x_train_mean,
                            x_train_std=x_train_std,
                            class_names=class_names)

# Load image
image = cv2.imread('test.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get results
result = classifier.classify(image)
print("Class:", result['predicted_class'])
print("Confidence:", result['confidence'])

# Save or display overlay image
cv2.imwrite("gradcam_result.jpg", cv2.cvtColor(result['gradcam_overlay'], cv2.COLOR_RGB2BGR))