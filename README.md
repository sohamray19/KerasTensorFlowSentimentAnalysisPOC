# KerasTensorFlowSentimentAnalysisPOC
Model for identification of Positive/Negative Sentiments in texts using Multi-Layer Perceptron built using Keras API running on TensorFlow. This model is then saved in protobuf format(.pb) and the variables are frozen & saved so that it can be served as a RESTful service using Tensorflow Serving.

Explanation:

1. The tweets and their corresponding sentiments are read from 'Sentiment Analysis Dataset.csv'
2. The tweets are tokenized, vectorized and encoded using TfidfVectorizer from sklearn.feature_extraction.text
3. Then the K-Best Features are selected using chi2 statistical function which takes the encoded tweets and labels as input and calculates the importance score of feature.
4. Then the enocoded tweets are transformed so that only the K-Best features are retained.
5. Then MLP is defined and compiled which consists of 4 layers: 2 Dense and 2 Dropout
6. 1st Dense Layer consists of 32 neurons and uses 'relu' activation function.
7. 2nd Dense Layer consists of 1 neurons and uses 'sigmoid' activation function.
8. Then using the encoded tweets, we train the contructed model.
9. Once trained, this model can be used to predict the sentiment behind any text entered by the user.
10. This model is then saved in the protobuf format and the variables are frozen and saved using the 'save_model_for_production' function.

Steps for serving the model as a RESTful Service:

1. git clone https://github.com/tensorflow/serving
2. cd serving/tensorflow_serving/tools/docker
3. git clone https://github.com/brianalois/tensorflow_serving_tutorial.git
4. cd tensorflow_serving_tutorial
5. docker build --pull -t test-tensorflow-serving .
6. docker run -it -p 8500:8500 -p 8501:8501 -v /home/exacon02/Models/:/home/ test-tensorflow-serving
(ie instead of '/home/exacon02/Models/', use the path to where '/Sentiment' directory is located.)
7. tensorflow_model_server --port=8500 --rest_api_port=8501 --model_config_file=/home/models.conf
(ie models.conf file should be located in the same directory as the '/Sentiment' directory.)
8. This starts the RESTful Service which hosts the model.
9. POST requests can be sent to url 'http://localhost:8501/v1/models/Sentiment/versions/1:predict' to obtain the corresponding prediction.
