import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class HandwrittenDigitRecognition {
    public static void main(String[] args) throws Exception {
        // Set the number of input and output neurons
        int numInput = 784; // 28x28 pixels
        int numOutputs = 10; // 10 classes (0-9)

        // Set the number of hidden layer neurons and the learning rate
        int numHidden = 100;
        double learningRate = 0.1;

        // Load the MNIST dataset
        MnistDataSetIterator mnistTrain = new MnistDataSetIterator(64, true, 12345);
        MnistDataSetIterator mnistTest = new MnistDataSetIterator(64, false, 12345);

        // Configure the neural network
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(123)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .learningRate(learningRate)
            .updater(org.nd4j.linalg.learning.config.Sgd.builder().build())
            .list()
            .layer(new DenseLayer.Builder().nIn(numInput).nOut(numHidden)
                .activation(Activation.RELU)
                .weightInit(org.deeplearning4j.nn.weights.WeightInit.XAVIER)
                .build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .weightInit(org.deeplearning4j.nn.weights.WeightInit.XAVIER)
                .nIn(numHidden).nOut(numOutputs).build())
            .build();

        // Create and initialize the neural network model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // Train the model
        int numEpochs = 15;
        for (int i = 0; i < numEpochs; i++) {
            model.fit(mnistTrain);
            System.out.println("Completed Epoch " + i);
        }

        // Evaluate the model on the test set
        DataSet testData = mnistTest.next();
        int[] predictions = model.predict(testData.getFeatures());
        int[] actual = testData.getLabels().argMax(1).toIntVector();

        // Calculate accuracy
        int correct = 0;
        for (int j = 0; j < predictions.length; j++) {
            if (predictions[j] == actual[j]) {
                correct++;
            }
        }
        double accuracy = (double) correct / predictions.length * 100;
        System.out.println("Accuracy: " + accuracy + "%");
    }
}
