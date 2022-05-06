function [] = Run_MLP()
%Run_MLP Train the MLP using the MNIST dataset and evaluate its performance.

    % Load MNIST.
    inputValues = LoadImages('train-images.idx3-ubyte');
    labels = LoadLabels('train-labels.idx1-ubyte');
    
    fprintf('Num of train Images: %d.\n',size(inputValues,2));
    
    % Transform the labels to correct target values.
    targetValues = 0.*ones(10, size(labels, 1));
    for n = 1: size(labels, 1)
        targetValues(labels(n) + 1, n) = 1;
    end;
    
    % Choose form of MLP:
    numberOfHiddenUnits = 1000;
    
    % Choose appropriate parameters.
    learningRate = 0.5;
    
    % Choose activation function.
    activationFunction = @logSigmoid;
    dActivationFunction = @dLogSigmoid;
    
    % Choose batch size and epochs.
    batchSize = 100;
    epochs = 500;
    
    fprintf('Train MLP with %d hidden units.\n', numberOfHiddenUnits);
    fprintf('Learning rate: %d.\n', learningRate);
    fprintf('Number of Epochs: %d.\n', epochs);
    
    [hiddenWeights, outputWeights, error] = TrainMLP(activationFunction, dActivationFunction, numberOfHiddenUnits, inputValues, targetValues, epochs, batchSize, learningRate);
    
    % Load validation set.
    inputValues = LoadImages('t10k-images.idx3-ubyte');
    labels = LoadLabels('t10k-labels.idx1-ubyte');
    
    % Choose decision rule.
    fprintf('Validation:\n');
    
    [correctlyClassified, classificationErrors] = ValidateMLP(activationFunction, hiddenWeights, outputWeights, inputValues, labels);
    
    fprintf('Classification errors: %d\n', classificationErrors);
    fprintf('Correctly classified: %d\n', correctlyClassified);
end