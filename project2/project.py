"""# global variables"""

# sneaky global variable !!!!!

# bump this version every time new install is added/removed/modified
gl_latest_env_version = "1.1.1";

# if using drive
gl_drive_root = "./drive/MyDrive/autonomous_project2/";

# required files
gl_dataset_directory = "dataset/";
gl_weight_directory = "weights/";
gl_testimages_directory = "testImages/";
gl_signnames_file = "signnames.csv";

# number of test images
gl_num_images = 10;

# training/validation/testing file location
gl_training_file = gl_dataset_directory + "train.p";
gl_validation_file = gl_dataset_directory + "valid.p";
gl_testing_file = gl_dataset_directory + "test.p";

# convert to .py for windows user without anaconda
# jupyter-nbconvert --to python project.ipynb

"""# environment setup"""

def installEnvWrapper() -> None:
    """
    Update environment and throw error if not already installed.

    This is a standalone function, doesn't require external import.
    """
    env_result = installEnv(gl_latest_env_version);
    if (env_result != True):
        print("Changes might not apply immediately, restart runtime to apply changes.");
        assert(False); # force error

def installEnv(latest_version: str) -> bool | None:
    """
    Return true if already installed, false if not, None (or error) if error.
    """
    import os;
    import shutil;

    env_file = "env.txt";
    env_version = "0.0.0"; # default version
    if os.path.isfile(env_file):
        with open(env_file, "r") as f:
            version = f.readline()[:-1];
            env_version = version;
    else:
        with open(env_file, "x") as f:
            f.write(env_version + "\n");

    if env_version == gl_latest_env_version:
        print("version matched, install skipped.");
        return True;

    print("installing package...");
    installPackage("notebook");
    installPackage("jupyter");
    installPackage("opencv-python");
    installPackage("tensorflow");
    installPackage("keras");
    installPackage("pandas");
    installPackage("matplotlib");

    if inColab():
        if os.path.isdir(gl_drive_root):
            print("drive already mounted.");
        else:
            print("mounting drive...");
            from google.colab import drive;
            drive.mount("./drive", force_remount=False);

    fileok = True;
    fileok = fileok and checkFile(gl_dataset_directory, autocreate=False);
    fileok = fileok and checkFile(gl_weight_directory, autocreate=True);
    fileok = fileok and checkFile(gl_testimages_directory, autocreate=False);
    fileok = fileok and checkFile(gl_signnames_file, autocreate=False);
    if not fileok:
        return False;

    with open(env_file, "w") as f:
        f.write(gl_latest_env_version + "\n");
    print("version updated");
    return False;

def inColab() -> bool:
    import importlib.util;
    return importlib.util.find_spec("google.colab") is not None;

def installPackage(package_name: str) -> None:
    """
    Automatically install package.
    """
    import importlib.util;
    import subprocess;
    if importlib.util.find_spec(package_name):
        print(package_name + " is installed.");
    else:
        command = ["pip", "install", package_name];
        print("installing "+ package_name + "...");
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT);
        _ = proc.communicate(input='y'.encode())[0];

def checkFile(dstpath: str, autocreate: bool = False) -> bool:
    """
    automatically verify files and copy if needed.
    """
    import os;
    import shutil;
    if os.path.exists(dstpath):
        print(dstpath + " is up to date.");
    else:
        if inColab():
            srcpath = gl_drive_root + dstpath;
            if autocreate and not os.path.isdir(srcpath):
                os.makedirs(srcpath);
                print("creating " + dstpath + "...");
            print("setup "+ dstpath + "...");
            if os.path.isdir(srcpath):
                shutil.copytree(src=srcpath,
                                dst=dstpath);
            else:
                shutil.copy(src=srcpath,
                            dst=dstpath);
        else:
            if autocreate:
                print("creating " + dstpath + "...");
                os.mkdir(dstpath);
            else:
                print(dstpath + " not found, please manually place it.");
                return False;
    return True;

if __name__ == "__main__":
    installEnvWrapper();

"""# import"""

import numpy as np;

import pickle;
import cv2;

import tensorflow as tf;
import keras;
from keras import Model, layers, activations, metrics, losses;
from keras.src.optimizers.optimizer import Optimizer;

from typing import Callable, Any;

"""# functions

## dataset & preprocess data
"""

def loadDataset(usage):
    if usage == "train":
        use_file = gl_training_file;
    elif usage == "valid":
        use_file = gl_validation_file;
    elif usage == "test":
        use_file = gl_testing_file;
    else:
        raise ValueError;
    # image = tf.image.rgb_to_grayscale(image);
    with open(use_file, mode="rb") as f:
        dataset = pickle.load(f);
    return dataset["features"], dataset["labels"];

def preprocess(image, label, input_size: tuple[int, int, int]):
    input_x, input_y, input_z = input_size;
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (input_x, input_y));
    if input_z == 1:
        image = tf.image.rgb_to_grayscale(image);
    image = (image / 255.0);
    return image, label;

def augment(image, label, input_size: tuple[int, int, int]):
    image, label = preprocess(image, label, input_size);
    pad_amt = int(input_size[0] / 10);
    image = tf.image.resize_with_pad(image, input_size[0] + pad_amt, input_size[1] + pad_amt);
    image = tf.image.random_crop(image, size=input_size);
    image = tf.image.random_brightness(image, max_delta=0.1);
    image = tf.clip_by_value(image, 0, 1);
    return image, label;

def datasetTransform(dataset: tf.data.Dataset, map_fn, batch_size: int, shuffle: bool):
    if shuffle:
        dataset = dataset.shuffle(50000, reshuffle_each_iteration=True);

    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE);
    dataset = dataset.batch(batch_size);
    return dataset.prefetch(tf.data.experimental.AUTOTUNE);

"""## define model"""

def Lenet(n_classes: int) -> Model:
    model = keras.Sequential();
    model.add(keras.Input((32, 32, 1)));

    model.add(layers.Conv2D(6, (5, 5), activation=activations.sigmoid));
    model.add(layers.AveragePooling2D((2, 2)));
    model.add(layers.Conv2D(16, (5, 5), activation=activations.sigmoid));
    model.add(layers.AveragePooling2D((2, 2)));

    model.add(layers.Flatten());

    model.add(layers.Dense(120, activation=activations.sigmoid));
    model.add(layers.Dense(84, activation=activations.sigmoid));
    model.add(layers.Dense(n_classes, activation=activations.softmax));
    return model;

def VGGFake(n_classes: int) -> Model:
    model = keras.Sequential();
    model.add(keras.Input((64, 64, 3)));

    model.add(layers.Conv2D(32, (3, 3), activation=activations.relu));
    model.add(layers.MaxPooling2D((2, 2)));
    model.add(layers.Conv2D(64, (3, 3), activation=activations.relu));
    model.add(layers.Conv2D(64, (3, 3), activation=activations.relu));
    model.add(layers.MaxPooling2D((2, 2)));
    model.add(layers.Conv2D(128, (3, 3), activation=activations.relu));
    model.add(layers.Conv2D(128, (3, 3), activation=activations.relu));
    model.add(layers.MaxPooling2D((2, 2)));

    model.add(layers.Flatten());

    model.add(layers.Dense(1024, activation=activations.relu));
    model.add(layers.Dense(1024, activation=activations.relu));
    model.add(layers.Dense(n_classes, activation=activations.softmax));

    return model;

def VGGFakePlus(n_classes: int) -> Model:
    model = keras.Sequential();
    model.add(keras.Input((64, 64, 3)));

    model.add(layers.Conv2D(32, (3, 3), activation=activations.relu));
    model.add(layers.MaxPooling2D((2, 2)));
    model.add(layers.Conv2D(64, (3, 3), activation=activations.relu));
    model.add(layers.Conv2D(64, (3, 3), activation=activations.relu));
    model.add(layers.MaxPooling2D((2, 2)));
    model.add(layers.Conv2D(128, (3, 3), activation=activations.relu));
    model.add(layers.Conv2D(128, (3, 3), activation=activations.relu));
    model.add(layers.MaxPooling2D((2, 2)));

    model.add(layers.Flatten());

    model.add(layers.Dense(1024, activation=activations.relu));
    model.add(layers.Dropout(0.2));
    model.add(layers.Dense(1024, activation=activations.relu));
    model.add(layers.Dropout(0.2));
    model.add(layers.Dense(n_classes, activation=activations.softmax));

    return model;

def VGGFakeBN(n_classes: int) -> Model:
    model = keras.Sequential();
    model.add(keras.Input((64, 64, 3)));

    model.add(layers.Conv2D(32, (3, 3)));
    model.add(layers.BatchNormalization());
    model.add(layers.ReLU());
    model.add(layers.MaxPooling2D((2, 2)));
    model.add(layers.Conv2D(64, (3, 3)));
    model.add(layers.BatchNormalization());
    model.add(layers.ReLU());
    model.add(layers.Conv2D(64, (3, 3)));
    model.add(layers.BatchNormalization());
    model.add(layers.ReLU());
    model.add(layers.MaxPooling2D((2, 2)));
    model.add(layers.Conv2D(128, (3, 3)));
    model.add(layers.BatchNormalization());
    model.add(layers.ReLU());
    model.add(layers.Conv2D(128, (3, 3)));
    model.add(layers.BatchNormalization());
    model.add(layers.ReLU());
    model.add(layers.MaxPooling2D((2, 2)));

    model.add(layers.Flatten());

    model.add(layers.Dense(1024));
    model.add(layers.BatchNormalization());
    model.add(layers.ReLU());
    model.add(layers.Dropout(0.2));
    model.add(layers.Dense(1024));
    model.add(layers.BatchNormalization());
    model.add(layers.ReLU());
    model.add(layers.Dropout(0.2));
    model.add(layers.Dense(n_classes, activation=activations.softmax));

    return model;

def getModel(model_fn: Callable[[int], Model], n_classes: int, optimizer: Optimizer) -> Model:
    model: Model = model_fn(n_classes=n_classes);
    model.compile(optimizer=optimizer,
                loss=losses.SparseCategoricalCrossentropy(),
                run_eagerly=False,
                metrics=[
                    metrics.SparseCategoricalAccuracy(),
                ],
                );
    return model;

"""## train, evaluate and test"""

def trainModel(model: Model, train_dataset, valid_dataset, batch_size: int, epochs: int) -> Any:
    return model.fit(x=train_dataset,
                     epochs=epochs,
                     validation_data=valid_dataset,
                     shuffle=False,
                     );

def testModel(model: Model, test_dataset, batch_size: int) -> Any:
    return model.evaluate(x=test_dataset,
                          );

def predictModel(model: Model, x_test) -> np.ndarray:
    return model.predict(x=x_test,
                         batch_size=1,
                         );

"""## save and load weight"""

def saveWeight(model: Model, filename: str) -> None:
    filepath = gl_weight_directory + filename + ".weights.h5";
    model.save_weights(filepath=filepath, overwrite=True);

def loadWeight(model: Model, filename: str) -> None:
    filepath = gl_weight_directory + filename + ".weights.h5";
    model.load_weights(filepath=filepath, skip_mismatch=False);

"""## test inference image"""

def getImages() -> np.ndarray:
    import matplotlib.image as mpimg;
    images = [None] * gl_num_images;
    for i in range(gl_num_images):
        filename = gl_testimages_directory + str(i + 1) + '.png';
        img = cv2.imread(filename, cv2.IMREAD_COLOR); # rgb, 0~255, int
        images[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
    images = np.array(images);
    return images;

def getSignNames() -> np.ndarray:
    import pandas;
    df = pandas.read_csv(gl_signnames_file)['SignName'];
    return df.to_numpy();

# Load the images and plot them here.
def inference(x_real: np.ndarray, y_real: np.ndarray, sign_names: np.ndarray, model: keras.models.Sequential) -> None:
    prediction = predictModel(model, x_real);
    prediction_class = np.argmax(prediction, axis=1);
    prediction_value = np.max(prediction, axis=1);
    correct_count = 0;

    for i in range(gl_num_images):
        printfmt = "predicted class = {} ({}), value={:.4f}\nactual class = {} ({}).";
        if prediction_class[i] == y_real[i]:
            correct_count += 1;
        printstring = printfmt.format(prediction_class[i],
                                      sign_names[prediction_class[i]],
                                      prediction_value[i],
                                      y_real[i],
                                      sign_names[y_real[i]],
                                      );
        print(printstring);
    print("inference accuracy: {}/{}".format(correct_count, gl_num_images));

"""# main function"""

def main() -> None:
    """
    main function
    """
    # start
    print("begin");

    # hyperparameters
    batch_size = 16;
    epochs = 30;

    # do final test / inference or not
    do_final_test = True;
    do_inference = True;

    # load dataset
    X_train_raw, Y_train_raw = loadDataset("train");
    X_valid_raw, Y_valid_raw = loadDataset("valid");
    X_test_raw, Y_test_raw = loadDataset("test");
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_raw, Y_train_raw));
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid_raw, Y_valid_raw));
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test_raw, Y_test_raw));
    X_inference = getImages();
    Y_inference = np.array([17,12,14,11,38,4,35,33,25,13]);
    num_classes = len(np.unique(Y_train_raw)); # = 43

    # optimizer needs seperate instance
    model_list = [
        getModel(model_fn=Lenet, n_classes=num_classes, optimizer=keras.optimizers.Adam(learning_rate=0.001)),
        getModel(model_fn=VGGFake, n_classes=num_classes, optimizer=keras.optimizers.Adam(learning_rate=0.001)),
        getModel(model_fn=VGGFakePlus, n_classes=num_classes, optimizer=keras.optimizers.Adam(learning_rate=0.001)),
        getModel(model_fn=VGGFakeBN, n_classes=num_classes, optimizer=keras.optimizers.Adam(learning_rate=0.001)),
    ];
    model_name_list = [
        "lenet_adam_v1",
        "vgg_adam_v1",
        "vggplus_adam_v1",
        "vggbn_adam_v1",
    ];

    for model, model_name in zip(model_list, model_name_list):
        i_s = model.input_shape;
        input_size = (i_s[1], i_s[2], i_s[3]);

        # augment dataset, I can't extract this because a lot of function contains singleton variables
        # singleton (global?) variables prevents reusability

        train_ds = datasetTransform(train_dataset,
                                    lambda x, y: augment(x, y, input_size),
                                    batch_size=batch_size,
                                    shuffle=True);
        valid_ds = datasetTransform(valid_dataset,
                                    lambda x, y: preprocess(x, y, input_size),
                                    batch_size=batch_size,
                                    shuffle=False);
        test_ds = datasetTransform(test_dataset,
                                   lambda x, y: preprocess(x, y, input_size),
                                   batch_size=batch_size,
                                   shuffle=False);

        X_inference_proc, _ = preprocess(X_inference, Y_inference, input_size=input_size);

        # load weight(optional)
        loadWeight(model, model_name);

        # train model
        #trainModel(model, train_ds, valid_ds, batch_size, epochs);

        # save weight(optional)
        #saveWeight(model, model_name);

        # evaluate model
        print("=====evaluate model result:======");
        testModel(model, valid_ds, batch_size);
        print("=================================");

        # test model
        if do_final_test:
            print("=======test model result:=======");
            testModel(model, test_ds, batch_size);
            print("================================");

        # inference
        if do_inference:
            sign_names = getSignNames();
            inference(x_real=X_inference_proc,
                    y_real=Y_inference,
                    sign_names=sign_names,
                    model=model,
                    );

    # finish
    print("end");

if __name__ == "__main__":
    installEnvWrapper();
    main();