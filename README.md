# Object Recognition using TensorFlow and Java

This small demo was written in Java to recognize objects in images and classify it using Inception models.

The pre-trained Inception V3 model can be downloaded from here https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip

## Getting Started

To get you started you can simply clone the `tensorflow-test` repository and install the dependencies:

### Prerequisites

You need [git][git] to clone the `tensorflow-test` repository.

You will need [Java™ SE Development Kit 8][jdk-download] and [Maven][maven].

### Clone `tensorflow-test`

Clone the `tensorflow-test` repository using git:

```bash
git clone https://github.com/systelab/tensorflow-test.git
cd tensorflow-test
```

### Install Dependencies

In order to install the dependencies you must run:

```bash
mvn install
```


## Run:

Specify the model and the images to test in the source code and then run TensorflowTest from your IDE.


[git]: https://git-scm.com/
[maven]: https://maven.apache.org/download.cgi
[jdk-download]: http://www.oracle.com/technetwork/java/javase/downloads
