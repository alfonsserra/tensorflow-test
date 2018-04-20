package com.systelab.tensorflow;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

public class TensorflowTest {

    private byte[] graphDef;
    private List<String> labels;


    public TensorflowTest() {
        try {
            graphDef = Files.readAllBytes(Paths.get("./tensorflow_inception_graph.pb"));
            labels = Files.readAllLines(Paths.get("./category_labels.txt"), Charset.forName("UTF-8"));
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public void predict(String imagepath) {
        if (graphDef != null && labels != null) {
            try {
                byte[] imageBytes = Files.readAllBytes(Paths.get(imagepath));
                predict(imageBytes);
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    private void predict(byte[] imageBytes) {
        try (Tensor image = Tensor.create(imageBytes)) {
            float[] labelProbabilities = executeInceptionGraph(graphDef, image);
            int bestLabelIdx = maxIndex(labelProbabilities);
            System.out.println(String.format("BEST MATCH: %s (%.2f%% likely)", labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    private float[] executeInceptionGraph(byte[] graphDef, Tensor image) {
        try (Graph g = new Graph()) {
            g.importGraphDef(graphDef);
            try (Session s = new Session(g);
                 Tensor result = s.runner().feed("DecodeJpeg/contents", image).fetch("final_result").run().get(0)) {
                final long[] rshape = result.shape();
                if (result.numDimensions() != 2 || rshape[0] != 1) {
                    throw new RuntimeException(String.format("Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s", Arrays.toString(rshape)));
                }
                int nlabels = (int) rshape[1];
                Object[] resultMatrix;
                resultMatrix = (Object[]) result.copyTo(new float[1][nlabels]);
                return (float[]) (resultMatrix[0]);
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        return null;
    }

    private int maxIndex(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }

    public static void main(String[] args) {
        TensorflowTest tft = new TensorflowTest();
        tft.predict("./images/image1.jpeg");
        tft.predict("./images/image2.jpeg");
        tft.predict("./images/image3.jpeg");
    }
}
