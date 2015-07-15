import java.io.*;


/**
 * Created by ziyihua on 07/07/15.
 */


public class ImportFile {

    public static int[] getLabel(String filename) throws IOException {
        DataInputStream labels = new DataInputStream(new FileInputStream(filename));

        int magicNumber = labels.readInt();
        if (magicNumber != 2049) {
            System.err.println("Label file has wrong magic number: " + magicNumber + " (should be 2049)");
            System.exit(0);
        }

        int numLabels = labels.readInt();

        int numLabelsRead = 0;

        int[] label_t = new int[numLabels];

        while (labels.available() > 0 && numLabelsRead < numLabels) {
            byte label = labels.readByte();
            label_t[numLabelsRead] = label;
            numLabelsRead++;
        }

        return label_t;
    }

    public static double[][][] getImage(String filename) throws IOException {
        DataInputStream images = new DataInputStream(new FileInputStream(filename));

        int magicNumber = images.readInt();
        if (magicNumber != 2051) {
            System.err.println("Image file has wrong magic number: " + magicNumber + " (should be 2051)");
            System.exit(0);
        }

        int numImages = images.readInt();
        int numRows = images.readInt();
        int numCols = images.readInt();

        int numImagesRead = 0;
        double[][][] image_t = new double[28][28][numImages];
        while (images.available() > 0 && numImagesRead < numImages) {
            for (int colIdx = 0; colIdx < numCols; colIdx++) {
                for (int rowIdx = 0; rowIdx < numRows; rowIdx++) {
                    image_t[colIdx][rowIdx][numImagesRead] = images.readUnsignedByte();
                    image_t[colIdx][rowIdx][numImagesRead] = image_t[colIdx][rowIdx][numImagesRead]/255;
                }
            }
            numImagesRead++;
        }
        return image_t;
    }

/*    public static void main(String[] args) throws IOException {
        DataInputStream labels = new DataInputStream(new FileInputStream("train-labels-idx1-ubyte"));
        DataInputStream images = new DataInputStream(new FileInputStream("train-images-idx3-ubyte"));
        int magicNumber = labels.readInt();
        if (magicNumber != 2049) {
            System.err.println("Label file has wrong magic number: " + magicNumber + " (should be 2049)");
            System.exit(0);
        }
        magicNumber = images.readInt();
        if (magicNumber != 2051) {
            System.err.println("Image file has wrong magic number: " + magicNumber + " (should be 2051)");
            System.exit(0);
        }
        int numLabels = labels.readInt();
        int numImages = images.readInt();
        int numRows = images.readInt();
        int numCols = images.readInt();
        if (numLabels != numImages) {
            System.err.println("Image file and label file do not contain the same number of entries.");
            System.err.println("  Label file contains: " + numLabels);
            System.err.println("  Image file contains: " + numImages);
            System.exit(0);
        }

        long start = System.currentTimeMillis();
        int numLabelsRead = 0;
        int numImagesRead = 0;
        int[] label_t = new int[numLabels];
        int[][][] image_t = new int[28][28][numImages];
        while (labels.available() > 0 && numLabelsRead < numLabels) {
            byte label = labels.readByte();
            label_t[numLabelsRead] = label;
            numLabelsRead++;
            int[][] image = new int[numCols][numRows];
            for (int colIdx = 0; colIdx < numCols; colIdx++) {
                for (int rowIdx = 0; rowIdx < numRows; rowIdx++) {
                    //image[colIdx][rowIdx] = images.readUnsignedByte();
                    image_t[colIdx][rowIdx][numImagesRead] = images.readUnsignedByte();
                }
            }
            numImagesRead++;


            // At this point, 'label' and 'image' agree and you can do whatever you like with them.

            if (numLabelsRead % 10 == 0) {
                System.out.print(".");
            }
            if ((numLabelsRead % 800) == 0) {
                System.out.print(" " + numLabelsRead + " / " + numLabels);
                long end = System.currentTimeMillis();
                long elapsed = end - start;
                long minutes = elapsed / (1000 * 60);
                long seconds = (elapsed / 1000) - (minutes * 60);
                System.out.println("  " + minutes + " m " + seconds + " s ");
            }
        }
        System.out.println();
        long end = System.currentTimeMillis();
        long elapsed = end - start;
        long minutes = elapsed / (1000 * 60);
        long seconds = (elapsed / 1000) - (minutes * 60);
        System.out.println("Read " + numLabelsRead + " samples in " + minutes + " m " + seconds + " s ");
        for (int i = 0; i <28; i++)
            for (int j = 0; j <28 ; j++) {
                System.out.println(image_t[i][j][1]);
            }

    }*/

}
