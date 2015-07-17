import org.math.plot.Plot2DPanel;

import javax.swing.*;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by ziyihua on 08/07/15.
 */
public class Main {
    public static void main(String[] args) throws IOException {


        /*double[][] input = new double[2][2];
        input[0][0] = 1;
        input[0][1] = 1;
        input[1][0] = 2;
        input[1][1] = 0;
        input[0][0][1] = 2;
        input[0][1][1] = 0;
        input[1][0][1] = -1;
        input[1][1][1] = 1;
        input[0][0][2] = 1;
        input[0][1][2] = 2;
        input[1][0][2] = -1;
        input[1][1][2] = 2;

        double[][][] kernel = new double[3][3][3];
        kernel[0][0][0]=1;
        kernel[0][1][0]=2;
        kernel[0][2][0]=3;
        kernel[1][0][0]=0;
        kernel[1][1][0]=1;
        kernel[1][2][0]=1;
        kernel[2][0][0]=-1;
        kernel[2][1][0]=3;
        kernel[2][2][0]=1;

        kernel[0][0][1]=3;
        kernel[0][1][1]=4;
        kernel[0][2][1]=-2;
        kernel[1][0][1]=0;
        kernel[1][1][1]=2;
        kernel[1][2][1]=2;
        kernel[2][0][1]=3;
        kernel[2][1][1]=1;
        kernel[2][2][1]=0;

        kernel[0][0][2]=0;
        kernel[0][1][2]=1;
        kernel[0][2][2]=0;
        kernel[1][0][2]=-1;
        kernel[1][1][2]=0;
        kernel[1][2][2]=3;
        kernel[2][0][2]=-2;
        kernel[2][1][2]=2;
        kernel[2][2][2]=0;

        double[][] output;
        for (int i = 0; i < 3; i++) {
            double[][] input_one = new double[3][3];
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    input_one[j][k] = kernel[j][k][i];
                }
            }
            output = Convolution.convolution2D(input_one,3,3,input,2,2);
            for (int k = 0; k < 2; k++) {
                for (int l = 0; l < 2; l++) {
                        System.out.println(output[k][l]);
                }
            }
        }*/


        /*double[][] output;
        output = Convolution.convolution3D(kernel,3,3,3,input,2,2,3);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                System.out.println(output[i][j]);
            }
        }*/

        //Layers of CNN
        //row 0 is type
        String[][] architecture = new String[3][5];
        architecture[0][0] = "i";
        architecture[0][1] = "c";
        architecture[0][2] = "s";
        architecture[0][3] = "c";
        architecture[0][4] = "s";
        //row 1 is kernelsize for convolutional layer and scale for subsampling layer
        architecture[1][1] = "5";
        architecture[1][2] = "2";
        architecture[1][3] = "5";
        architecture[1][4] = "2";
        //row 2 is number of output maps for convolutional layer
        architecture[2][1] = "6";
        architecture[2][3] = "12";


        /*Structure.network net = new Structure.network();
        Structure.LAYER layer = new Structure.LAYER();
        net.layers.add(0,layer);
        Structure.weights w = new Structure.weights();
        net.layers.get(0).k.add(0,w);
        net.layers.get(0).k.get(0).k_list.add(0,1);
        net.layers.get(0).k.get(0).k_list.add(1,2);
        net.layers.get(0).k.get(0).k_list.add(2,3);
        System.out.println(net.layers.get(0).k.get(0).k_list.get(2));
        ArrayList a = new ArrayList();
        a.add(0,5);
        a.add(1,9);
        a.add(2,4);
        net.layers.get(0).k.get(0).k_list.clear();
        net.layers.get(0).k.get(0).k_list.addAll(a);
        System.out.println(net.layers.get(0).k.get(0).k_list.get(2));*/




        Structure.network convnet = new Structure.network();
        convnet = SetUp.SetUp(architecture);

        int[] label_t;
        label_t = ImportFile.getLabel("train-labels-idx1-ubyte");
        double[][][] image_t;
        image_t = ImportFile.getImage("train-images-idx3-ubyte");


        int[][] label_t_new = new int[10][label_t.length];
        for (int i = 0; i < label_t.length ; i++) {
            for (int j = 0; j < 10; j++) {
                if (label_t[i]==j){
                    label_t_new[j][i]=1;
                }else {
                    label_t_new[j][i]=0;
                }
            }
        }

        double alpha = 1;
        int batchsize = 50;
        int numepochs = 1;

        convnet = CNNtrain.CNNTrain(architecture, convnet, alpha, numepochs, batchsize, label_t_new, image_t);

        double[] loss = new double[convnet.rL.length];
        for (int i = 0; i < convnet.rL.length; i++) {
            loss[i]=convnet.rL[i];
        }
        double[] indx = new double[loss.length];
        for (int i = 0; i < loss.length; i++) {
            indx[i] = i;
        }

        int[] test_y;
        test_y = ImportFile.getLabel("t10k-labels-idx1-ubyte");
        double[][][] test_x;
        test_x = ImportFile.getImage("t10k-images-idx3-ubyte");

        double rate = CNNtest.CNNtest(architecture,convnet,test_x,test_y);
        System.out.println(rate);

        Plot2DPanel plot = new Plot2DPanel();
        plot.addLegend("SOUTH");

        // add a line plot to the PlotPanel
        plot.addLinePlot("Squared Loss", indx, loss);

        // put the PlotPanel in a JFrame like a JPanel
        JFrame frame = new JFrame("a plot panel");
        frame.setSize(600, 600);
        frame.setContentPane(plot);
        frame.setVisible(true);



        /*double[][][] p = (double [][][]) convnet.layers.get(1).d.get(0).d_list.get(5);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                System.out.println(p[i][j][1]);
            }
        }*/
        //System.out.println(convnet.L);

    }
}
