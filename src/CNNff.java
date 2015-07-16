/**
 * Created by ziyihua on 15/07/15.
 */
public class CNNff extends Structure{
    public CNNff(){
    }

    public static network CNNff(String[][] architecture, network convnet, double[][][] batch_x){
        int n = convnet.layers.size();
        A Ai = new A();
        if(convnet.layers.get(0).a.isEmpty()){
            convnet.layers.get(0).a.add(0,Ai);
        }else convnet.layers.get(0).a.set(0,Ai);
        convnet.layers.get(0).a.get(0).a_list.add(0, batch_x);
        int inputmaps = 1;


        for (int i = 1; i < n; i++) {
            if ("c".equals(architecture[0][i])){
                A Ac = new A();
                if (convnet.layers.get(i).a.isEmpty()) {
                    convnet.layers.get(i).a.add(0, Ac);
                }else convnet.layers.get(i).a.set(0,Ac);

                LAYER layer_previous = convnet.layers.get(i-1);
                LAYER layer_current = convnet.layers.get(i);

                for (int j = 0; j < layer_current.outmaps; j++) {
                    //temporary output
                    int a = ((double[][][]) (layer_previous.a.get(0).a_list).get(0)).length;
                    int b = ((double[][][]) (layer_previous.a.get(0).a_list).get(0))[0].length;
                    int c = ((double[][][]) (layer_previous.a.get(0).a_list).get(0))[0][0].length;
                    int a_new = a-layer_current.kernelsize+1;
                    int b_new = b-layer_current.kernelsize+1;
                    int c_new = c;
                    double[][][] z = new double[a_new][b_new][c_new];
                    for (int k = 0; k < a_new; k++) {
                        for (int l = 0; l < b_new; l++) {
                            for (int m = 0; m < c_new; m++) {
                                z[k][l][m]=0;
                            }
                        }
                    }

                    //convolution
                    for (int k = 0; k < inputmaps; k++) {
                        for (int l = 0; l < c_new; l++) {
                            double[][] x_one = new double[a][b];
                            for (int m = 0; m < a; m++) {
                                for (int o = 0; o < b ; o++) {
                                    x_one[m][o] = ((double[][][]) layer_previous.a.get(0).a_list.get(k))[m][o][l];
                                }
                            }
                            double[][] z_conv = Convolution.convolution2D(x_one,a,b,(double[][]) layer_current.k.get(0).k_list.get(j+k*layer_current.outmaps),layer_current.kernelsize,layer_current.kernelsize);
                            for (int m = 0; m < a_new; m++) {
                                for (int o = 0; o < b_new; o++) {
                                    z[m][o][l]=z[m][o][l]+z_conv[m][o];
                                }
                            }
                        }

                    }
                    double[][][] m = new double[a_new][b_new][c_new];
                    for (int k = 0; k < a_new; k++) {
                        for (int l = 0; l < b_new; l++) {
                            for (int o = 0; o < c_new; o++) {
                                m[k][l][o]=1/(1+Math.exp(-z[k][l][o]-layer_current.b[j]));
                            }
                        }
                    }
                    layer_current.a.get(0).a_list.add(j, m);
                }
                inputmaps=layer_current.outmaps;
            }
            else if ("s".equals(architecture[0][i])){
                A As = new A();
                if (convnet.layers.get(i).a.isEmpty()) {
                    convnet.layers.get(i).a.add(0, As);
                }else convnet.layers.get(i).a.set(0,As);
                LAYER layer_previous = convnet.layers.get(i-1);
                LAYER layer_current = convnet.layers.get(i);
                int a = ((double[][][]) (layer_previous.a.get(0).a_list).get(0)).length;
                int b = ((double[][][]) (layer_previous.a.get(0).a_list).get(0))[0].length;
                int c = ((double[][][]) (layer_previous.a.get(0).a_list).get(0))[0][0].length;
                int a_new = a-layer_current.scale+1;
                int b_new = b-layer_current.scale+1;
                double[][] subsample = new double[layer_current.scale][layer_current.scale];
                for (int j = 0; j < layer_current.scale; j++) {
                    for (int k = 0; k < layer_current.scale; k++) {
                        subsample[j][k]=(double) 1/(layer_current.scale*layer_current.scale);
                    }
                }
                for (int j = 0; j < inputmaps; j++) {
                    double[][][] z = new double[a_new][b_new][c];
                    for (int k = 0; k < c; k++) {
                        double[][] a_one = new double[a][b];
                        for (int m = 0; m < a; m++) {
                            for (int o = 0; o < b ; o++) {
                                a_one[m][o] = ((double[][][]) layer_previous.a.get(0).a_list.get(j))[m][o][k];
                            }
                        }
                        double[][] a_conv=Convolution.convolution2D(a_one,a,b,subsample,layer_current.scale,layer_current.scale);
                        for (int l = 0; l < a_new; l++) {
                            for (int m = 0; m < b_new; m++) {
                                z[l][m][k] = a_conv[l][m];
                            }
                        }
                    }
                    double[][][] m = new double[(a_new+1)/2][(b_new+1)/2][c];
                    for (int k = 0; k < (a_new+1)/2; k++) {
                        for (int l = 0; l < (b_new+1)/2; l++) {
                            for (int o = 0; o < c; o++) {
                                m[k][l][o]=z[k*2][l*2][o];
                            }
                        }
                    }
                    layer_current.a.get(0).a_list.add(j, m);
                }
            }
        }

        /**
         * concatenate all end layer feature maps into vector
         */
        LAYER layer_current = convnet.layers.get(n-1);
        int a = ((double[][][]) layer_current.a.get(0).a_list.get(0)).length;
        int b = ((double[][][]) layer_current.a.get(0).a_list.get(0))[0].length;
        int c = ((double[][][]) layer_current.a.get(0).a_list.get(0))[0][0].length;
        int outputmaps = layer_current.a.get(0).a_list.size();
        double[][] fv= new double[a*b*outputmaps][c];
        for (int i = 0; i < c; i++) {
            int row = 0;
            for (int j = 0; j < outputmaps; j++) {
                for (int k = 0; k < b; k++) {
                    for (int l = 0; l < a; l++) {
                        fv[row][i]=((double[][][])layer_current.a.get(0).a_list.get(j))[l][k][i];
                        row++;
                    }
                }
            }
        }
        convnet.fv=fv;



        int d = convnet.ffW.length;
        int e = convnet.fv.length;
        double[][] product = new double[d][c];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < c; j++) {
                product[i][j]=0;
            }
        }
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < c; j++) {
                for (int k = 0; k < e; k++) {
                    product[i][j]=product[i][j]+convnet.ffW[i][k]*convnet.fv[k][j];
                }
            }
        }

        double[][] o = new double[d][c];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < c; j++) {
                o[i][j]=1/(1+Math.exp(-product[i][j]-convnet.ffb[i]));
            }
        }
        convnet.o=o;
        return convnet;
    }
}
