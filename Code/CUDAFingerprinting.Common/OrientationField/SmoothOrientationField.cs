using System;
using System.IO;
namespace CUDAFingerprinting.Common.OrientationField
{
    public class SmoothOrientationField
    {
        private int height, width;
        private double[] O;

        // конструктор
        public SmoothOrientationField(double[,] O_2D)
        {
            height = O_2D.GetLength(0);
            width = O_2D.GetLength(1);
            O = new double[height * width];
            Console.WriteLine(height + " " + width);
             System.Threading.Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("en-US");
            FileStream file = new FileStream("D:\\inputGaus.txt", FileMode.Create, FileAccess.ReadWrite);
            StreamWriter writer = new StreamWriter(file);
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    O[i * width + j] = O_2D[i, j];
                    writer.Write(O[i * width + j] + " ");
                    //  Console.WriteLine(O[i * width + j] + " ");
                }
                //writer.WriteLine();
            }
            writer.Close();
            file.Close();
        }

        public double[] VectorFieldX()
        {
            double[] F = new double[height * width];
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++) F[i * width + j] = Math.Cos(2 * O[i * width + j]);
            return F;
        }

        public double[] VectorFieldY()
        {
            double[] F = new double[height * width];
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++) F[i * width + j] = Math.Sin(2 * O[i * width + j]);
            return F;
        }
        public double[,] LocalOrientation()
        {
            double[] Fx = VectorFieldX();
            double[] Fy = VectorFieldY();
            double[] Fx1 = Filterx(Fx);
            double[] Fy1 = Filtery(Fy);
            int ind;
            FileStream file = new FileStream("D:\\outGausCs.txt", FileMode.Create, FileAccess.ReadWrite);
            StreamWriter writer = new StreamWriter(file);
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    ind = i * width + j;
                    if (Fx1[ind] <= 0 && Fy1[ind] >= 0) O[ind] = 0.5 * Math.Atan(Fy1[ind] / Fx1[ind]) + Math.PI / 2;
                    else if (Fx1[ind] <= 0 && Fy1[ind] <= 0) O[ind] = 0.5 * Math.Atan(Fy1[ind] / Fx1[ind]) - Math.PI / 2;
                    else O[ind] = 0.5 * Math.Atan(Fy1[ind] / Fx1[ind]);
                    writer.Write(O[ind] + " ");
                }
            }
            writer.Close();
            file.Close();

            return LocalOrientation_2D(O);
        }

        public double[,] LocalOrientation_2D(double[] O)
        {
            int ind;
            double[,] O_2D = new double[height, width];
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                {
                    ind = i * width + j;
                    O_2D[i, j] = O[ind];
                }
            return O_2D;
        }

        public double[] Filterx(double[] Fx)
        {
            double[] F1 = new double[height * width];
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++) F1[i * width + j] = FilterX(i, j, Fx);
            return F1;
        }
        public double[] Filtery(double[] Fy)
        {
            double[] F1 = new double[height * width];
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++) F1[i * width + j] = FilterY(i, j, Fy);
            return F1;
        }

        public double FilterX(int i, int j, double[] F)
        {
            int w = 1;
            const int wf = 3;
            double temp = 0;
            //double[] W = new double[wf * wf] { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
            double[] W = new double[wf * wf] { 0, 0, 0, 0, 1, 0, 0, 0, 0 };
            //double[] W = new double[wf * wf] { 0.1111111, 0.1111111, 0.1111111, 0.1111111, 0.1111111, 0.1111111, 0.1111111, 0.1111111, 0.1111111 };
            /*double[] W = new double[wf * wf];
            for (int ind = 0; ind < wf * wf; ind++)
                W[ind] = (double)1 / (wf * wf);*/
            int halfWf = ((wf % 2) == 0 ? wf / 2 - 1 : wf / 2);
            for (int u = -wf / 2; u <= halfWf; u++)
                for (int v = -wf / 2; v <= halfWf; v++)
                    if (((i - u * w) >= 0) & ((i - u * w) < height) & ((j - v * w) >= 0) & ((j - v * w) < width))
                        temp += W[(u + wf / 2) * wf + v + wf / 2] * F[(i - u * w) * width + j - v * w];

            return temp;
        }
        public double FilterY(int i, int j, double[] F)
        {
            int w = 1;
            const int wf = 3;
            double temp = 0;
            //double[] W = new double[wf * wf] { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
            double[] W = new double[wf * wf] { 0, 0, 0, 0, 1, 0, 0, 0, 0 };
            //double[] W = new double[wf * wf] { 0.1111111, 0.1111111, 0.1111111, 0.1111111, 0.1111111, 0.1111111, 0.1111111, 0.1111111, 0.1111111 };
            /*double[] W = new double[wf * wf];
            double sigma = (double)2.5;
            for (int i1 = wf - 1; i1 > -1; i1--)
                for (int j1 = wf - 1; j1 > -1; j1--)
                    W[i1 * wf + j1] = Gaussian2D(i1, j1, sigma);*/
            int halfWf = ((wf % 2) == 0 ? wf / 2 - 1 : wf / 2);
            for (int u = -halfWf; u <= wf / 2; u++)
                for (int v = -halfWf; v <= wf / 2; v++)

                    if (((i - u * w) >= 0) & ((i - u * w) < height) & ((j - v * w) >= 0) & ((j - v * w) < width))
                        temp += W[(u + halfWf) * wf + v + halfWf] * F[(i - u * w) * width + j - v * w];

            return temp;
        }
        public static double Gaussian2D(double x, double y, double sigma)
        {
            var commonDenom = 2.0d * sigma * sigma;
            var denominator = Math.PI * commonDenom;
            var result = Math.Exp(-(x * x + y * y) / commonDenom) / denominator;
            return result;
        }
    }
}
