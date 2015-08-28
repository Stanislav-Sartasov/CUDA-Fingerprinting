using System;
using System.IO;
namespace CUDAFingerprinting.Common.OrientationField
{
  public class OrientationFieldRegularization
  {
    private int height, width;
    private double[] O;
    private int sizeFil;

    // конструктор
    public OrientationFieldRegularization(double[,] O_2D, int sizeF)
    {
      height = O_2D.GetLength(0);
      width = O_2D.GetLength(1);
      O = new double[height * width];
      for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++) O[i * width + j] = O_2D[i, j];
      sizeFil = sizeF;
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
      double[] Fx = FilterGaussian(VectorFieldX());
      double[] Fy = FilterGaussian(VectorFieldY());
      int ind;
      for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
        {
          ind = i * width + j;
          O[ind] = 0.5 * Math.Atan(Fy[ind] / Fx[ind]);
          if (Fx[ind] <= 0 && Fy[ind] >= 0) O[ind] += Math.PI / 2;
          else if (Fx[ind] <= 0 && Fy[ind] <= 0) O[ind] -= Math.PI / 2;
        }
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
    public double [] FilterGaussian(double [] F) 
    {
      double[,] F_2D = F.Make2D(height, width);
      Filter f_gaus = new Filter(sizeFil, (sizeFil-1)/6);
      F = (ConvolutionHelper.Convolve(F_2D, f_gaus.Matrix, 1)).Make1D();
      return F;
    }
  }
}