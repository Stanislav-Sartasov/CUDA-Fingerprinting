using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OrientationImg_Test
{
 class Program
  {
   static void Main(string[] args)
    {
     const int width = 7;
     const int height = 7;
     OrientationImage Obj = new OrientationImage (height, width);
     float [] O = new float [height*width] {1.5f,    0,     2,  1,-1, -1,    2,
                                               3,-1.5f,     2,  0, 2,  2,    1,
                                               2, 0.5f, -1.5f,  0, 1,  1, 1.5f,
                                               2,     1,    0, -1, 0, -2, 0.5f,
                                               3,     0,   -2,  1,-1,  0,    1,
                                            -1.5f,    0,   -2,  2, 0, -2, 1.5f,
                                             1.5f, -0.5f,   0,  4, 1,  3,-1.5f };
    Console.WriteLine("Array:");
    for (int i = 0; i < height; i++)
     {
      for (int j = 0; j < width; j++)
       {
        Console.Write("{0}  ", O[i*width + j]);
       }
      Console.Write("\n");
     }
    Console.Write("\n\n\n"); 
    float [] Fx = Obj.VectorFieldX(O);
    float [] Fy = Obj.VectorFieldY(O);
    float [] Fx1 = Obj.Filter(Fx);
    float [] Fy1 = Obj.Filter(Fy);
    Console.Write("\n\nFx1:\n");
    for (int i = 0; i < height; i++)
     {
      for (int j = 0; j < width; j++) Console.Write("{0} ", Fx1[i*width + j]);
      Console.Write("\n");
      }
     Console.Write("\n\nFy1:\n");
     for (int i = 0; i < height; i++)
      {
       for (int j = 0; j < width; j++) Console.Write("{0} ", Fy1[i*width + j]);
       Console.Write("\n");
      }
    Obj.LocalOrientation(ref O, Fx1, Fy1);
    Console.Write("\n\nO:\n");
    for (int i = 0; i < height; i++)
     {
      for (int j = 0; j < width; j++) Console.Write("{0} ", O[i*width + j]);
      Console.Write("\n");
     }
    Console.ReadKey();
   }
 }
  class OrientationImage
{
   private int height, width;
   public OrientationImage(int h, int w)
    {
      height = h;
      width = w;
     }
  public float [] VectorFieldX(float [] O)
   {
     float [] F = new float[height * width];
     for (int i = 0; i < height; i++)
     {
       for (int j = 0; j < width; j++)
       {
         F[i * width + j] =(float) (Math.Cos(2 * O[i * width + j]));
       }
     }
     return F;
   }
   public float [] VectorFieldY(float [] O)
   {
     float[] F = new float[height * width];
     for (int i = 0; i < height; i++)
     {
       for (int j = 0; j < width; j++)
       {
         F[i * width + j] = (float)(Math.Sin(2 * O[i * width + j]));
       }
     }
     return F;
   }
   public void LocalOrientation(ref float [] O, float [] Fx1, float [] Fy1)
   {
    int ind;
    for (int i = 0; i < height; i++)
     {
      for (int j = 0; j < width; j++)
      {
       ind = i * width + j;
       O[ind] = (float)(0.5 * Math.Atan (Fy1[ind] /  Fx1[ind]));
      }
     }
    }

   public float [] Filter(float [] F)
    {
     float [] F1 =new float [height * width];
     for (int i = 0; i < height; i++)
      {
       for (int j = 0; j < width; j++)
        {
         F1[i * width + j] = Filter(i, j, F);
         }
      }
     return F1;
     }

    public float Filter(int i, int j, float [] F)
     {
      int w = 1;
      const int wf = 3;
      float temp = 0;
      float [] W = new float [wf*wf] { 0.5f, 0.3f, 0.5f, 0.3f, 1, 0.3f, 0.5f, 0.3f, 0.5f };
      for (int u = -wf / 2; u <= wf / 2; u++)
       {
        for (int v = -wf / 2; v <= wf / 2; v++)
         {
          if (((i - u*w) >= 0) & ((i - u*w) < height) & ((j - v*w) >= 0) & ((j - v*w) < width))
          temp += W[(u + wf / 2)*wf + v + wf / 2] * F[(i - u*w)*width + j - v*w];
         }
        }
      return temp;
     }
   }
}
