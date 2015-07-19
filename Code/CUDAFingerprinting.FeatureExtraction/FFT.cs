using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CUDAFingerprinting.Common;
using System.Numerics;


namespace CUDAFingerprinting.FeatureExtraction
{
    public class Fft
    {
        public const int Size = 32;
        public const int Amplitude = 255;
        public const double k1 = -3, k2 = 2;
        public static Bitmap GenerateSinusoid()
        {
            const int piDiv = 12;
            Bitmap img = new Bitmap(Size, Size);
            
            for (int i = 0; i < Size; i++)
            {
                for (int j = 0; j < Size; j++)
                {
                   // img.SetPixel(i, j, Color.FromArgb((int)(Math.Abs(Math.Sin((i) * Math.PI / piDiv)) * Amplitude), (int)(Math.Abs(Math.Sin((i)* Math.PI / piDiv)) * Amplitude), (int)(Math.Abs(Math.Sin((i) * Math.PI / piDiv)) * Amplitude)));
                   
                    img.SetPixel(i, j, Color.FromArgb( 128+(int) (Math.Cos(2* Math.PI * i* k1 / Size + 2*Math.PI * j * k2 / Size) * 127),
                       128+ (int) (Math.Cos(2* Math.PI * i* k1 / Size + 2*Math.PI * j * k2 / Size) * 127),
                        128+(int) (Math.Cos(2* Math.PI * i* k1 / Size + 2*Math.PI * j * k2 / Size) * 127)));
                }
                
            }
           // img.Save(Path.GetTempPath() + Guid.NewGuid() + ".bmp");
            return img;
        }

        public static double GeneratorCos(int i, int j)
        {
            return (Math.Cos(2*Math.PI*i*k1/Size + 2*Math.PI*j*k2/Size));
        }
/*
        public static Complex[] FftAmplitude(Complex[] xSrc)
        {
            int n = xSrc.Length;
            Complex[] x = new Complex[n];

            if (n == 2)
            {
                x[0] = Complex.Summ(xSrc[0], xSrc[1]); 
                x[1] = Complex.Diff(xSrc[0], xSrc[1]);
            }
            else
            {
                Complex[] xEven = new Complex[n/2];
                Complex[] xOdd = new Complex[n/2];
                for (int i = 0; i < n/2; i++)
                {
                    xEven[i] = xSrc[i*2];
                    xOdd[i] = xSrc[i*2 + 1];
                }
                Complex[] xEvenFft = FftAmplitude(xEven);
                Complex[] xOddFft = FftAmplitude(xOdd);
                for (int i = 0; i < n/2; i++)
                {
                    x[i] = Complex.Summ(xEvenFft[i], Complex.Multiply(Complex.W(i, n), xOddFft[i]));
                    x[i+n/2] = Complex.Diff(xEvenFft[i], Complex.Multiply(Complex.W(i, n), xOddFft[i]));


                }
            }
            return x;

        }
 */

     /*   public static void GradientMagnitudes(Bitmap img)
        {
            var arr = ImageHelper.LoadImageAsInt(img);
            int[,] res = new int[arr.GetLength(0), arr.GetLength(1)];
            int deltaX = 0, deltaY = 0;
            for (int i = 0; i < arr.GetLength(0)-1; i++)
            {
                for (int j = 0; j < arr.GetLength(1)-1; j++)
                {
                    deltaX = arr[i + 1, j] - arr[i, j];
                    deltaY = arr[i, j + 1] - arr[i, j];
                    res[i, j] = (int)Math.Sqrt(deltaX*deltaX + deltaY*deltaY);
                }
            }

            ImageHelper.SaveArrayToBitmap(res).Save(Path.GetTempPath() + Guid.NewGuid() + ".bmp");
        }
      */
        public static double GenerateAmplitudeSpectrum(int x, int y, int[,] arr, int number)
        {
            Complex fur = 0;
            for (int i = -number/2; i < number/2; i++)
            {
                for (int j = -number/2; j < number/2; j++)
                {
                    
                    fur += arr[i + number/2, j + number/2]*Complex.Exp(-Complex.ImaginaryOne* 2 * Math.PI * (i*x + j*y) / number);
                }
            }
            return Complex.Abs(fur);
        }

        public static double FindDominantFrequency(int[,] arr)
        {
            double dominant;
            int currentCircle = 1;
           // int[] numberOfPoints = new int[Math.Min(arr.GetLength(0), arr.GetLength(1))];

            int maxCircle = Math.Min(arr.GetLength(0), arr.GetLength(1))/2;
           // int[] proj = new int[Math.Min(arr.GetLength(0), arr.GetLength(1))];
            double[] normValue = new double[Math.Min(arr.GetLength(0), arr.GetLength(1)) / 2-1]; 
            while (currentCircle < maxCircle)
            {
                int sum = 0;
                int currentNumberOfPoints = 0;
              //  proj[currentCircle - 1] = 0;
                for (int i = -currentCircle; i <= currentCircle; i++)
                {
                    for (int j = (int) Math.Floor(Math.Sqrt(currentCircle*currentCircle - i*i));
                        j <= (int) Math.Ceiling(Math.Sqrt(currentCircle*currentCircle - i*i));
                        j++)
                    {
                        sum += arr[i + currentCircle, j];
                        currentNumberOfPoints++;
                    }
                }
             //   proj[currentCircle - 1] = sum;
              //  numberOfPoints[currentCircle - 1] = currentNumberOfPoints;
                normValue[currentCircle - 1] = (double)sum/currentNumberOfPoints;
                currentCircle++;
            }
            double min = 255;
            int minIndex = 0;
            for (int i = 0; i < normValue.Length; i++)
            {
                if (min > normValue[i])
                {
                    min = normValue[i];
                    minIndex = i;
                }
            }
            
            return min;
        }
    }
}
