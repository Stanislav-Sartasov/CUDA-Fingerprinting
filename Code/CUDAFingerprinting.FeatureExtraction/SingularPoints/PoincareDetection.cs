using System;
using System.Drawing;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.Common.OrientationField;

namespace CUDAFingerprinting.FeatureExtraction.SingularPoints
{

    public class PoincareDetection
    {

        public static double AngleSum(int x, int y, int[,] arr, int fieldSize, double[,] oriented)
        {
            double angleSum = 0;
            int orientationSize = fieldSize*fieldSize - (fieldSize-2)*(fieldSize-2);
            double[] orientationArray = new double[orientationSize];

            
            for (int i = 0; i < orientationSize/4; i++)
            {
                if (x - fieldSize / 2 <= 0 || y + i - fieldSize / 2 <= 0 || y + i - fieldSize / 2 >= arr.GetLength(1))
                    continue;
                orientationArray[i] = oriented[x - fieldSize/2, y + i - fieldSize/2];
            }

            for (int i = orientationSize/4; i < orientationSize/2; i++)
            {
                if (x - fieldSize / 2 + i - orientationSize / 4 <= 0 || x - fieldSize / 2 + i - orientationSize / 4 >= arr.GetLength(0)
                    || y + fieldSize / 2 >= arr.GetLength(1))
                    continue;
                orientationArray[i] = oriented[x - fieldSize/2 + i - orientationSize/4, y + fieldSize/2];
            }

            for (int i = orientationSize/2; i < orientationSize/2 + orientationSize/4; i++)
            {
                if (x + fieldSize / 2 >= arr.GetLength(0) ||
                    y - fieldSize / 2 - (i - 3 * orientationSize / 4) >= arr.GetLength(1) || y - fieldSize / 2 - (i - 3 * orientationSize / 4) <= 0)
                    continue;
                orientationArray[i] = oriented[x + fieldSize/2, y - fieldSize/2 - (i - 3*orientationSize/4)];
            }

            for (int i = orientationSize/4 + orientationSize/2; i < orientationSize; i++)
            {
                if (x - fieldSize / 2 - (i - orientationSize) >= arr.GetLength(0) ||
                    x - fieldSize / 2 - (i - orientationSize) <= 0
                    || y - fieldSize/2 <= 0)
                    continue;
                orientationArray[i] = oriented[x - fieldSize/2 - (i - orientationSize), y - fieldSize/2];
            }

            for (int i = 0; i < orientationSize; i++)
            {
                double angleKandK1;
                if (Math.Abs(-orientationArray[i] + orientationArray[(i + 1) % orientationSize]) <
                   Math.Abs(Math.PI + (-orientationArray[i] + orientationArray[(i + 1) %orientationSize]) ) )
                {
                    angleKandK1 = -orientationArray[i] + orientationArray[(i + 1) %orientationSize];
                }
                else
                {
                    angleKandK1 = Math.PI + (-orientationArray[i] + orientationArray[(i + 1) % orientationSize]);
                }
                angleSum += angleKandK1;
            }
            return angleSum;
        }

        public static Bitmap SingularityDetect(int[,] arr, int blockSize)
        {
            PixelwiseOrientationField img = new PixelwiseOrientationField(arr, 16);
            var oriented = img.Orientation;
            
           for (int i = 1; i < arr.GetLength(0)-2; i += blockSize)
            {
                for (int j = 1; j < arr.GetLength(1)-2; j += blockSize)
                {
                    var k = AngleSum(i, j, arr, blockSize, oriented);
                    if (Math.Abs(k - Math.PI) < 0.001 || Math.Abs(k + Math.PI) < 0.001 || Math.Abs(k - 2 * Math.PI) < 0.001 )
                    {
                        for (int t = i-blockSize/2; t <= i+blockSize/2; t++)
                        {
                            for (int l = j-blockSize/2; l <= j+blockSize/2; l++)
                            {
                                if (i - blockSize/2 < 0 || i + blockSize/2 > arr.GetLength(0) || j - blockSize/2 < 0 ||
                                    j + blockSize / 2 > arr.GetLength(1) || arr.GetLength(0) - t <0)
                                    continue;
                               // detectedImg.SetPixel(t+1, arr.GetLength(1) - t+1, Color.Chartreuse);
                                arr[t, l] = 128;
                            }
                        }
                        
                    }
                }
            }
     
           // var k = AngleSum(40, 40, arr, blockSize, oriented);
            Bitmap detectedImg = ImageHelper.SaveArrayToBitmap(arr);
            return detectedImg;
        }
    }
}
