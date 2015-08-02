using System;
using CUDAFingerprinting.Common.OrientationField;

namespace CUDAFingerprinting.ImageProcessing.Binarization
{
    public struct Point
    {
        public int X;
        public int Y;
    }

    public class AdaptiveBinarization
    {
        public const int FieldSizex = 8;

        public static Point Turn(int x, int y, int xCentre, int yCentre, double angleCos, double angleSin)
        {
            Point p;
            p.X = (int) Math.Round(xCentre + (x - xCentre)*angleCos - (y - yCentre)*angleSin);
            p.Y = (int)Math.Round(yCentre + (x - xCentre) * angleSin + (y - yCentre) *angleCos);
            return p;
        }

        public static double find_cos(double ax, double ay, double bx, double by)
        {
            // выражение косинуса из скалярного произведения векторов
            return (ax * bx + ay * by) / (Math.Sqrt(ax * ax + ay * ay) * Math.Sqrt(bx * bx + by * by));
        }
        public static double find_sin(double ax, double ay, double bx, double by)
        {
            // выражения синуса из псевдовекторного произведения векторов
            return (ax * by - ay * bx) / (Math.Sqrt(ax * ax + ay * ay) * Math.Sqrt(bx * bx + by * by));
        }

        public static int[] ProjectionX(int xCentre, int yCentre, int[,] arr) 
        {
            int fieldSizey = FieldSizex/2;
            int[] projX = new int[FieldSizex];
            OrientationField img = new OrientationField(arr);
            var angleOfX = img.GetOrientation(xCentre, yCentre) - Math.PI/2.0;
            Point tmpPoint;
            double angleSin = Math.Sin(angleOfX);
            double angleCos = Math.Cos(angleOfX);
            for (int i = -FieldSizex/2; i < FieldSizex/2; i++)
            {
                projX[i + FieldSizex/2] = 255;
                for (int j = -fieldSizey/2; j < fieldSizey/2; j++) // find the darkest 
                {
                    double localSegment = Math.Sqrt(i*i + j*j);
                    if (Math.Abs(localSegment) > 0.000001)  //  double tolerance
                    {
                        
                        angleSin = Math.Sin(angleOfX + Math.Asin(find_sin(1.0, 0.0, i, j)));
                        angleCos = Math.Cos(angleOfX + Math.Acos(find_cos(1.0, 0.0, i, j)));
                        tmpPoint = Turn(0, (int) Math.Round(localSegment), 0, 0, angleCos, angleSin);
                    }
                    else
                    {
                        tmpPoint.X = 0;
                        tmpPoint.Y = 0;
                    }

                    if (tmpPoint.X + xCentre < 0 || tmpPoint.X + xCentre >= arr.GetLength(0) || tmpPoint.Y + yCentre < 0 ||
                        tmpPoint.Y + yCentre >= arr.GetLength(1))
                    {
                        continue;
                    }
                    
                    if (projX[i + FieldSizex/2] >= arr[tmpPoint.X + xCentre, tmpPoint.Y + yCentre])
                    {
                        projX[i + FieldSizex/2] = arr[tmpPoint.X + xCentre, tmpPoint.Y + yCentre];
                    }

                }
            }
            return projX;   
        }
        public static int[,] AdaptiveBinarize(int[,] img)
        {
            
            int[,] arr = new int[img.GetLength(0), img.GetLength(1)];
            const int projDiviation = 2;        // +-2  pixels from peak
            const int almostMinThreshold = 100;
            for (int xCurrent = 0; xCurrent < img.GetLength(0); xCurrent++)
            {
                for (int yCurrent = 0; yCurrent < img.GetLength(1); yCurrent++)
                {
                    int[] projX = ProjectionX(xCurrent, yCurrent, img);
                    int min = 255;
                    int almostMin = 255;
                    int minIndex = 0;
                    for (int t = 0; t < projX.Length; t++)
                    {
                        if (projX[t] < min)
                        {
                            min = projX[t];
                            minIndex = t;
                        }
                    }
                    for (int t = 0; t < projX.Length; t++)
                    {
                        if (projX[t] < almostMin && projX[t] > min && Math.Abs(minIndex - t) > projDiviation)
                        {
                            almostMin = projX[t];

                        }
                    }
                    
                   arr[xCurrent, yCurrent] = 255;
                   for (int i = FieldSizex / 2 - projDiviation; i <= FieldSizex / 2 + projDiviation; i++)
                    {
                        if (projX[i] == min || (projX[i] == almostMin && almostMin < almostMinThreshold))
                        {
                            arr[xCurrent, yCurrent] = 0;
                            break;
                        }
                    }
                }
            }
            return arr;
        }
    }
}
