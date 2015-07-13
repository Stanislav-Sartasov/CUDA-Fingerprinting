using System;
using System.Drawing;
using System.IO;
using System.Net;
using CUDAFingerprinting.Common.OrientationField;

namespace CUDAFingerprinting.Common.AdaptiveBinarization
{
    public struct Point
    {
        public int X;
        public int Y;
    }

    public class AdaptiveBinarization
    {
        public const int FieldSizex = 16;

        public static Point Turn(int x, int y, int xCentre, int yCentre, double angleCos, double angleSin)
        {
            Point p;
            p.X = (int) Math.Floor(xCentre + (x - xCentre)*angleCos - (y - yCentre)*angleSin);
            p.Y = (int)Math.Floor(yCentre + (x - xCentre) * angleSin + (y - yCentre) *angleCos);
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

        public static int[] ProjectionX(int xCentre, int yCenre, int[,] arr) 
        {
            int fieldSizey = FieldSizex / 2;
            int[] projX = new int[FieldSizex];
            OrientationField.OrientationField img = new OrientationField.OrientationField(arr);
            var angleOfX = img.GetOrientation(xCentre, yCenre) - Math.PI/2.0;
            Point tmpPoint;
            double angleSin;
            double angleCos;
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
                        tmpPoint = Turn(0, (int) Math.Floor(localSegment), 0, 0, angleCos, angleSin);
                    }
                    else
                    {
                        tmpPoint.X = 0;
                        tmpPoint.Y = 0;
                    }

                    if (tmpPoint.X + xCentre < 0 || tmpPoint.X + xCentre >= arr.GetLength(0) || tmpPoint.Y + yCenre < 0 ||
                        tmpPoint.Y + yCenre >= arr.GetLength(1))
                    {
                        continue;
                    }
                    
                    if (projX[i + FieldSizex/2] >= arr[tmpPoint.X + xCentre, tmpPoint.Y + yCenre])
                    {
                        projX[i + FieldSizex/2] = arr[tmpPoint.X + xCentre, tmpPoint.Y + yCenre];
                    }

                }
            }
            return projX;   
        }

       /* public static int[,] AdaptiveImageBinarization(int[,] arr)              // неправильная бинаризация
        {
            int[,] resArr = new int[arr.GetLength(0),arr.GetLength(1)]; // values 0 or 255
            
            for (int xCentre = 0; xCentre < arr.GetLength(0); xCentre++) //смени границы
            {
                for (int yCente = 0; yCente < arr.GetLength(1); yCente++)
                {
                    var projX = ProjectionX(xCentre, yCente, arr);
                    double max = 0;
                    for (int k = -FieldSizex / 2; k < FieldSizex / 2 ; k++)
                    {
                        if (k == 0)
                        {
                            continue;
                        }
                        if (max < projX[k+FieldSizex/2])
                        {
                            max = projX[k+FieldSizex/2];
                        }
                    }

                    if (projX[FieldSizex / 2] >= max)
                    {
                        resArr[xCentre, yCente] = 255;
                    }
                    else
                    {
                        resArr[xCentre, yCente] = 0;
                    }
                }
                
            }
            return resArr;
        }*/

    }
}
