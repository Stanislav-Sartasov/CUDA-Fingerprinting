using System;
using System.IO;
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

        public static int[] ProjectionX(int xCentre, int yCenre, int[,] arr) //  orthogonal projection on Ox
        {
            int FieldSizey = 8;
            // просуммировать по игреку все в этом ряду из блока 16х16 ортоганально оси Ох
            int[] projX = new int[FieldSizex]; //16
            OrientationField.OrientationField img = new OrientationField.OrientationField(arr); // size of bmp
            


            var angleOfX = img.GetOrientation(xCentre, yCenre) - Math.PI/2.0;

            double angleSin, angleCos;
           // double localSegment;
            Point tmpPoint;
            angleSin = Math.Sin(angleOfX);
            angleCos = Math.Cos(angleOfX);
            for (int i = -FieldSizex/2; i < FieldSizex/2; i++)
            {
                projX[i + FieldSizex/2] = 0;
                for (int j = -FieldSizey/2; j < FieldSizey/2; j++) //summ all column elements
                {
                   // localSegment = Math.Sqrt(i*i + j*j);
                    
                    if (i == 0 && j == 0) continue; // double tolerance
                    
                    //angleSin = Math.Sin(angleOfX + Math.Asin(find_sin(1.0, 0.0, i, j)));
                    //angleCos = Math.Cos(angleOfX + Math.Acos(find_cos(1.0, 0.0, i, j)));
                    //tmpPoint = Turn(0, (int) Math.Floor(localSegment), 0, 0, angleCos, angleSin);
                    tmpPoint = Turn(i, j, 0, 0, angleCos, angleSin);
                    if (tmpPoint.X + xCentre < 0 || tmpPoint.X + xCentre >= arr.GetLength(0) || tmpPoint.Y + yCenre < 0 ||
                        tmpPoint.Y + yCenre >= arr.GetLength(1))
                    {
                        FieldSizey--;
                        continue;
                    }
                    projX[i + FieldSizex/2] += arr[tmpPoint.X + xCentre, tmpPoint.Y + yCenre];
                }
                if (FieldSizey <= 0)
                {
                    continue;
                }
                projX[i + FieldSizex/2] /= FieldSizey;
            }
            return projX;
        }

        public static int[,] AdaptiveImageBinarization(int[,] arr)
        {
            int[,] resArr = new int[arr.GetLength(0),arr.GetLength(1)]; // values 0 or 255
            
            for (int xCentre = FieldSizex; xCentre < arr.GetLength(0)-FieldSizex; xCentre++) //смени границы
            {
                for (int yCente = FieldSizex; yCente < arr.GetLength(1)-FieldSizex; yCente++)
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
        }
            
        

    }
}
