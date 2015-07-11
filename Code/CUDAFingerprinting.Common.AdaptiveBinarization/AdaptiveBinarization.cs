using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Mime;
using System.Text;
using System.Threading.Tasks;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.Common.AdaptiveBinarization
{
    public struct Point
    {
        public int X;
        public int Y;
    }

    /*public struct Square
    {
        public Point Point0;
        public Point Point1;
        public Point Point2;
        public Point Point3;
    }*/

    public class AdaptiveBinarization
    {
        public int Width;
        public int Height;
        public float[,] ImageArray;
        public const int FieldSize = 16;
        public AdaptiveBinarization(float[,] array)
        {
            Width = array.GetLength(0);
            Height = array.GetLength(1);
            ImageArray = array;
        }

        public static Point Turn(int x, int y, int xCentre, int yCentre, double angleCos, double angleSin)
        {
            Point p;
            
            p.X = (int)Math.Round(xCentre + (x - xCentre)*angleCos 
                - (y - yCentre)*angleSin);
            
            p.Y = (int)Math.Round(yCentre + (x - xCentre) * angleSin 
                + (y - yCentre) *angleCos);

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

        public static double[] ProjectionX(int xCentre, int yCenre, int xCurrentPoint, int yCurrentPoint, int[,] arr) //  orthogonal projection on Ox
        {
            // просуммировать по игреку все в этом ряду из блока 16х16 ортоганально оси Ох
            double[] projX = new double[FieldSize]; //16
            OrientationField.OrientationField img = new OrientationField.OrientationField(arr); // size of bmp
            var angleOfY = img.GetOrientation(xCentre, yCenre);
            var angleOfX = angleOfY - Math.PI/2.0;

           // Square orientWindow;
           // orientWindow.Point0.X = 
            double angleSin, angleCos;
            double localSegment;
            Point tmpPoint;
            for (int i = -FieldSize/2; i < FieldSize/2 - 1; i++)
            {
                projX[i] = 0;
                for (int j = -FieldSize / 2; j < FieldSize/2 - 1; j++) //summ all column elements
                {
                    localSegment = Math.Sqrt(i*i + j*j);
                    angleSin = Math.Sin(angleOfX) + find_sin(1.0, 0.0, i, j);
                    angleCos = Math.Cos(angleOfX) + find_cos(1.0, 0.0, i, j);
                    tmpPoint = Turn(0, (int) Math.Round(localSegment), xCentre, yCenre, angleCos, angleSin);
                    projX[i] += arr[tmpPoint.X + xCentre, tmpPoint.Y + yCenre];
                    
                }
                
            }
            return projX;
        }

    }
}
