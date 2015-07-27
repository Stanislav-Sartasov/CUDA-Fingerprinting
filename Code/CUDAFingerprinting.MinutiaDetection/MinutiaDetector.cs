using System;
using CUDAFingerprinting.Common;
using System.Collections.Generic;

namespace CUDAFingerprinting.MinutiaDetection
{
    public static class MinutiaDetector
    {
        private const int BLACK = 0;
        private const int GREY = 128;
        private const int WHITE = 255;

        private static double GetPixel(int[,] data, int x, int y)
        {
            return (x < 0 || y < 0 || x >= data.GetLength(1) || y >= data.GetLength(0)) ?
                WHITE :
                data[data.GetLength(0) - 1 - y, x] > GREY ?
                    WHITE :
                    BLACK;
        }

        //contains 'minutia code' of current pixel
        private static int NeigboursCount;

        private static bool IsMinutia(int[,] data, int x, int y)
        {
            if (GetPixel(data, x, y) != BLACK)
                return false;
            //check 8-neigbourhood
            bool[] p = {
                            GetPixel(data, x, y - 1) > 0,
                            GetPixel(data, x + 1, y - 1) > 0,
                            GetPixel(data, x + 1, y) > 0,
                            GetPixel(data, x + 1, y + 1) > 0,
                            GetPixel(data, x, y + 1) > 0,
                            GetPixel(data, x - 1, y + 1) > 0,
                            GetPixel(data, x - 1, y) > 0,
                            GetPixel(data, x - 1, y - 1) > 0,
                        };

            NeigboursCount = 0;
            for (int i = 1; i < 9; i++)
            {
                NeigboursCount += p[i % 8] ^ p[i - 1] ? 1 : 0;
            }
            NeigboursCount /= 2;
            //count == 0 <=> isolated point - NOT minutia
            //count == 1 <=> 'end line' - minutia
            //count == 2 <=> part of the line - NOT minutia
            //count == 3 <=> 'fork' - minutia
            //count >= 3 <=> composit minutia - ignoring in this implementation
            return ((NeigboursCount == 1) || (NeigboursCount == 3));
        }

        private static bool InCircle(int xC, int yC, int R, int x, int y)
        {
            return Math.Pow(xC - x, 2) + Math.Pow(yC - y, 2) < R * R;
        }

        //rotate on PI angle if not right direction
        private static double GetCorrectAngle(int[,] data, PixelwiseOrientationField oField, int x, int y)
        {
            double angle = oField.GetOrientation(data.GetLength(0) - 1 - y, x);
            float PI = 3.141592654f;
            //for 'end line' minutia
            if (NeigboursCount == 1)
            {
                if (angle > 0.0)
                {
                    if ((GetPixel(data, x, y - 1) + 
                        GetPixel(data, x + 1, y - 1) + 
                        GetPixel(data, x + 1, y))
                        <
                        (GetPixel(data, x, y + 1) + 
                        GetPixel(data, x - 1, y + 1) + 
                        GetPixel(data, x - 1, y)))
                    {
                        angle += PI;
                    }
                }
                else
                {
                    if ((GetPixel(data, x, y + 1) + 
                        GetPixel(data, x + 1, y + 1) + 
                        GetPixel(data, x + 1, y))
                        <
                        (GetPixel(data, x, y - 1) + 
                        GetPixel(data, x - 1, y - 1) + 
                        GetPixel(data, x - 1, y)))
                    {
                        angle += PI;
                    }
                }
            }
            //for 'fork' minutia
            else if (NeigboursCount == 3)
            {
                for (int r = 1; r < 16; r++)
                {
                    double normal = angle + PI / 2;
                    int aboveNormal = 0;
                    int belowNormal = 0;

                    for (int i = -r; i <= r; i++)
                    {
                        for (int j = -r; j <= r; j++)
                        {
                            if (i == j && j == 0)
                            {
                                continue;
                            }
                            if (GetPixel(data, x + j, y + i) == BLACK && 
                                InCircle(x, y, r, x + j, y + i))
                            {
                                double deltaNormalY = - Math.Tan(normal) * j;
                                if (i < deltaNormalY)
                                {
                                    aboveNormal++;
                                }
                                else
                                {
                                    belowNormal++;
                                }
                            }
                        }
                    }
                    if (aboveNormal == belowNormal)
                    {
                        continue;//?
                    }
                    else
                    {
                        if ((aboveNormal > belowNormal &&
                            Math.Tan(angle) > 0.0) ||
                            (aboveNormal < belowNormal &&
                            Math.Tan(angle) < 0.0))
                        {
                            angle += PI;
                        }
                        break;
                    }
                }
            }
            return angle;
        }

        public static List<Minutia> GetMinutias(int[,] data, PixelwiseOrientationField oField)
        {
            int width = data.GetLength(1);
            int height = data.GetLength(0);
            List<Minutia> minutias = new List<Minutia>();
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (IsMinutia(data, x, y))
                    {
                        Minutia m = new Minutia();
                        m.X = x;
                        m.Y = y;
                        m.Angle = (float) GetCorrectAngle(
                            data,
                            oField,
                            x,
                            y
                        );
                        minutias.Add(m);
                    }
                }
            }
            return minutias;
        }
    }
}
