using System;

namespace CUDAFingerprinting.FeatureExtraction
{
    public static class Thinner
    {
        private const double BLACK = 0.0;
        private const double WHITE = 255.0;

        private enum PixelType
        {
            FILLED,
            EMPTY,
            ANY,
            CENTER,
            AT_LEAST_ONE_EMPTY
        }

        private static PixelType[, ,] patterns = new PixelType[,,] 
        {
            {//a
                {PixelType.FILLED, PixelType.FILLED, PixelType.AT_LEAST_ONE_EMPTY}, 
                {PixelType.FILLED, PixelType.CENTER, PixelType.EMPTY}, 
                {PixelType.FILLED, PixelType.FILLED, PixelType.AT_LEAST_ONE_EMPTY}
            },
            {//b
                {PixelType.FILLED,             PixelType.FILLED, PixelType.FILLED}, 
                {PixelType.FILLED,             PixelType.CENTER, PixelType.FILLED}, 
                {PixelType.AT_LEAST_ONE_EMPTY, PixelType.EMPTY,  PixelType.AT_LEAST_ONE_EMPTY}
            },
            {//c - needs special processing
                {PixelType.AT_LEAST_ONE_EMPTY, PixelType.FILLED, PixelType.FILLED}, 
                {PixelType.EMPTY,              PixelType.CENTER, PixelType.FILLED},//PixelType.FILLED 
                {PixelType.AT_LEAST_ONE_EMPTY, PixelType.FILLED, PixelType.FILLED}
            },
            {//d - needs special processing
                {PixelType.AT_LEAST_ONE_EMPTY, PixelType.EMPTY,  PixelType.AT_LEAST_ONE_EMPTY}, 
                {PixelType.FILLED,             PixelType.CENTER, PixelType.FILLED}, 
                {PixelType.FILLED,             PixelType.FILLED, PixelType.FILLED}
                                             //PixelType.FILLED
            },
            {//e
                {PixelType.ANY,    PixelType.EMPTY,  PixelType.EMPTY}, 
                {PixelType.FILLED, PixelType.CENTER, PixelType.EMPTY}, 
                {PixelType.ANY,    PixelType.FILLED, PixelType.ANY}
            },
            {//f
                {PixelType.ANY,   PixelType.FILLED, PixelType.FILLED}, 
                {PixelType.EMPTY, PixelType.CENTER, PixelType.FILLED}, 
                {PixelType.EMPTY, PixelType.EMPTY,  PixelType.ANY}
            },
            {//g
                {PixelType.EMPTY, PixelType.FILLED, PixelType.EMPTY}, 
                {PixelType.EMPTY, PixelType.CENTER, PixelType.FILLED}, 
                {PixelType.EMPTY, PixelType.EMPTY,  PixelType.EMPTY}
            },
            {//h
                {PixelType.ANY,    PixelType.FILLED, PixelType.ANY}, 
                {PixelType.FILLED, PixelType.CENTER, PixelType.EMPTY}, 
                {PixelType.ANY,    PixelType.EMPTY,  PixelType.EMPTY}
            },
            {//i
                {PixelType.EMPTY, PixelType.EMPTY,  PixelType.ANY}, 
                {PixelType.EMPTY, PixelType.CENTER, PixelType.FILLED}, 
                {PixelType.ANY,   PixelType.FILLED, PixelType.FILLED}
            },
            {//j
                {PixelType.EMPTY, PixelType.EMPTY,  PixelType.EMPTY}, 
                {PixelType.EMPTY, PixelType.CENTER, PixelType.FILLED}, 
                {PixelType.EMPTY, PixelType.FILLED, PixelType.EMPTY}
            },
            {//k
                {PixelType.EMPTY,  PixelType.EMPTY,  PixelType.EMPTY}, 
                {PixelType.EMPTY,  PixelType.CENTER, PixelType.EMPTY}, 
                {PixelType.FILLED, PixelType.FILLED, PixelType.FILLED}
            },
            {//l
                {PixelType.FILLED, PixelType.EMPTY,  PixelType.EMPTY}, 
                {PixelType.FILLED, PixelType.CENTER, PixelType.EMPTY}, 
                {PixelType.FILLED, PixelType.EMPTY,  PixelType.EMPTY}
            },
            {//m
                {PixelType.FILLED, PixelType.FILLED, PixelType.FILLED}, 
                {PixelType.EMPTY,  PixelType.CENTER, PixelType.EMPTY}, 
                {PixelType.EMPTY,  PixelType.EMPTY,  PixelType.EMPTY}
            },
            {//n
                {PixelType.EMPTY, PixelType.EMPTY,  PixelType.FILLED}, 
                {PixelType.EMPTY, PixelType.CENTER, PixelType.FILLED}, 
                {PixelType.EMPTY, PixelType.EMPTY,  PixelType.FILLED}
            }
        };

        private static double[,] a = null;

        private static int w = 0;
        private static int h = 0;

        private static double GetPixel(double[,] array, int x, int y)
        {
            return (x < 0 || y < 0 || x >= w || y >= h) ?
                WHITE :
                array[h - 1 - y, x] > 128.0 ?
                    WHITE :
                    BLACK;
        }

        private static void SetPixel(double[,] array, int x, int y, double value)
        {
            if (x < 0 || y < 0 || x >= w || y >= h) return;
            array[h - 1 - y, x] = value;
        }

        private static bool AreEqual(double value, PixelType patternPixel)
        {
            switch (patternPixel)
            {
                case PixelType.FILLED:
                    {
                        if (value == BLACK)
                            return true;
                        break;
                    }
                case PixelType.EMPTY:
                    {
                        if (value == WHITE)
                            return true;
                        break;
                    }
                case PixelType.AT_LEAST_ONE_EMPTY://y
                    return true;
                case PixelType.CENTER://c
                    if (value == BLACK)
                        return true;
                    break;
                case PixelType.ANY://x
                    return true;
                default:
                    break;
            }
            return false;
        }

        //-1 - no match
        private static int MatchPattern(int x, int y)
        {
            if (GetPixel(a, x, y) == WHITE) return -1;
            for (int i = 0; i < 14; i++)
            {
                bool yInPattern = false;
                int yCounter = 0;
                bool bad = false;
                for (int dX = -1; dX < 2; dX++)
                {
                    if (bad)
                        break;
                    for (int dY = -1; dY < 2; dY++)
                    {
                        if (patterns[i, 1 + dX, 1 + dY] == PixelType.AT_LEAST_ONE_EMPTY)
                        {
                            yInPattern = true;
                            yCounter += GetPixel(a, x + dX, y + dY) == WHITE ? 1 : 0;
                            continue;
                        }
                        if (!AreEqual(GetPixel(a, x + dX, y + dY), patterns[i, 1 + dX, 1 + dY]))
                        {
                            bad = true;
                            break;
                        }
                    }
                }
                if (bad)
                    continue;
                if (yInPattern && yCounter == 0)
                    continue;
                if (i == 2 && !AreEqual(GetPixel(a, x, y + 2), PixelType.FILLED))
                {
                    continue;
                }
                else if (i == 3 && !AreEqual(GetPixel(a, x + 2, y), PixelType.FILLED))
                {
                    continue;
                }
                return i;
            }
            return -1;
        }

        public static double[,] Thin(double[,] array, int width, int height)
        {
            w = width;
            h = height;
            a = new double[h, w];
            Array.Copy(array, 0, a, 0, h * w);

            bool isSkeleton;
            double[,] buffer = new double[h, w];
            Array.Copy(a, 0, buffer, 0, h * w);
            do
            {
                isSkeleton = true;
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        int pattern = MatchPattern(x, y);
                        if (pattern != -1)
                        {
                            SetPixel(buffer, x, y, WHITE);
                            isSkeleton = false;
                        }
                    }
                }
                Array.Copy(buffer, 0, a, 0, h * w);
            } while (!isSkeleton);
            a = null;
            return buffer;
        }
    }
}
