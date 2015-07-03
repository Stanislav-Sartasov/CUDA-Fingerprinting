using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CUDAFingerprinting.Common.Thinning
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
       
        private static PixelType[,,] patterns = new PixelType[,,] 
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
        //private static bool[,] concaveCornerCandidats = null;
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
        /*
        private static bool IsMarked(int x, int y)
        {
            if (x < 0 || y < 0 || x >= w || y >= h) return false;
            return concaveCornerCandidats[h - 1 - y, x];
        }

        private static void InvertMark(int x, int y)
        {
            if (!(x < 0 || y < 0 || x >= w || y >= h))
            {
                concaveCornerCandidats[h - 1 - y, x] = !concaveCornerCandidats[h - 1 - y, x];
            }
        }
        */
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
        /*
        private static bool IsConcaveCornerCandidate(int xC, int yC, int dX, int dY, int pattern)
        {
            if (dX == 0 && dY == -1)//p0
            {
                return (pattern == 0 && GetPixel(a, xC + 1, yC - 1) == BLACK) ||
                    (pattern == 5 && GetPixel(a, xC - 1, yC - 1) == BLACK) ||
                    (pattern == 7 && 
                        GetPixel(a, xC - 1, yC - 1) == BLACK &&
                        GetPixel(a, xC + 1, yC - 1) == BLACK);
            }
            else if (dX == 1 && dY == 0)//p2
            {
                return (pattern == 1 && GetPixel(a, xC + 1, yC + 1) == BLACK) ||
                    (pattern == 2 && GetPixel(a, xC - 1, yC - 1) == BLACK) ||
                    (pattern == 3 && GetPixel(a, xC + 1, yC - 1) == BLACK) ||
                    (pattern == 5 && GetPixel(a, xC - 1, yC + 1) == BLACK) ||
                    (pattern == 8 && GetPixel(a, xC + 1, yC - 1) == BLACK);
            }
            else if (dX == 0 && dY == 1)//p4
            {
                return (pattern == 0 && GetPixel(a, xC + 1, yC + 1) == BLACK) ||
                    (pattern == 2 && GetPixel(a, xC - 1, yC + 1) == BLACK) ||
                    (pattern == 4 &&
                        GetPixel(a, xC - 1, yC + 1) == BLACK &&
                        GetPixel(a, xC + 1, yC + 1) == BLACK) ||
                    (pattern == 8 && GetPixel(a, xC - 1, yC + 1) == BLACK);
            }
            else if (dX == -1 && dY == 0)//p6
            {
                return (pattern == 1 && GetPixel(a, xC - 1, yC + 1) == BLACK) ||
                    (pattern == 3 && GetPixel(a, xC - 1, yC - 1) == BLACK) ||
                    (pattern == 4 && 
                        GetPixel(a, xC - 1, yC + 1) == BLACK &&
                        GetPixel(a, xC - 1, yC - 1) == BLACK) ||
                    (pattern == 7 &&
                        GetPixel(a, xC - 1, yC - 1) == BLACK &&
                        GetPixel(a, xC - 1, yC + 1) == BLACK);
            }
            return false;
        }

        private static bool IsConcaveCornerPattern(int x, int y)
        {
            PixelType[,,] concavePatterns = new PixelType[,,]
            {
                {
                    {PixelType.ANY,    PixelType.FILLED, PixelType.ANY,    PixelType.FILLED},
                    {PixelType.FILLED, PixelType.ANY,    PixelType.FILLED, PixelType.FILLED},
                    {PixelType.ANY,    PixelType.FILLED, PixelType.CENTER, PixelType.FILLED},
                    {PixelType.FILLED, PixelType.FILLED, PixelType.FILLED, PixelType.EMPTY}
                },
                {
                    {PixelType.FILLED, PixelType.ANY,    PixelType.FILLED, PixelType.ANY},
                    {PixelType.FILLED, PixelType.FILLED, PixelType.ANY,    PixelType.FILLED},
                    {PixelType.FILLED, PixelType.CENTER, PixelType.FILLED, PixelType.ANY},
                    {PixelType.EMPTY,  PixelType.FILLED, PixelType.FILLED, PixelType.FILLED}
                },
                {
                    {PixelType.FILLED, PixelType.FILLED, PixelType.FILLED, PixelType.EMPTY},
                    {PixelType.ANY,    PixelType.FILLED, PixelType.CENTER, PixelType.FILLED},
                    {PixelType.FILLED, PixelType.ANY,    PixelType.FILLED, PixelType.FILLED},
                    {PixelType.ANY,    PixelType.FILLED, PixelType.ANY,    PixelType.FILLED}
                },
                {
                    {PixelType.EMPTY,  PixelType.FILLED, PixelType.FILLED, PixelType.FILLED},
                    {PixelType.FILLED, PixelType.CENTER, PixelType.FILLED, PixelType.ANY},
                    {PixelType.FILLED, PixelType.FILLED, PixelType.ANY,    PixelType.FILLED},
                    {PixelType.FILLED, PixelType.ANY,    PixelType.FILLED, PixelType.ANY}
                }
            };
            for (int p = 0; p < 4; p++)
            {
                bool bad = false;
                for (int i = -2 + (p / 2); i < 2 + (p / 2); i++)
                {
                    for (int j = -2 + (p % 2); j < 2 + (p % 2); j++)
                    {
                        if (!AreEqual(GetPixel(a, x + j, y + i),
                            concavePatterns[p, 2 - (p / 2) + i, 2 - (p % 2) + j]))
                        {
                            bad = true;
                            break;
                        }
                    }
                    if (bad)
                        break;
                }
                if (!bad)
                    return true;
            }
            return false;
        }
        */
        public static double[,] Thin(double[,] array, int width, int height)
        {
            w = width;
            h = height;
            a = new double[h, w];
            Array.Copy(array, 0, a, 0, h * w);
            //concaveCornerCandidats = new bool[h, w];
            //Array.Clear(concaveCornerCandidats, 0, h * w);

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
                            /*neighbors-concave corners processing
                            for (int i = -1; i < 2; i++)
                            {
                                for (int j = -1; j < 2; j++)
                                {
                                    if (i == 0 && j == 0) continue;
                                    if (IsConcaveCornerCandidate(x, y, i, j, pattern))
                                    {
                                        if (IsMarked(x + i, y + j))
                                        {
                                            if (IsConcaveCornerPattern(x + i, y + j))
                                            {
                                                SetPixel(buffer, x + i, y + j, WHITE);
                                            }
                                        }
                                        InvertMark(x + i, y + j);
                                    }
                                }
                            }
                            end concave corners processing*/
                        }
                    }
                }
                Array.Copy(buffer, 0, a, 0, h * w);
            } while (!isSkeleton);
            a = null;
            //concaveCornerCandidats = null;
            return buffer;
        }
    }
}
