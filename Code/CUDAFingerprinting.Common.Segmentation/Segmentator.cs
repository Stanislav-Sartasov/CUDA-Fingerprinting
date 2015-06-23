using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;
using CUDAFingerprinting.Common.OrientationField;

namespace CUDAFingerprinting.Common.Segmentation
{
    //class Programm
    //{
    //    public static int[,] Normalize(int[,] arr)
    //    {
    //        int xLength = arr.GetLength(0);
    //        int yLength = arr.GetLength(1);

    //        for (int i = 0; i < xLength; i++)
    //        {
    //            for (int j = 0; j < yLength; j++)
    //            {
    //                arr[i, j] = arr[i, j] == 1 ? 240 : 0;
    //            }
    //        }

    //        return arr;
    //    }
    //    static void Main(string[] args)
    //    {

    //        int windowSize = 12;
    //        double weight = 0.3;
    //        int threshold = 5;


    //        double[,] img1 = ImageHelper.LoadImage("D:/103_7.tif");
    //        int[,] resultImg1;

    //        resultImg1 = Segmentator.Segmetator(img1, 5, weight, threshold);
    //        ImageHelper.SaveIntArray(Normalize(resultImg1), Path.GetTempPath() + "Segm_104_6_5" + ".png");
    //        resultImg1 = Segmentator.Segmetator(img1, 6, weight, threshold);
    //        ImageHelper.SaveIntArray(Normalize(resultImg1), Path.GetTempPath() + "Segm_104_6_6" + ".png");
    //        resultImg1 = Segmentator.Segmetator(img1, 7, weight, threshold);
    //        ImageHelper.SaveIntArray(Normalize(resultImg1), Path.GetTempPath() + "Segm_104_6_7" + ".png");
    //        resultImg1 = Segmentator.Segmetator(img1, 8, weight, threshold);
    //        ImageHelper.SaveIntArray(Normalize(resultImg1), Path.GetTempPath() + "Segm_104_6_8" + ".png");
    //        resultImg1 = Segmentator.Segmetator(img1, 9, weight, threshold);
    //        ImageHelper.SaveIntArray(Normalize(resultImg1), Path.GetTempPath() + "Segm_104_6_9" + ".png");
    //        resultImg1 = Segmentator.Segmetator(img1, 10, weight, threshold);
    //        ImageHelper.SaveIntArray(Normalize(resultImg1), Path.GetTempPath() + "Segm_104_6_10" + ".png");
    //        resultImg1 = Segmentator.Segmetator(img1, 11, weight, threshold);
    //        ImageHelper.SaveIntArray(Normalize(resultImg1), Path.GetTempPath() + "Segm_104_6_11" + ".png");
    //        resultImg1 = Segmentator.Segmetator(img1, 12, weight, threshold);
    //        ImageHelper.SaveIntArray(Normalize(resultImg1), Path.GetTempPath() + "Segm_104_6_12" + ".png");
    //        resultImg1 = Segmentator.Segmetator(img1, 13, weight, threshold);
    //        ImageHelper.SaveIntArray(Normalize(resultImg1), Path.GetTempPath() + "Segm_104_6_13" + ".png");
    ////        ImageHelper.SaveArray(resultImg1, Path.GetTempPath() + "Segm_104_6" + ".png");

    //    }
    //}
    public static class Segmentator
    {
        private static int N;
        private static int M;
        private static bool[,] mask = new bool[N, M];

        //public static bool[,] GetMask(int[] mask1D, int maskY, int imgX, int imgY, int windowSize)
        //{
        //    bool[,] bigMask = new bool[imgX, imgY];

        //    bigMask = bigMask.Select2D((value, x, y) =>
        //        {
        //            int xBlock = (int)(((double)x) / windowSize);
        //            int yBlock = (int)(((double)y) / windowSize);
        //            return mask1D[xBlock + yBlock * maskY] == 1;
        //        });

        //    return bigMask;
        //}

        public static int[,] Segmetator(double[,] img, int windowSize, double weight, int threshold)
        {
            int[,] xGradients = OrientationFieldGenerator.GenerateXGradients(img.Select2D(a => (int)a));
            int[,] yGradients = OrientationFieldGenerator.GenerateYGradients(img.Select2D(a => (int)a));
            double[,] magnitudes =
                xGradients.Select2D(
                    (value, x, y) => Math.Sqrt(xGradients[x, y] * xGradients[x, y] + yGradients[x, y] * yGradients[x, y]));
            double averege = KernelHelper.Average(magnitudes);
            double[,] window = new double[windowSize, windowSize];

            N = (int)Math.Ceiling(((double)img.GetLength(0)) / windowSize);
            M = (int)Math.Ceiling(((double)img.GetLength(1)) / windowSize);

            int[,] mask = new int[N, M];

            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < M; j++)
                {
                    window = window.Select2D((value, x, y) =>
                    {
                        if (i * windowSize + x >= magnitudes.GetLength(0)
                            || j * windowSize + y >= magnitudes.GetLength(1))
                        {
                            return 0;
                        }

                        return magnitudes[(int)(i * windowSize + x), (int)(j * windowSize + y)];
                    });

                    if (KernelHelper.Average(window) < averege * weight)
                    {
                        mask[i, j] = 0;
                    }
                    else
                    {
                        mask[i, j] = 1;
                    }
                }
            }

            PostProcessing(mask, threshold);

            return mask; // GetBigMask(mask, img.GetLength(0), img.GetLength(1), windowSize);
            //return ColorImage(img, mask, windowSize);
        }

        public static int[,] GetBigMask(int[,] mask, int imgX, int imgY, int windowSize)
        {
            int[,] bigMask = new int[imgX, imgY];

            bigMask = bigMask.Select2D((value, x, y) =>
            {
                int xBlock = (int)(((double)x) / windowSize);
                int yBlock = (int)(((double)y) / windowSize);
                return mask[xBlock , yBlock];
            });

            return bigMask;
        }

        // mask is supposed to be the size of the image
        public static double[,] ColorImage(double[,] img, int[,] mask)
        {
            return img.Select2D((value, x, y) => mask[x, y]>0 ? img[x, y] : 255);
        }

        public static float[] ColorImage(float[] img, int rows, int columns, int[,] mask)
        {
            return img.Select2D(rows, columns, (value, x, y) => mask[x, y] > 0 ? img[x*columns + y] : 255);
        }

        public static void PostProcessing(int[,] mask, int threshold)
        {
            var blackAreas = GenerateAreas(mask,true);

            foreach (
                var blackArea in
                    blackAreas.Where(x => x.Count < threshold && !IsNearBorder(x, mask.GetLength(0), mask.GetLength(1)))
                )
            {
                {
                    foreach (var point in blackArea)
                    {
                        mask[point.X, point.Y] = 1;
                    }
                }
            }
            
            var imageAreas = GenerateAreas(mask, false);
            
            foreach (var imageArea in imageAreas.Where(x => x.Count < threshold && !IsNearBorder(x, mask.GetLength(0), mask.GetLength(1))))
            {
                foreach (var point in imageArea)
                {
                    mask[point.X, point.Y] = 0;
                }
            }
        }

        private static List<List<Point>> GenerateAreas(int[,] mask, bool black)
        {
            var lengthX = mask.GetLength(0);
            var lengthY = mask.GetLength(1);

            List<Point>[,] bucketsMap = new List<Point>[mask.GetLength(0), mask.GetLength(1)];

            List<List<Point>> buckets = new List<List<Point>>();

            for (int x = 0; x < lengthX; x++)
            {
                for (int y = 0; y < lengthY; y++)
                {
                    if (black?mask[x, y]==0:mask[x,y]>0)
                    {
                        var validTop = x > 0 && (black ? mask[x - 1, y]==0 : mask[x - 1, y]>0);
                        var validLeft = y > 0 && (black ? mask[x, y - 1]==0 : mask[x, y - 1]>0);

                        if (!validLeft && !validTop)
                        {
                            buckets.Add(new List<Point>());
                            bucketsMap[x, y] = buckets.Last();
                        }
                        else if (validLeft ^ validTop)
                        {
                            if (validLeft)
                            {
                                bucketsMap[x, y] = bucketsMap[x, y - 1];
                            }
                            else
                            {
                                bucketsMap[x, y] = bucketsMap[x - 1, y];
                            }
                        }
                        else //both true
                        {
                            if (bucketsMap[x - 1, y] == bucketsMap[x, y - 1])
                            {
                                bucketsMap[x, y] = bucketsMap[x - 1, y];
                            }
                            else
                            {
                                var leftBucket = bucketsMap[x - 1, y];
                                var topBucket = bucketsMap[x , y-1];

                                foreach (var point in leftBucket)
                                {
                                    bucketsMap[point.X, point.Y] = topBucket;
                                }
                                buckets.Remove(leftBucket);
                                topBucket.AddRange(leftBucket);

                                bucketsMap[x, y] = topBucket;

                            }
                        }
                        bucketsMap[x, y].Add(new Point(x, y));
                    }
                }
            }
            return buckets;
        }

        private static bool IsNearBorder(List<Point> areas, int xBorder, int yBorder)
        {
            return areas.Any(area => area.X == 0 ||
                                         area.Y == 0 ||
                                         area.X == xBorder ||
                                         area.Y == yBorder);
        }
    }
}













