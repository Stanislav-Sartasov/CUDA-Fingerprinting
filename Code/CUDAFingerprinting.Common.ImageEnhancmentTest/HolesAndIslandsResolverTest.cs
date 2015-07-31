using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common;
using System.IO;
using System.Diagnostics;
using CUDAFingerprinting.ImageEnhancement;

namespace ImageEnhancmentTest
{
    [TestClass]
    public class HolesAndIslandsResolverTest
    {
        public static int[] Binarization(int[] a)
        {
            const int BINARIZATION_BARIER = 128;
            for (int i = 0; i < a.Length; i++)
            {
                a[i] = a[i] > BINARIZATION_BARIER ? 255 : 0;
            }
            return a;
        }

        [TestMethod]
        public void ResolveHolesTest()
        {
            var bmp = Resources.f;
            var img = ImageHelper.LoadImageAsInt(bmp);

            int h = img.GetLength(0);
            int w = img.GetLength(1);

            int[] withoutHoles = HolesAndIslandsResolver.ResolveHoles(
                Binarization(Array2Dto1D(img)), 
                16, 
                w, h);
            var enhanced = ImageHelper.SaveArrayToBitmap(
                Array1Dto2D(withoutHoles, w, h)
            );
            var name = Path.GetTempPath() + bmp.GetHashCode().ToString() + "WithoutHoles.bmp";
            enhanced.Save(name, ImageHelper.GetImageFormatFromExtension(name));
            Process.Start(name);
        }

        [TestMethod]
        public void ResolveIslandsTest()
        {
            var bmp = Resources.islands;
            var img = ImageHelper.LoadImageAsInt(bmp);

            int h = img.GetLength(0);
            int w = img.GetLength(1);

            int[] withoutIslands = HolesAndIslandsResolver.ResolveIslands(
                Binarization(Array2Dto1D(img)),
                9,
                w, h);
            var enhanced = ImageHelper.SaveArrayToBitmap(
                Array1Dto2D(withoutIslands, w, h)
            );
            var name = Path.GetTempPath() + bmp.GetHashCode().ToString() + "WithoutIslands.bmp";
            enhanced.Save(name, ImageHelper.GetImageFormatFromExtension(name));
            Process.Start(name);
        }

        private static int[] Array2Dto1D(int[,] data)
        {
            int h = data.GetLength(0);
            int w = data.GetLength(1);
            int[] result = new int[h * w];
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    result[(h - 1 - y) * w + x] = data[h - 1 - y, x];
                }
            }
            return result;
        }

        private static int[,] Array1Dto2D(int[] data, int w, int h)
        {
            int[,] result = new int[h, w];
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    result[(h - 1 - y), x] = data[(h - 1 - y) * w + x];
                }
            }
            return result;
        }
    }
}
