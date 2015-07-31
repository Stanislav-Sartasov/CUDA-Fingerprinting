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
        public static int[] Binarization(int[] a, int BINARIZATION_BARIER)
        {
            for (int i = 0; i < a.Length; i++)
            {
                a[i] = a[i] > BINARIZATION_BARIER ? 255 : 0;
            }
            return a;
        }

        public static int[] OverlapArrays(int[] data, int[] background)
        {
            int[] result = new int[data.Length];
            for (int i = 0; i < result.Length; i++)
            {
                if (data[i] == background[i])
                {
                    result[i] = data[i] == 0 ? 128 : 255;
                }
                else
                {
                    result[i] = data[i] == 0 ? 190 : 0;
                }
            }
            return result;
        }

        [TestMethod]
        public void ResolveHolesTest()
        {
            var bmp = Resources.holes;
            var img = ImageHelper.LoadImageAsInt(bmp);

            int h = img.GetLength(0);
            int w = img.GetLength(1);

            int[] withoutHoles = HolesAndIslandsResolver.ResolveHoles(
                Binarization(Array2Dto1D(img), 128), 
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
                Binarization(Array2Dto1D(img), 128),
                9,
                w, h);
            var enhanced = ImageHelper.SaveArrayToBitmap(
                Array1Dto2D(withoutIslands, w, h)
            );
            var name = Path.GetTempPath() + bmp.GetHashCode().ToString() + "WithoutIslands.bmp";
            enhanced.Save(name, ImageHelper.GetImageFormatFromExtension(name));
            Process.Start(name);
        }

        [TestMethod]
        public void ResolveHolesAndIslandsTest()
        {
            var bmp = Resources.SampleFinger;
            var img = ImageHelper.LoadImageAsInt(bmp);

            int h = img.GetLength(0);
            int w = img.GetLength(1);
            
            int[] withoutHolesAndIslands = HolesAndIslandsResolver.ResolveHolesAndIslands(
                Binarization(Array2Dto1D(img), 128),
                16,
                9,
                w, h);
            
            var enhanced = ImageHelper.SaveArrayToBitmap(
                Array1Dto2D(
                    OverlapArrays(
                        Binarization(Array2Dto1D(img), 128),
                        withoutHolesAndIslands
                    ),
                    w,
                    h
                )
            );
            var name = Path.GetTempPath() + bmp.GetHashCode().ToString() + "WithoutHolesAndIslands.bmp";
            enhanced.Save(name, ImageHelper.GetImageFormatFromExtension(name));
            Process.Start(name);
        }
        /*
        [TestMethod]
        public void ResolveHolesAndIslandsJOKETest()
        {
            var bmp = Resources.SampleFinger;
            var img = ImageHelper.LoadImageAsInt(bmp);

            int h = img.GetLength(0);
            int w = img.GetLength(1);
            
            int[] withoutHolesAndIslands = HolesAndIslandsResolver.ResolveHolesAndIslandsJOKE(
                Binarization(Array2Dto1D(img), 128),
                16,
                9,
                w, h);

            var enhanced = ImageHelper.SaveArrayToBitmap(
                Array1Dto2D(withoutHolesAndIslands, w, h)
            );
            var name = Path.GetTempPath() + bmp.GetHashCode().ToString() + "WithoutHolesAndIslandsJOKE.bmp";
            enhanced.Save(name, ImageHelper.GetImageFormatFromExtension(name));
            Process.Start(name);
        }

        [TestMethod]
        public void ResolveHolesAndIslandsJOKEcomparationTest()
        {
            var bmp = Resources.SampleFinger;
            var img = ImageHelper.LoadImageAsInt(bmp);

            int h = img.GetLength(0);
            int w = img.GetLength(1);

            int[] withoutHolesAndIslands = HolesAndIslandsResolver.ResolveHolesAndIslands(
                Binarization(Array2Dto1D(img), 128),
                16,
                9,
                w, h);

            int[] withoutHolesAndIslandsJOKE = HolesAndIslandsResolver.ResolveHolesAndIslandsJOKE(
                Binarization(Array2Dto1D(img), 128),
                16,
                9,
                w, h);

            var comparationAND = ImageHelper.SaveArrayToBitmap(
                Array1Dto2D(
                    OverlapAND(withoutHolesAndIslands, 
                        withoutHolesAndIslandsJOKE), 
                    w, h)
            );

            var comparationOR = ImageHelper.SaveArrayToBitmap(
                Array1Dto2D(
                    OverlapOR(withoutHolesAndIslands,
                        withoutHolesAndIslandsJOKE),
                    w, h)
            );

            var name = Path.GetTempPath() + bmp.GetHashCode().ToString() + "compareJOKEand.bmp";
            comparationAND.Save(name, ImageHelper.GetImageFormatFromExtension(name));
            Process.Start(name);

            name = Path.GetTempPath() + bmp.GetHashCode().ToString() + "compareJOKEor.bmp";
            comparationOR.Save(name, ImageHelper.GetImageFormatFromExtension(name));
            Process.Start(name);
        }

        private static int[] OverlapAND(int[] data, int[] background)
        {
            int[] result = new int[data.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = data[i] + background[i] == 0 ? 0 : 255;
            }
            return result;
        }

        private static int[] OverlapOR(int[] data, int[] background)
        {
            int[] result = new int[data.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = data[i] + background[i] < 400 ? 0 : 255;
            }
            return result;
        }
        */
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
