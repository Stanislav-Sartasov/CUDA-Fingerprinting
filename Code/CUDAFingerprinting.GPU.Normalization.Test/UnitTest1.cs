using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using System.Text;
using System.Drawing;
using System.Runtime.InteropServices;
using CUDAFingerprinting.Common;


namespace CUDAFingerprinting.GPU.Normalization.Test
{
    [TestClass]
    public class UnitTest1
    {
        public static float[,] LoadImage(Bitmap bmp)
        {
            float[,] imgBytes = new float[bmp.Height, bmp.Width];
            for (int x = 0; x < bmp.Width; x++)
            {
                for (int y = 0; y < bmp.Height; y++)
                {
                    imgBytes[bmp.Height - 1 - y, x] = bmp.GetPixel(x, y).R;
                }
            }
            return imgBytes;
        }

        public static Bitmap SaveArrayToBitmap(float[,] data)
        {
            int x = data.GetLength(1);
            int y = data.GetLength(0);
            var max = float.NegativeInfinity;
            var min = float.PositiveInfinity;
            foreach (var num in data)
            {
                if (num > max) max = num;
                if (num < min) min = num;
            }
            var bmp = new Bitmap(x, y);
            data.Select2D((value, row, column) =>
            {
                var gray = (int)((value - min) / (max - min) * 255);
                lock (bmp)
                    bmp.SetPixel(column, bmp.Height - 1 - row, Color.FromArgb(gray, gray, gray));
                return value;
            });
            return bmp;
        }

        [DllImport("CUDAFingerprinting.GPU.Normalisation.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "Normalize")]
        public static extern IntPtr Normalize(float[,] source, int imgWidth, int imgHeight, int bordMean, int bordVar);

        [TestMethod]
        public void NormalizationTest()
        {
            var bmp = Resources.SimpleFinger1;
            float[,] array = LoadImage(bmp);
            IntPtr cur = Normalize(array, bmp.Width, bmp.Height, 1000, 1000);
            int[] result = new int[bmp.Width * bmp.Height];
            Marshal.Copy(cur, result, 0, 3);
            var bmp2 = SaveArrayToBitmap(array);

            bmp2.Save("test.bmp", ImageHelper.GetImageFormatFromExtension("test.bmp"));
        }
        [TestMethod]
        public void NormalizationTest2()
        {
            var bmp = Resources.SimpleFinger1;
            float[,] array = LoadImage(bmp);
            IntPtr cur = Normalize(array, bmp.Width, bmp.Height, 1000, 1000);
            int[] result = new int[bmp.Width * bmp.Height];
            Marshal.Copy(cur, result, 0, 3);
            var bmp2 = SaveArrayToBitmap(array);

            bmp2.Save("test.bmp", ImageHelper.GetImageFormatFromExtension("test.bmp"));
        }
    }
}
