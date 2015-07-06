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
        [return: MarshalAs(UnmanagedType.SafeArray)]
        public static extern float[,] Normalize(float[,] source, int imgWidth, int imgHeight, int bordMean, int bordVar);

        [TestMethod]
        public void NormalizationTest()
        {
            var bmp = Resources.SimpleFinger1;
            var array = LoadImage(bmp);

            

            array = Normalize(array, bmp.Width, bmp.Height, 1000, 1000);

            var bmp2 = SaveArrayToBitmap(array);

            bmp.Save("test.bmp", ImageHelper.GetImageFormatFromExtension("test.bmp"));
        }
    }
}
