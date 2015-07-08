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
        public static T[] Make1D<T>(T[,] arr)
        {
            var rows = arr.GetLength(0);
            var columns = arr.GetLength(1);

            var result = new T[rows * columns];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    result[i * columns + j] = arr[i, j];
                }
            }
            return result;
        }
        [DllImport("CUDAFingerprinting.GPU.Normalization.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "Normalize")]
       // [return: MarshalAs(UnmanagedType.SafeArray)]
        public static extern void Normalize(float[] source, float[] res, int imgWidth, int imgHeight, int bordMean, int bordVar);
        //static void WriteFloat(IntPtr ptr, float value)
        //{
        //    foreach (var b in BitConverter.GetBytes(value))
        //    {
        //        Marshal.WriteByte(ptr, b);
        //        ptr += 1;
        //    }
        //}

        [TestMethod]
        public void NormalizationTest()
        {
            var bmp = Resources.SampleFinger1;
            var array0 = LoadImage(bmp);
            var array = Make1D(array0);
            float[] result = new float[bmp.Width * bmp.Height];
            Normalize(array, result, bmp.Width, bmp.Height, 100, 1000);
            //IntPtr ptr = Normalize(array, bmp.Width, bmp.Height, 1000, 1000);
            //float[] result = new float[bmp.Width * bmp.Height];
            //float a = (float)1.0;
            //WriteFloat(ptr, a);
            //Marshal.Copy(ptr, result, 0, bmp.Width * bmp.Height);
            //array = Normalize(array, bmp.Width, bmp.Height, 1000, 1000);
            float[,] ar = result.Make2D(bmp.Width, bmp.Height);
            var bmp2 = SaveArrayToBitmap(ar);

            bmp2.Save("test.bmp", ImageHelper.GetImageFormatFromExtension("test.bmp"));
        }
    }
}
